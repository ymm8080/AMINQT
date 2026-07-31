"""MAIN: Compare 5000 vs 3200 feature pool prediction quality."""
import warnings, time, re, json
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
np.random.seed(42)
import lightgbm as lgb

t_total = time.time()

# Load & clean
print('='*60)
print('5000 vs 3200 FEATURE POOL: MAIN BOARD')
print('='*60)
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
# Use full 3-year data
with open('data/csi300_constituents.json') as f: csi300 = set(json.load(f))
panel = panel[panel['symbol'].isin(csi300 & set(panel['symbol'].unique()))]
from app.pipeline1.cleaning_pipeline import CleaningPipeline
main_df, _ = CleaningPipeline().run_train(panel)
print(f'Clean: {len(main_df):,} rows, {main_df.symbol.nunique()} stocks')

from app.pipeline1.label_engine import LabelEngine
df = LabelEngine.build_path_labels(main_df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=3)
LABEL = 'label_pm_1d_net'
if LABEL not in df.columns: LABEL = 'label_1d_net'
print(f'Label: {LABEL}')

ID_COLS = {'symbol','date','board','industry','announce_date','is_suspended','is_st','tradestatus'}

# Identify all numeric raw columns
ALL_NUMERIC = [c for c in df.columns if c not in ID_COLS and not c.startswith('label_') and not c.startswith('dim') and df[c].dtype in ('float64','int64')]
OHLCV_12 = [c for c in ['open','high','low','close','volume','amount','open_hfq','high_hfq','low_hfq','close_hfq','turnover_rate','pre_close'] if c in df.columns]
print(f'ALL_NUMERIC: {len(ALL_NUMERIC)} cols')
print(f'OHLCV_12: {len(OHLCV_12)} cols')

def gen_brute(df, raw_cols, label_prefix):
    t0 = time.time()
    all_new = {}
    for sym, g in df.groupby('symbol'):
        g = g.sort_values('date'); feats = {}
        for col in raw_cols:
            if col not in g.columns: continue
            s = g[col].astype(float).values; n = len(s)
            for w in [1,2,3,5,10,20,40,60]:
                o=np.full(n,np.nan); d=np.abs(s[:-w]); o[w:]=np.divide((s[w:]-s[:-w])*100,d,out=np.zeros_like(d),where=d>0)
                feats[f'{col}_pct{w}']=o
            for w in [5,10,20,40,60]:
                feats[f'{col}_ma{w}']=pd.Series(s).rolling(w,min_periods=1).mean().values
            for w in [5,10,20,40]:
                feats[f'{col}_std{w}']=pd.Series(s).rolling(w,min_periods=1).std().values
            for w in [10,20,40]:
                r=pd.Series(s).rolling(w,min_periods=1)
                feats[f'{col}_max{w}']=r.max().values; feats[f'{col}_min{w}']=r.min().values
            for w in [1,5,20]:
                o=np.full(n,np.nan); o[w:]=s[w:]-s[:-w]; feats[f'{col}_d{w}']=o
            for w in [5,20,40]:
                o=np.full(n,np.nan); d=np.abs(s[:-w]); o[w:]=np.divide(s[w:],d,out=np.zeros_like(d),where=d>0); feats[f'{col}_mom{w}']=o
            for w in [5,20,40]:
                feats[f'{col}_ema{w}']=pd.Series(s).ewm(span=w,min_periods=1).mean().values
        all_new[sym]=pd.DataFrame(feats,index=g.index)
    new = pd.concat(all_new.values())
    print(f'  {label_prefix}: Generated {len(new.columns)} features ({time.time()-t0:.0f}s)')
    return new

# Generate both pools
pool_32k = gen_brute(df, OHLCV_12, 'OHLCV_12')
pool_5k = gen_brute(df, ALL_NUMERIC, 'ALL_NUMERIC')

def dedup(feats, df_ref, th=0.7):
    groups={}
    for c in feats:
        p=c.split('_pct')[0].split('_ma')[0].split('_std')[0].split('_d')[0].split('_mom')[0].split('_ema')[0].split('_max')[0].split('_min')[0]
        groups.setdefault(p,[]).append(c)
    kept=[]
    for dg,cols in groups.items():
        if len(cols)<=1: kept.extend(cols); continue
        av=[c for c in cols if c in df_ref.columns]
        if len(av)<=1: kept.extend(av); continue
        s=df_ref[av].sample(min(5000,len(df_ref)),random_state=42)
        cm=s.corr().abs(); dropped=set()
        for i,ci in enumerate(av):
            if ci in dropped: continue
            for cj in av[i+1:]:
                if cj in dropped: continue
                if cm.loc[ci,cj]>th: dropped.add(cj)
        kept.extend([c for c in av if c not in dropped])
    return kept

def ev(preds, df_te, lab, label_text):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics = [spearmanr(g['pred'],g[lab])[0] for _,g in df_e.groupby('date') if len(g)>=10]
    a = np.array([x for x in ics if not np.isnan(x)])
    ic = float(round(a.mean(),5))
    icir = float(round(a.mean()/a.std() if a.std()>0 else 0,4))
    pos = sum(1 for x in ics if not np.isnan(x) and x>0)
    dt = [g.nlargest(10,'pred') for _,g in df_e.groupby('date')]
    top = pd.concat(dt); rets = top[lab].dropna()
    sh = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    wr = float((rets>0).mean())
    comp = round(icir*0.4+sh*0.35+wr*0.25,4)
    print(f'  {label_text}: IC={ic:+.5f} ICIR={icir:.4f} Sharpe={sh:.2f} WinRate={wr:.1%} Composite={comp:.4f}')
    return {'IC':ic,'ICIR':icir,'IC_pos':pos,'IC_days':len(ics),'Sharpe':round(sh,4),'WinRate':round(wr,4),'Composite':comp}

# Evaluate both
print()
results = {}
for name, pool_df in [('3200 (OHLCV 12 cols)', pool_32k), ('5000 (ALL ~100 cols)', pool_5k)]:
    print(f'--- {name} ---')
    df_exp = df.join(pool_df)
    all_c = [c for c in df_exp.columns if c not in ID_COLS and not c.startswith('label_') and df_exp[c].dtype in ('float64','int64') and df_exp[c].isna().mean()<0.95]
    print(f'  NaN filter: {len(all_c)} features')

    dates = sorted(df_exp['date'].unique())
    split = int(len(dates)*0.75)
    tr = df_exp[df_exp['date'].isin(dates[:split])].dropna(subset=[LABEL])
    te = df_exp[df_exp['date'].isin(dates[split:])].dropna(subset=[LABEL])

    d = dedup(all_c, tr, 0.7)
    print(f'  Dedup: {len(all_c)} -> {len(d)}')

    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr[d].fillna(0), tr[LABEL])
    r = ev(m.predict(te[d].fillna(0)), te, LABEL, f'FINAL')
    r['Pool_Size'] = len(pool_df.columns)
    r['After_NaN'] = len(all_c)
    r['After_Dedup'] = len(d)
    results[name] = r

# Comparison
print()
print('='*60)
print('COMPARISON')
print('='*60)
print(f'{"Pool":<30} {"Gen":>6} {"NaN":>6} {"Dedup":>6} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Composite":>10}')
print('-'*95)
for name, r in results.items():
    print(f'{name:<30} {r["Pool_Size"]:>6} {r["After_NaN"]:>6} {r["After_Dedup"]:>6} {r["IC"]:>+9.5f} {r["ICIR"]:>7.4f} {r["Sharpe"]:>7.2f} {r["WinRate"]:>8.1%} {r["Composite"]:>10.4f}')

best = max(results.items(), key=lambda x: x[1]['Composite'])
print(f'\nWINNER: {best[0]} (Composite={best[1]["Composite"]:.4f})')
print(f'Total: {time.time()-t_total:.0f}s')
