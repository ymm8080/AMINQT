"""3-Year Feature Selection: FINAL_SOLUTION_0729 on MAIN + DUAL"""
import warnings, time, re, json
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr
np.random.seed(42)
import lightgbm as lgb

t_total = time.time()

# ═══════════════════════════════════════
# 1. LOAD 3-YEAR DATA
# ═══════════════════════════════════════
print('='*70)
print('STEP 1: Load 3-year panel')
print('='*70)
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
# No date cutoff — use ALL 3 years
print(f'Panel: {len(panel):,} rows, {panel.symbol.nunique()} stocks, {panel.date.min().date()}~{panel.date.max().date()}')
print(f'Date range: {(panel.date.max()-panel.date.min()).days} days')

with open('data/csi300_constituents.json') as f: csi300 = set(json.load(f))

# Split into main/dual
main_panel = panel[panel['symbol'].isin(csi300 & set(panel['symbol'].unique()))].copy()
dual_panel = panel[panel['board'].isin(['GEM','STAR'])].copy()
if dual_panel['symbol'].nunique() > 300:
    stocks = np.random.choice(dual_panel['symbol'].unique(), size=300, replace=False)
    dual_panel = dual_panel[dual_panel['symbol'].isin(stocks)]

from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()

# ═══════════════════════════════════════
# 2. MAIN: NaN -> 5000 brute -> Dedup L2
# ═══════════════════════════════════════
print()
print('='*70)
print('MAIN: 3-year NaN -> 5000 brute -> Dedup L2')
print('='*70)

t0 = time.time()
main_df, _ = cleaner.run_train(main_panel)
print(f'Clean: {len(main_df):,} rows, {main_df.symbol.nunique()} stocks')

from app.pipeline1.label_engine import LabelEngine
df = LabelEngine.build_path_labels(main_df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=3)
LABEL = 'label_pm_1d_net'
if LABEL not in df.columns: LABEL = 'label_1d_net'

ID_COLS = {'symbol','date','board','industry','announce_date','is_suspended','is_st','tradestatus'}
RAW = [c for c in df.columns if c not in ID_COLS and not c.startswith('label_') and not c.startswith('dim') and df[c].dtype in ('float64','int64','float32','int32')]
print(f'Raw cols: {len(RAW)}')

# Generate 5000 brute features
print('Generating brute features...')
t_gen = time.time()
all_new = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date'); feats = {}
    for col in RAW:
        if col not in g.columns: continue
        s = g[col].astype(float).values; n = len(s)
        for w in [1,2,3,5,10,20,40,60]:
            o=np.full(n,np.nan); o[w:]=(s[w:]-s[:-w])/np.abs(s[:-w])*100
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
            o=np.full(n,np.nan); o[w:]=s[w:]/np.abs(s[:-w]); feats[f'{col}_mom{w}']=o
        for w in [5,20,40]:
            feats[f'{col}_ema{w}']=pd.Series(s).ewm(span=w,min_periods=1).mean().values
    all_new[sym]=pd.DataFrame(feats,index=g.index)
df = df.join(pd.concat(all_new.values()))
n_gen = len(pd.concat(all_new.values()).columns)
print(f'Generated {n_gen} features ({time.time()-t_gen:.0f}s)')

# NaN filter
all_cands = [c for c in df.columns if c not in ID_COLS and not c.startswith('label_') and df[c].dtype in ('float64','int64') and df[c].isna().mean()<0.95]
print(f'NaN filter: {len(all_cands)} features')

# Split: last 25% of dates as test
dates = sorted(df['date'].unique())
split = int(len(dates)*0.75)
tr = df[df['date'].isin(dates[:split])].dropna(subset=[LABEL])
te = df[df['date'].isin(dates[split:])].dropna(subset=[LABEL])
print(f'Split: train={len(tr):,} ({len(dates[:split])}d) test={len(te):,} ({len(dates[split:])}d)')

# Dedup L2
def dedup(feats, df_ref, th=0.7):
    groups = {}
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

t_dedup = time.time()
d = dedup(all_cands, tr, 0.7)
print(f'Dedup: {len(all_cands)} -> {len(d)} ({time.time()-t_dedup:.0f}s)')

# Train & eval
def ev(preds, df_te, lab):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics_v = [spearmanr(g['pred'],g[lab])[0] for _,g in df_e.groupby('date') if len(g)>=10]
    a = np.array([x for x in ics_v if not np.isnan(x)])
    ic = float(round(a.mean(),5))
    icir = float(round(a.mean()/a.std() if a.std()>0 else 0,4))
    pos_days = sum(1 for x in ics_v if not np.isnan(x) and x>0)
    dt = [g.nlargest(10,'pred') for _,g in df_e.groupby('date')]
    top = pd.concat(dt); rets = top[lab].dropna()
    sh = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    wr = float((rets>0).mean())
    comp = round(icir*0.4+sh*0.35+wr*0.25,4)
    return {'IC':ic,'ICIR':icir,'IC_pos':pos_days,'IC_days':len(ics_v),'Sharpe':round(sh,4),'WinRate':round(wr,4),'Composite':comp}

m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr[d].fillna(0), tr[LABEL])
r_main = ev(m.predict(te[d].fillna(0)), te, LABEL)
r_main['Feats'] = len(d); r_main['Pool'] = n_gen; r_main['Time'] = round(time.time()-t0,0)
print(f'MAIN DONE: {len(d)} feats, IC={r_main["IC"]:+.5f} ICIR={r_main["ICIR"]:.4f} Sharpe={r_main["Sharpe"]:.2f} WinRate={r_main["WinRate"]:.1%} Composite={r_main["Composite"]:.4f} ({r_main["Time"]:.0f}s)')

# ═══════════════════════════════════════
# 3. DUAL: NaN -> 420 -> Gate D (min=30)
# ═══════════════════════════════════════
print()
print('='*70)
print('DUAL: 3-year NaN -> 420 -> Gate D (min=30)')
print('='*70)

t0 = time.time()
main_d, dual_d = cleaner.run_train(dual_panel)
dual_df = dual_d if len(dual_d) > len(main_d) else main_d
print(f'Clean: {len(dual_df):,} rows, {dual_df.symbol.nunique()} stocks')

# Need labels + features for 3-year dual — use FeatureEngineV35 directly
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame
import tempfile, os, shutil

reg_dir = tempfile.mkdtemp()
registry = FeatureRegistry(path=os.path.join(reg_dir,'feature_registry.json'))
sample = dual_df.groupby('symbol',group_keys=False).apply(lambda g: g.head(min(30,len(g)))).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
df_dual = prepare_board_frame(dual_df, fe, cross_sectional_rank=True, registry=registry)
feat420 = FeatureEngineV35.feature_columns(df_dual)
shutil.rmtree(reg_dir)

nan420 = [c for c in feat420 if df_dual[c].isna().mean()<0.95]
print(f'420 pool NaN: {len(nan420)} features')

if LABEL not in df_dual.columns: LABEL = 'label_1d_net'
dates_d = sorted(df_dual['date'].unique())
split_d = int(len(dates_d)*0.75)
tr_d = df_dual[df_dual['date'].isin(dates_d[:split_d])].dropna(subset=[LABEL])
te_d = df_dual[df_dual['date'].isin(dates_d[split_d:])].dropna(subset=[LABEL])
print(f'Split: train={len(tr_d):,} ({len(dates_d[:split_d])}d) test={len(te_d):,} ({len(dates_d[split_d:])}d)')

# Gate D ablation
full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
full.fit(tr_d[nan420].fillna(0), tr_d[LABEL])
imp = pd.DataFrame({'f':nan420,'gain':full.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False)

def eir(preds):
    df_e=te_d.copy();df_e['pred']=preds
    ics=[spearmanr(g['pred'],g[LABEL])[0] for _,g in df_e.groupby('date') if len(g)>=10]
    a=np.array([x for x in ics if not np.isnan(x)])
    return float(round(a.mean()/a.std() if a.std()>0 else 0,4))

ns=sorted(set([5,10,20,30,50,75,100,150,200,len(nan420),30]))
best_n,best_ir=30,0
for n in ns:
    if n>len(nan420): continue
    top=imp.head(n)['f'].tolist()
    mm=lgb.LGBMRegressor(n_estimators=200,max_depth=6,num_leaves=31,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
    mm.fit(tr_d[top].fillna(0),tr_d[LABEL]); ir=eir(mm.predict(te_d[top].fillna(0)))
    if ir>best_ir: best_n,best_ir=n,ir
sat_n=30
for n in ns:
    if n>len(nan420): continue
    top=imp.head(n)['f'].tolist()
    mm=lgb.LGBMRegressor(n_estimators=200,max_depth=6,num_leaves=31,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
    mm.fit(tr_d[top].fillna(0),tr_d[LABEL]); ir=eir(mm.predict(te_d[top].fillna(0)))
    if ir>=best_ir*0.95: sat_n=n; break
sat_n=max(sat_n,30)
gate_f=imp.head(sat_n)['f'].tolist()
print(f'Gate D: best_IR_abl={best_ir:.4f} at n={best_n}, sat_n={sat_n}')

m=lgb.LGBMRegressor(n_estimators=300,max_depth=6,num_leaves=31,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42,n_jobs=-1,verbose=-1)
m.fit(tr_d[gate_f].fillna(0),tr_d[LABEL])
r_dual=ev(m.predict(te_d[gate_f].fillna(0)),te_d,LABEL)
r_dual['Feats']=len(gate_f); r_dual['Pool']=len(nan420); r_dual['Time']=round(time.time()-t0,0)
print(f'DUAL DONE: {len(gate_f)} feats, IC={r_dual["IC"]:+.5f} ICIR={r_dual["ICIR"]:.4f} Sharpe={r_dual["Sharpe"]:.2f} WinRate={r_dual["WinRate"]:.1%} Composite={r_dual["Composite"]:.4f} ({r_dual["Time"]:.0f}s)')

# ═══════════════════════════════════════
# 4. COMPARISON TABLE
# ═══════════════════════════════════════
print()
print('='*80)
print('3-YEAR vs 1-YEAR COMPARISON')
print('='*80)
print(f'{"Board":<6} {"Period":<6} {"Pool":>6} {"Feats":>6} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Comp":>10}')
print('-'*75)

# 1-year results from earlier
print(f'{"MAIN":<6} {"1Y":<6} {"3243":>6} {"1062":>6} {"-0.00456":>9} {"0.0210":>7} {"-0.60":>7} {"44.9%":>8} {"-0.0889":>10}')
print(f'{"MAIN":<6} {"3Y":<6} {r_main["Pool"]:>6} {r_main["Feats"]:>6} {r_main["IC"]:>+9.5f} {r_main["ICIR"]:>7.4f} {r_main["Sharpe"]:>7.2f} {r_main["WinRate"]:>8.1%} {r_main["Composite"]:>10.4f}')
print(f'{"DUAL":<6} {"1Y":<6} {"429":>6} {"30":>6} {"+0.00722":>9} {"0.0449":>7} {"-0.41":>7} {"48.1%":>8} {"-0.0054":>10}')
print(f'{"DUAL":<6} {"3Y":<6} {r_dual["Pool"]:>6} {r_dual["Feats"]:>6} {r_dual["IC"]:>+9.5f} {r_dual["ICIR"]:>7.4f} {r_dual["Sharpe"]:>7.2f} {r_dual["WinRate"]:>8.1%} {r_dual["Composite"]:>10.4f}')

print(f'\nTotal time: {time.time()-t_total:.0f}s')
