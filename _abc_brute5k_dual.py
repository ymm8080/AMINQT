"""DUAL: v3 raw -> brute 5000 -> Gate D (IC pre vs post vs none) vs 420 Gate D"""
import warnings, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata
np.random.seed(42)
import lightgbm as lgb

t_total = time.time()

# 1. Load v3 panel -> filter GEM/STAR
print('Loading v3 panel...')
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
dual = panel[panel['board'].isin(['GEM','STAR'])].copy()
if dual['symbol'].nunique() > 300:
    stocks = np.random.choice(dual['symbol'].unique(), size=300, replace=False)
    dual = dual[dual['symbol'].isin(stocks)]
print(f'Dual: {len(dual)} rows, {dual.symbol.nunique()} stocks')

from app.pipeline1.cleaning_pipeline import CleaningPipeline
main_d, dual_d = CleaningPipeline().run_train(dual)
df = dual_d if len(dual_d) > len(main_d) else main_d
print(f'Clean: {len(df)} rows, {df.symbol.nunique()} stocks, {len(df.columns)} cols')

# 2. Labels
from app.pipeline1.label_engine import LabelEngine
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=3)
LABEL = 'label_pm_1d_net'
if LABEL not in df.columns: LABEL = 'label_1d_net'
print(f'Label: {LABEL}')

# 3. Raw base
ID_COLS = {'symbol','date','board','industry','announce_date','is_suspended','is_st','tradestatus'}
RAW = [c for c in df.columns if c not in ID_COLS and not c.startswith('label_')
       and not c.startswith('dim') and df[c].dtype in ('float64','int64','float32','int32')]
print(f'Raw base: {len(RAW)} cols')

# 4. Generate brute-force features
print('Generating brute features...')
t0 = time.time()
all_new = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date')
    feats = {}
    for col in RAW:
        if col not in g.columns: continue
        s = g[col].astype(float).values; n = len(s)
        for w in [1,2,3,5,10,20,40,60]:
            out = np.full(n, np.nan); out[w:] = (s[w:]-s[:-w])/np.abs(s[:-w])*100
            feats[f'{col}_pct{w}'] = out
        for w in [5,10,20,40,60]:
            feats[f'{col}_ma{w}'] = pd.Series(s).rolling(w,min_periods=1).mean().values
        for w in [5,10,20,40]:
            feats[f'{col}_std{w}'] = pd.Series(s).rolling(w,min_periods=1).std().values
        for w in [10,20,40]:
            r = pd.Series(s).rolling(w,min_periods=1)
            feats[f'{col}_max{w}'] = r.max().values
            feats[f'{col}_min{w}'] = r.min().values
        for w in [1,5,20]:
            out = np.full(n, np.nan); out[w:] = s[w:]-s[:-w]
            feats[f'{col}_d{w}'] = out
        for w in [5,20,40]:
            out = np.full(n, np.nan); out[w:] = s[w:]/np.abs(s[:-w])
            feats[f'{col}_mom{w}'] = out
        for w in [5,20,40]:
            feats[f'{col}_ema{w}'] = pd.Series(s).ewm(span=w,min_periods=1).mean().values
    all_new[sym] = pd.DataFrame(feats, index=g.index)

df_exp = df.join(pd.concat(all_new.values()))
n_gen = len(pd.concat(all_new.values()).columns)
print(f'Generated {n_gen} features in {time.time()-t0:.0f}s')

# 5. NaN filter
all_cands = [c for c in df_exp.columns if c not in ID_COLS and not c.startswith('label_')
             and df_exp[c].dtype in ('float64','int64','float32','int32')
             and df_exp[c].isna().mean() < 0.95]
print(f'After NaN: {len(all_cands)} features')

# 6. Split
dates = sorted(df_exp['date'].unique())
split = int(len(dates)*0.75)
tr = df_exp[df_exp['date'].isin(dates[:split])].dropna(subset=[LABEL])
te = df_exp[df_exp['date'].isin(dates[split:])].dropna(subset=[LABEL])
print(f'Train: {len(tr):,} ({len(dates[:split])}d) Test: {len(te):,} ({len(dates[split:])}d)')

# 7. Variance pre-filter top 2000
variances = {}
for c in all_cands:
    if c in tr.columns and tr[c].var() > 1e-8: variances[c] = tr[c].var()
top2k = sorted(variances, key=variances.get, reverse=True)[:2000]
print(f'Top 2000 by variance')

# 8. Fast IC on top 2000
lr = tr.groupby('date')[LABEL].transform(lambda x: rankdata(x))
def fic(c):
    sub = tr[[c]].copy(); sub['lr'] = lr; sub = sub.dropna()
    if len(sub) < 10: return 0.0
    fr = sub.groupby(tr.loc[sub.index,'date'])[c].transform(lambda x: rankdata(x))
    di = sub.groupby(tr.loc[sub.index,'date']).apply(
        lambda g: np.corrcoef(fr.loc[g.index], g['lr'])[0,1] if len(g)>=10 else np.nan)
    return di.dropna().mean()

print('IC screening 2000 features...')
t1 = time.time()
ic_scores = {}
for i, c in enumerate(top2k):
    ic_scores[c] = fic(c)
    if (i+1)%400==0: print(f'  {i+1}/{len(top2k)} ({time.time()-t1:.0f}s)')
print(f'IC done: {time.time()-t1:.0f}s')
ic_pass_015 = [c for c,v in ic_scores.items() if abs(v) >= 0.015]
ic_pass_01  = [c for c,v in ic_scores.items() if abs(v) >= 0.01]
print(f'|IC|>=0.015: {len(ic_pass_015)}, |IC|>=0.01: {len(ic_pass_01)}')

# 9. Gate D ablation
def gate_abl(feats, X_tr, y_tr, X_te, df_te, lab, min_n=30):
    full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    full.fit(X_tr, y_tr)
    imp = pd.DataFrame({'f': feats, 'gain': full.booster_.feature_importance(importance_type='gain')})
    imp = imp.sort_values('gain', ascending=False)
    def eir(preds):
        df_e = df_te.copy(); df_e['pred'] = preds
        ics = [spearmanr(g['pred'],g[lab])[0] for _,g in df_e.groupby('date') if len(g)>=10]
        a = np.array([x for x in ics if not np.isnan(x)])
        return float(round(a.mean()/a.std() if a.std()>0 else 0,4))
    ns = sorted(set([5,10,20,30,50,75,100,150,200,len(feats),min_n]))
    best_n, best_ir = min_n, 0
    for n in ns:
        if n > len(feats): continue
        top = imp.head(n)['f'].tolist()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr[top], y_tr); ir = eir(m.predict(X_te[top]))
        if ir > best_ir: best_n, best_ir = n, ir
    sat_n = min_n
    for n in ns:
        if n > len(feats): continue
        top = imp.head(n)['f'].tolist()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr[top], y_tr); ir = eir(m.predict(X_te[top]))
        if ir >= best_ir * 0.95: sat_n = n; break
    return imp.head(max(sat_n,min_n))['f'].tolist(), best_ir, max(sat_n,min_n)

def ev(preds, df_te, lab):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics_v = [spearmanr(g['pred'],g[lab])[0] for _,g in df_e.groupby('date') if len(g)>=10]
    a = np.array([x for x in ics_v if not np.isnan(x)])
    ic_v = float(round(a.mean(),5))
    icir_v = float(round(a.mean()/a.std() if a.std()>0 else 0,4))
    dt = [g.nlargest(10,'pred') for _,g in df_e.groupby('date')]
    top = pd.concat(dt); rets = top[lab].dropna()
    sh = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    wr = float((rets>0).mean())
    return ic_v, icir_v, sh, wr, round(icir_v*0.4+sh*0.35+wr*0.25,4)

print()
print(f'DUAL: 5000 Gate D — IC pre vs post vs none')
print(f'{"Pipeline":<50} {"Feats":>6} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Comp":>10}')
print('-'*105)

# A) 5000 -> Gate D (no IC)
t0 = time.time()
g1, ir1, sn1 = gate_abl(top2k, tr[top2k].fillna(0), tr[LABEL], te[top2k].fillna(0), te, LABEL, min_n=30)
m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr[g1].fillna(0), tr[LABEL])
ic_v,icir_v,sh,wr,comp = ev(m.predict(te[g1].fillna(0)), te, LABEL)
print(f'5000 -> Gate D (no IC, min=30): {len(g1):>24} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')

# B) 5000 -> Gate D -> IC post
t0 = time.time()
# Gate D first on top2k
full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
full.fit(tr[top2k].fillna(0), tr[LABEL])
imp = pd.DataFrame({'f': top2k, 'gain': full.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)
gate_top100 = imp.head(100)['f'].tolist()
# IC on gate_top100
ic_gate = {c: fic(c) for c in gate_top100}
gate_ic_pass = [c for c,v in ic_gate.items() if abs(v) >= 0.015]
final_b = gate_ic_pass if len(gate_ic_pass) >= 30 else gate_top100[:30]
m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr[final_b].fillna(0), tr[LABEL])
ic_v,icir_v,sh,wr,comp = ev(m.predict(te[final_b].fillna(0)), te, LABEL)
print(f'5000 -> Gate(top100) -> IC post(>=0.015): {len(final_b):>13} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')

# C) 5000 -> IC pre -> Gate D
t0 = time.time()
g3, ir3, sn3 = gate_abl(ic_pass_015, tr[ic_pass_015].fillna(0), tr[LABEL], te[ic_pass_015].fillna(0), te, LABEL, min_n=30)
m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr[g3].fillna(0), tr[LABEL])
ic_v,icir_v,sh,wr,comp = ev(m.predict(te[g3].fillna(0)), te, LABEL)
print(f'5000 -> IC(>=0.015) -> Gate D (min=30): {len(g3):>14} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')

# D) ALSO test with |IC|>=0.01 pre-filter
t0 = time.time()
g4, ir4, sn4 = gate_abl(ic_pass_01, tr[ic_pass_01].fillna(0), tr[LABEL], te[ic_pass_01].fillna(0), te, LABEL, min_n=30)
m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr[g4].fillna(0), tr[LABEL])
ic_v,icir_v,sh,wr,comp = ev(m.predict(te[g4].fillna(0)), te, LABEL)
print(f'5000 -> IC(>=0.01) -> Gate D (min=30): {len(g4):>14} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')

# E) 420 baseline: use pre-built dual parquet Gate D
print()
print('Loading pre-built 420 for baseline...')
df420 = pd.read_parquet('data/abc_test_results/prebuilt_dual.parquet')
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
feat420 = FeatureEngineV35.feature_columns(df420)
tr420 = df420[df420['date'].isin(dates[:split])].dropna(subset=[LABEL])
te420 = df420[df420['date'].isin(dates[split:])].dropna(subset=[LABEL])
t0 = time.time()
g5, ir5, sn5 = gate_abl(feat420, tr420[feat420].fillna(0), tr420[LABEL], te420[feat420].fillna(0), te420, LABEL, min_n=30)
m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
m.fit(tr420[g5].fillna(0), tr420[LABEL])
ic_v,icir_v,sh,wr,comp = ev(m.predict(te420[g5].fillna(0)), te420, LABEL)
print(f'420 -> Gate D (min=30, baseline): {len(g5):>21} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')

print(f'\nTotal: {time.time()-t_total:.0f}s')
