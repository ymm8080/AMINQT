"""MAIN: v3 raw panel -> brute-force 5000 features -> IC screen -> Dedup L2 -> train"""
import warnings, time
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata
np.random.seed(42)
import lightgbm as lgb

t_total = time.time()

# ── 1. Load v3 raw panel ──
print('Loading v3 panel...')
panel = pd.read_parquet('data/panel_full_enriched_v3.parquet')
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]

import json
with open('data/csi300_constituents.json') as f: csi300 = set(json.load(f))
panel = panel[panel['symbol'].isin(csi300 & set(panel['symbol'].unique()))]

from app.pipeline1.cleaning_pipeline import CleaningPipeline
main_df, _ = CleaningPipeline().run_train(panel)
print(f'Clean: {len(main_df)} rows, {main_df.symbol.nunique()} stocks, {len(main_df.columns)} cols')

# ── 2. Build labels ──
from app.pipeline1.label_engine import LabelEngine
# Just build labels, not features
df = LabelEngine.build_path_labels(main_df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=3)
LABEL = 'label_pm_1d_net'
if LABEL not in df.columns: LABEL = 'label_1d_net'
print(f'Label: {LABEL}, non-NaN: {df[LABEL].notna().sum():,}')

# ── 3. Raw numeric base (exclude ids, labels) ──
ID_COLS = {'symbol','date','board','industry','announce_date','is_suspended','is_st','tradestatus'}
RAW = [c for c in df.columns if c not in ID_COLS and not c.startswith('label_')
       and not c.startswith('dim') and df[c].dtype in ('float64','int64','float32','int32')]
print(f'Raw base cols: {len(RAW)}')

# ── 4. Generate brute-force features per stock ──
print('Generating brute-force features...')
t0 = time.time()
all_new = {}
for sym, g in df.groupby('symbol'):
    g = g.sort_values('date')
    feats = {}
    for col in RAW:
        if col not in g.columns: continue
        s = g[col].astype(float).values
        n = len(s)
        # pct_change
        for w in [1,2,3,5,10,20,40,60]:
            out = np.full(n, np.nan)
            out[w:] = (s[w:] - s[:-w]) / np.abs(s[:-w]) * 100
            feats[f'{col}_pct{w}'] = out
        # rolling mean
        for w in [5,10,20,40,60]:
            feats[f'{col}_ma{w}'] = pd.Series(s).rolling(w, min_periods=1).mean().values
        # rolling std
        for w in [5,10,20,40]:
            feats[f'{col}_std{w}'] = pd.Series(s).rolling(w, min_periods=1).std().values
        # rolling max-min range
        for w in [10,20,40]:
            r = pd.Series(s).rolling(w, min_periods=1)
            feats[f'{col}_max{w}'] = r.max().values
            feats[f'{col}_min{w}'] = r.min().values
            feats[f'{col}_range{w}'] = r.max().values - r.min().values
        # diff (absolute change)
        for w in [1,5,20]:
            out = np.full(n, np.nan)
            out[w:] = s[w:] - s[:-w]
            feats[f'{col}_d{w}'] = out
        # momentum (current / N-days ago)
        for w in [5,20,40]:
            out = np.full(n, np.nan)
            out[w:] = s[w:] / np.abs(s[:-w])
            feats[f'{col}_mom{w}'] = out
        # EMA
        for w in [5,20,40]:
            feats[f'{col}_ema{w}'] = pd.Series(s).ewm(span=w, min_periods=1).mean().values
    all_new[sym] = pd.DataFrame(feats, index=g.index)

df_exp = df.join(pd.concat(all_new.values()))
n_gen = len(pd.concat(all_new.values()).columns)
print(f'Generated {n_gen} features in {time.time()-t0:.0f}s')

# ── 5. NaN filter ──
all_cands = [c for c in df_exp.columns if c not in ID_COLS and not c.startswith('label_')
             and df_exp[c].dtype in ('float64','int64','float32','int32')
             and df_exp[c].isna().mean() < 0.95]
print(f'After NaN filter: {len(all_cands)} features')

# ── 6. Train/test split ──
dates = sorted(df_exp['date'].unique())
split = int(len(dates)*0.75)
tr = df_exp[df_exp['date'].isin(dates[:split])].dropna(subset=[LABEL])
te = df_exp[df_exp['date'].isin(dates[split:])].dropna(subset=[LABEL])
print(f'Train: {len(tr):,} ({len(dates[:split])}d) | Test: {len(te):,} ({len(dates[split:])}d)')

# ── 7. Variance pre-filter (top 2000) ──
print('Variance filter...')
variances = {}
for c in all_cands:
    if c in tr.columns:
        v = tr[c].var()
        if v > 1e-8: variances[c] = v
top2k = sorted(variances, key=variances.get, reverse=True)[:2000]
print(f'Top 2000 by variance')

# ── 8. Fast IC screening ──
print('IC screening...')
lr = tr.groupby('date')[LABEL].transform(lambda x: rankdata(x))
def fic(c):
    sub = tr[[c]].copy(); sub['lr'] = lr; sub = sub.dropna()
    if len(sub) < 10: return 0.0
    fr = sub.groupby(tr.loc[sub.index,'date'])[c].transform(lambda x: rankdata(x))
    di = sub.groupby(tr.loc[sub.index,'date']).apply(
        lambda g: np.corrcoef(fr.loc[g.index], g['lr'])[0,1] if len(g)>=10 else np.nan)
    return di.dropna().mean()

t1 = time.time()
ic_scores = {}
for i, c in enumerate(top2k):
    ic_scores[c] = fic(c)
    if (i+1)%400==0: print(f'  {i+1}/{len(top2k)} ({time.time()-t1:.0f}s)')
print(f'IC done: {time.time()-t1:.0f}s')

for th in [0.03, 0.02, 0.015, 0.01, 0.005]:
    n = sum(1 for v in ic_scores.values() if abs(v) >= th)
    print(f'  |IC| >= {th:.3f}: {n}')

# ── 9. Dedup ──
import re
def dedup(feats, df_ref, th=0.7):
    groups = {}
    for c in feats:
        m = re.match(r'([a-z_]+)', c.split('_pct')[0].split('_ma')[0].split('_std')[0].split('_d')[0].split('_mom')[0].split('_ema')[0].split('_max')[0].split('_min')[0].split('_range')[0])
        dg = m.group(1) if m else 'other'
        groups.setdefault(dg, []).append(c)
    kept = []
    for dg, cols in groups.items():
        if len(cols)<=1: kept.extend(cols); continue
        av = [c for c in cols if c in df_ref.columns]
        if len(av)<=1: kept.extend(av); continue
        s = df_ref[av].sample(min(5000,len(df_ref)), random_state=42)
        cm = s.corr().abs(); dropped = set()
        for i,ci in enumerate(av):
            if ci in dropped: continue
            for cj in av[i+1:]:
                if cj in dropped: continue
                if cm.loc[ci,cj] > th: dropped.add(cj)
        kept.extend([c for c in av if c not in dropped])
    return kept

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
print(f'MAIN: Brute-Force {n_gen} features -> IC -> Dedup L2')
print(f'{"Pipeline":<45} {"Pool":>6} {"IC":>6} {"+Dedup":>8} {"IC_OOS":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Comp":>10}')
print('-'*115)

for ic_label, ic_th in [('|IC|>=0.005',0.005),('|IC|>=0.01',0.01),('|IC|>=0.015',0.015),('|IC|>=0.02',0.02)]:
    ic_list = [c for c,v in ic_scores.items() if abs(v) >= ic_th]
    if len(ic_list) < 10: continue
    t0 = time.time()
    deduped = dedup(ic_list, tr, 0.7)
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr[deduped].fillna(0), tr[LABEL])
    ic_v,icir_v,sh,wr,comp = ev(m.predict(te[deduped].fillna(0)), te, LABEL)
    pipe_name = f"Brute{n_gen} -> {ic_label} -> Dedup"
    print(f"{pipe_name:<45} {len(all_cands):>6} {len(ic_list):>6} {len(deduped):>8} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)")

# Also test: IC pass only, no dedup (just to see)
for ic_label, ic_th in [('|IC|>=0.015',0.015)]:
    ic_list = [c for c,v in ic_scores.items() if abs(v) >= ic_th]
    t0 = time.time()
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
    m.fit(tr[ic_list].fillna(0), tr[LABEL])
    ic_v,icir_v,sh,wr,comp = ev(m.predict(te[ic_list].fillna(0)), te, LABEL)
    pipe_name = f"Brute{n_gen} -> {ic_label} (NO dedup)"
    print(f"{pipe_name:<45} {len(all_cands):>6} {len(ic_list):>6} {"-":>8} {ic_v:>+9.5f} {icir_v:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)")

print(f'\nBaseline: orig 420 pool NaN+Var->Dedup: 204 feats, Composite=-0.137')
print(f'Total time: {time.time()-t_total:.0f}s')
