"""Main board: NaN+Var -> Gate D (N features) -> Dedup L2 -> train."""
import json, os, time, warnings, re
import pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb

BOARD = 'main'
PREBUILT = f'data/abc_test_results/prebuilt_{BOARD}.parquet'
LABEL = 'label_pm_1d_net'
GATE_NS = [30, 50, 75, 100, 150]

df = pd.read_parquet(PREBUILT)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
all_feats = FeatureEngineV35.feature_columns(df)
base = [c for c in all_feats if c in df.columns and df[c].isna().mean() < 0.95 and df[c].var() > 1e-8]
print(f'Base (NaN+Var): {len(base)}/{len(all_feats)}')

if LABEL not in df.columns: LABEL = 'label_1d_net'
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[LABEL])
test_df  = df[df['date'].isin(dates[split:])].dropna(subset=[LABEL])
X_tr = train_df[base].fillna(0); y_tr = train_df[LABEL]

full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
full.fit(X_tr, y_tr)
imp = pd.DataFrame({'feature': base, 'gain': full.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)

def corr_dedup(feats, df_ref, threshold=0.7):
    dim_groups = {}
    for c in feats:
        m = re.match(r'(dim\d+)', c); dg = m.group(1) if m else 'other'
        dim_groups.setdefault(dg, []).append(c)
    kept = []
    for dg, cols in dim_groups.items():
        if len(cols) <= 1: kept.extend(cols); continue
        avail = [c for c in cols if c in df_ref.columns]
        if len(avail) <= 1: kept.extend(avail); continue
        s = df_ref[avail].sample(min(5000, len(df_ref)), random_state=42)
        cm = s.corr().abs(); dropped = set()
        for i, ci in enumerate(avail):
            if ci in dropped: continue
            for cj in avail[i+1:]:
                if cj in dropped: continue
                if cm.loc[ci, cj] > threshold: dropped.add(cj)
        kept.extend([c for c in avail if c not in dropped])
    return kept

def eval_all(preds, df_te, lab):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics = [spearmanr(g['pred'], g[lab])[0] for _, g in df_e.groupby('date') if len(g)>=10]
    a = np.array([x for x in ics if not np.isnan(x)])
    ic = float(round(a.mean(),5))
    icir = float(round(a.mean()/a.std() if a.std()>0 else 0,4))
    dt = [g.nlargest(10, 'pred') for _, g in df_e.groupby('date')]
    top = pd.concat(dt); rets = top[lab].dropna()
    sh = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    wr = float((rets>0).mean())
    return ic, icir, sh, wr, round(icir*0.40+sh*0.35+wr*0.25,4)

print(f'\n{"="*65}')
print(f'MAIN: NaN+Var -> Gate D (top N) -> Dedup L2 |r|>0.7')
print(f'{"="*65}')
print(f'{"N_gate":<8} {"->Dedup":>9} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Composite":>10}')
print('-'*60)

results = []
for n_gate in GATE_NS:
    t0 = time.time()
    top_n = imp.head(n_gate)['feature'].tolist()
    deduped = corr_dedup(top_n, train_df, threshold=0.7)
    n_final = len(deduped)
    m = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    m.fit(train_df[deduped].fillna(0), y_tr)
    ic, icir, sh, wr, comp = eval_all(m.predict(test_df[deduped].fillna(0)), test_df, LABEL)
    print(f'{n_gate:<8} {n_final:>9} {ic:>+9.5f} {icir:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}  ({time.time()-t0:.0f}s)')
    results.append({'gate_n': n_gate, 'final_n': n_final, 'ic': ic, 'icir': icir, 'sharpe': sh, 'winrate': wr, 'composite': comp})

print(f'\n{"="*65}')
print(f'vs BASELINES')
print(f'{"="*65}')
print(f'{"Method":<35} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Composite":>10}')
print('-'*75)
for name, (ic, icir, sh, wr, comp) in [
    ('A (NaN only, n=402)', (-0.01903, 0.1234, -1.63, 0.396, -0.421)),
    ('C_dedup only (|r|>0.7, n=204)', (+0.00569, 0.0362, -0.74, 0.429, -0.137)),
    ('Gate min50 only (n=50)', (-0.01596, 0.1117, -1.61, 0.424, -0.413)),
]:
    print(f'{name:<35} {ic:>+9.5f} {icir:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}')

best = max(results, key=lambda r: r['composite'])
print(f'\nBEST COMBO: Gate D top-{best["gate_n"]} -> dedup -> {best["final_n"]} features, Composite={best["composite"]:.4f}')
