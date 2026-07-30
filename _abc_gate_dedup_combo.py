"""Dual board: NaN+Var → Gate D (N features) → Dedup L2 → train."""
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

# Load
df = pd.read_parquet(PREBUILT)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
all_feats = FeatureEngineV35.feature_columns(df)

# NaN+Var base
base = [c for c in all_feats if c in df.columns and df[c].isna().mean() < 0.95 and df[c].var() > 1e-8]
print(f'Base (NaN+Var): {len(base)}/{len(all_feats)}')

# Split
if LABEL not in df.columns: LABEL = 'label_1d_net'
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[LABEL])
test_df  = df[df['date'].isin(dates[split:])].dropna(subset=[LABEL])

X_tr = train_df[base].fillna(0); y_tr = train_df[LABEL]
X_te = test_df[base].fillna(0)

# Get importance ranking (train once, reuse)
full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
full.fit(X_tr, y_tr)
imp = pd.DataFrame({'feature': base, 'gain': full.booster_.feature_importance(importance_type='gain')})
imp = imp.sort_values('gain', ascending=False)

# Correlation dedup function
def corr_dedup(feats, df_ref, threshold=0.7):
    dim_groups = {}
    for c in feats:
        m = re.match(r'(dim\d+)', c)
        dg = m.group(1) if m else 'other'
        dim_groups.setdefault(dg, []).append(c)
    kept = []
    for dg, cols in dim_groups.items():
        if len(cols) <= 1: kept.extend(cols); continue
        avail = [c for c in cols if c in df_ref.columns]
        if len(avail) <= 1: kept.extend(avail); continue
        sample = df_ref[avail].sample(min(5000, len(df_ref)), random_state=42)
        corr_mat = sample.corr().abs()
        dropped = set()
        for i, ci in enumerate(avail):
            if ci in dropped: continue
            for cj in avail[i+1:]:
                if cj in dropped: continue
                if corr_mat.loc[ci, cj] > threshold:
                    dropped.add(cj)
        kept.extend([c for c in avail if c not in dropped])
    return kept

def eval_icir(preds, df_te, lab):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics = [spearmanr(g['pred'], g[lab])[0] for _, g in df_e.groupby('date') if len(g)>=10]
    a = np.array([x for x in ics if not np.isnan(x)])
    return float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

def eval_full(preds, df_te, lab):
    df_e = df_te.copy(); df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[lab])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    ic = float(round(a.mean(), 5))
    icir = float(round(a.mean()/a.std() if a.std()>0 else 0, 4))
    daily_top = [g.nlargest(10, 'pred') for _, g in df_e.groupby('date')]
    top_df = pd.concat(daily_top)
    rets = top_df[lab].dropna()
    sharpe = float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    winrate = float((rets>0).mean())
    comp = round(icir*0.40 + sharpe*0.35 + winrate*0.25, 4)
    return ic, icir, sharpe, winrate, comp

print(f'\n{"="*70}')
print(f'DUAL: NaN+Var → Gate D (top N) → Dedup L2 |r|>0.7')
print(f'{"="*70}')
print(f'{"N_gate":<8} {"After dedup":>12} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Composite":>10}')
print('-'*62)

results = []
for n_gate in GATE_NS:
    t0 = time.time()
    top_n = imp.head(n_gate)['feature'].tolist()
    deduped = corr_dedup(top_n, train_df, threshold=0.7)
    n_final = len(deduped)

    X_tr_sel = train_df[deduped].fillna(0)
    X_te_sel = test_df[deduped].fillna(0)

    model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_tr_sel, y_tr)
    preds = model.predict(X_te_sel)

    ic, icir, sharpe, winrate, comp = eval_full(preds, test_df, LABEL)
    elapsed = time.time() - t0
    print(f'{n_gate:<8} {n_final:>12} {ic:>+9.5f} {icir:>7.4f} {sharpe:>7.2f} {winrate:>8.1%} {comp:>10.4f}  ({elapsed:.0f}s)')
    results.append({'gate_n': n_gate, 'final_n': n_final, 'ic': ic, 'icir': icir,
                    'sharpe': sharpe, 'winrate': winrate, 'composite': comp})

# Compare with baselines
print(f'\n{"="*70}')
print(f'COMPARISON')
print(f'{"="*70}')
print(f'{"Method":<35} {"IC":>9} {"ICIR":>7} {"Sharpe":>7} {"WinRate":>8} {"Composite":>10}')
print('-'*75)

baselines = {
    'A (NaN only)': (+0.00444, 0.0285, -1.83, 0.443, -0.519),
    'C_dedup only (|r|>0.7, n=199)': (+0.01821, 0.0997, -1.46, 0.470, -0.354),
    'C_gate only (n=10)': (+0.06110, 0.3728, -1.74, 0.438, -0.351),
    'Gate min30 only (n=30)': (+0.00181, 0.0103, -1.01, 0.453, -0.237),
}
for name, (ic, icir, sh, wr, comp) in baselines.items():
    print(f'{name:<35} {ic:>+9.5f} {icir:>7.4f} {sh:>7.2f} {wr:>8.1%} {comp:>10.4f}')

best = max(results, key=lambda r: r['composite'])
print(f'\nBEST COMBO: Gate D top-{best["gate_n"]} → dedup → {best["final_n"]} features, Composite={best["composite"]:.4f}')
