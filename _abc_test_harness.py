"""
ABC Test Harness — fast feature selection, no slow IC screening.
All agents share pre-built 420-feature pool. Each does its own selection.

A   = all pool features (baseline, no screening)
B_dedup = NaN+variance filter (BEFORE dedup L2)
C_dedup = correlation dedup |r|>0.7 per dim (AFTER dedup L2)
B_gate  = NaN+variance filter (BEFORE gate D)
C_gate  = importance forward ablation (AFTER gate D)
"""
import argparse, json, os, sys, time, warnings
import pandas as pd, numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')
np.random.seed(42)

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True, choices=['A','B','C'])
ap.add_argument('--solution', required=True, choices=['dedup_l2','gate_d_v2'])
ap.add_argument('--board', required=True, choices=['main','dual'])
ap.add_argument('--prebuilt', required=True)
args = ap.parse_args()

LABEL = f"{args.variant}_{args.solution}_{args.board}"
t_start = time.time()
print(f"[{LABEL}] START")

# ═══════════════════════════════════════
# 1. LOAD SHARED POOL
# ═══════════════════════════════════════
df = pd.read_parquet(args.prebuilt)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
all_feats = FeatureEngineV35.feature_columns(df)
print(f"[{LABEL}] Pool: {len(df):,} rows, {df.symbol.nunique()} stocks, {len(all_feats)} features")

# ═══════════════════════════════════════
# 2. TRAIN/TEST SPLIT
# ═══════════════════════════════════════
label = 'label_pm_1d_net'
if label not in df.columns: label = 'label_1d_net'
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
test_df  = df[df['date'].isin(dates[split:])].dropna(subset=[label])
print(f"[{LABEL}] Split: train={len(train_df):,} ({len(dates[:split])}d) test={len(test_df):,} ({len(dates[split:])}d)")

# ═══════════════════════════════════════
# 3. FEATURE SELECTION (per variant+solution)
# ═══════════════════════════════════════
import lightgbm as lgb

def fast_nan_filter(feats, df_ref):
    """Drop features with >95% NaN in train set."""
    good = []
    for c in feats:
        if c in df_ref.columns and df_ref[c].isna().mean() < 0.95:
            good.append(c)
    return good

def fast_var_filter(feats, df_ref, min_var=1e-8):
    """Drop near-zero variance features."""
    good = []
    for c in feats:
        if c in df_ref.columns:
            try:
                if df_ref[c].var() > min_var:
                    good.append(c)
            except Exception:
                pass
    return good

def fast_corr_dedup(feats, df_ref, threshold=0.7):
    """Remove highly correlated features within same dim group, keeping first."""
    # Extract dim groups from feature names (dimNN_ prefix)
    import re
    dim_groups = {}
    for c in feats:
        m = re.match(r'(dim\d+)', c)
        dg = m.group(1) if m else 'other'
        dim_groups.setdefault(dg, []).append(c)

    kept = []
    for dg, cols in dim_groups.items():
        if len(cols) <= 1:
            kept.extend(cols)
            continue
        # Compute correlation on a sample for speed
        avail = [c for c in cols if c in df_ref.columns]
        if len(avail) <= 1:
            kept.extend(avail)
            continue
        sample = df_ref[avail].sample(min(5000, len(df_ref)), random_state=42)
        corr = sample.corr().abs()
        dropped = set()
        for i, ci in enumerate(avail):
            if ci in dropped: continue
            for cj in avail[i+1:]:
                if cj in dropped: continue
                if corr.loc[ci, cj] > threshold:
                    dropped.add(cj)
        kept.extend([c for c in avail if c not in dropped])
    return kept

def forward_ablation(feats, X_tr, y_tr, X_te, y_te, df_te, lab):
    """Gate D V2: train on all, rank by importance, find optimal N."""
    t0 = time.time()
    full = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1)
    full.fit(X_tr, y_tr)
    imp = pd.DataFrame({'feature': feats, 'gain': full.booster_.feature_importance(importance_type='gain')})
    imp = imp.sort_values('gain', ascending=False)

    def eval_ic(preds):
        df_e = df_te.copy(); df_e['pred'] = preds
        ics = [spearmanr(g['pred'], g[lab])[0] for _, g in df_e.groupby('date') if len(g)>=10]
        a = np.array([x for x in ics if not np.isnan(x)])
        return float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

    ns = [5, 10, 20, 30, 50, 75, 100, 150, 200, len(feats)]
    best_n, best_icir = len(feats), 0
    ablation_log = []
    for n in ns:
        top = imp.head(n)['feature'].tolist()
        m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, num_leaves=31,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_tr[top], y_tr)
        ir = eval_ic(m.predict(X_te[top]))
        ablation_log.append({'n': n, 'icir': ir})
        if ir > best_icir: best_n, best_icir = n, ir
        print(f"[{LABEL}]   abl n={n:>3}: ICIR={ir:.4f}")

    # 95% saturation
    sat_n = ns[0]
    for log in ablation_log:
        if log['icir'] >= best_icir * 0.95:
            sat_n = log['n']; break

    top_feats = imp.head(sat_n)['feature'].tolist()
    t_total = time.time() - t0
    print(f"[{LABEL}] Ablation done ({t_total:.0f}s): best ICIR={best_icir:.4f} at n={best_n}, 95% sat at n={sat_n}")
    return top_feats, ablation_log, imp

# ── Apply selection ──
# ── ALL agents share same NaN+Var base (fair IC screen) ──
base_feats = fast_var_filter(fast_nan_filter(all_feats, train_df), train_df)
print(f"[{LABEL}] Common base (NaN+Var): {len(base_feats)}/{len(all_feats)} features")

use_feats = base_feats
feat_source = ''
ablation_data = None

if args.variant == 'A':
    # A: NaN+Var → train (strongest baseline, no module)
    use_feats = base_feats
    feat_source = f'A-baseline: NaN+Var ({len(use_feats)}/{len(all_feats)})'

elif args.variant == 'B':
    # B: NaN+Var → train (BEFORE solution, same base as A)
    use_feats = base_feats
    sol_label = 'dedup L2' if args.solution == 'dedup_l2' else 'gate D'
    feat_source = f'B_BEFORE {sol_label}: NaN+Var ({len(use_feats)}/{len(all_feats)})'

elif args.variant == 'C':
    if args.solution == 'dedup_l2':
        # C_dedup: NaN+Var → correlation dedup (AFTER dedup L2)
        use_feats = fast_corr_dedup(base_feats, train_df, threshold=0.7)
        feat_source = f'C_dedup AFTER: NaN+Var→corr dedup |r|>0.7 ({len(use_feats)}/{len(all_feats)})'
    else:
        # C_gate: NaN+Var → forward ablation (AFTER gate D)
        X_tr = train_df[base_feats].fillna(0)
        y_tr = train_df[label]
        X_te = test_df[base_feats].fillna(0)
        use_feats, ablation_data, _ = forward_ablation(base_feats, X_tr, y_tr, X_te, test_df[label], test_df, label)
        feat_source = f'C_gate AFTER: NaN+Var→ablation top-{len(use_feats)}/{len(base_feats)}'

n_feat_used = len(use_feats)
print(f"[{LABEL}] Feature selection: {feat_source}")

# ═══════════════════════════════════════
# 4. TRAIN & EVALUATE
# ═══════════════════════════════════════
X_train = train_df[use_feats].fillna(0)
y_train = train_df[label]
X_test = test_df[use_feats].fillna(0)

t_train = time.time()
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_train, y_train)
train_time = time.time() - t_train
preds = model.predict(X_test)

# IC/ICIR
df_e = test_df.copy(); df_e['pred'] = preds
ics = []
for d, g in df_e.groupby('date'):
    if len(g) < 10: continue
    ic, _ = spearmanr(g['pred'], g[label])
    if not np.isnan(ic): ics.append(ic)
a = np.array(ics)
ic = float(round(a.mean(), 5))
icir = float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

# Top-10 profit/risk
daily_top = []
for d, g in df_e.groupby('date'):
    daily_top.append(g.nlargest(10, 'pred'))
top_df = pd.concat(daily_top)
rets = top_df[label].dropna()
top10_mean_ret = float(rets.mean())
top10_std_ret = float(rets.std())
top10_win_rate = float((rets > 0).mean())
top10_sharpe = float(top10_mean_ret/top10_std_ret*np.sqrt(252)) if top10_std_ret>0 else 0
top10_n_obs = len(rets)

# Composite
composite = round(icir*0.40 + top10_sharpe*0.35 + top10_win_rate*0.25, 4)

# ═══════════════════════════════════════
# 5. OUTPUT
# ═══════════════════════════════════════
result = {
    'label': LABEL, 'variant': args.variant, 'solution': args.solution, 'board': args.board,
    'elapsed_s': round(time.time()-t_start, 1),
    'n_stocks': test_df['symbol'].nunique(),
    'n_train_rows': len(train_df), 'n_test_rows': len(test_df),
    'n_train_days': len(dates[:split]), 'n_test_days': len(dates[split:]),
    'label_col': label,
    'n_feat_pool': len(all_feats), 'n_feat_used': n_feat_used,
    'feat_source': feat_source,
    'train_time_s': round(train_time, 1),
    'oos_ic': ic, 'oos_icir': icir,
    'n_pos_ic_days': sum(1 for x in ics if x>0),
    'ic_std': round(float(np.std(ics)),5),
    'top10_mean_ret': round(top10_mean_ret,6),
    'top10_std_ret': round(top10_std_ret,6),
    'top10_win_rate': round(top10_win_rate,4),
    'top10_sharpe': round(top10_sharpe,4),
    'top10_n_obs': top10_n_obs,
    'composite_score': composite,
}
if ablation_data:
    result['ablation'] = ablation_data

print(f"[{LABEL}] IC={ic:+.5f} ICIR={icir:.4f} Sharpe={top10_sharpe:.2f} WinRate={top10_win_rate:.1%} Composite={composite:.4f}")
print(f"ABC_RESULT_JSON: {json.dumps(result)}")
print(f"[{LABEL}] DONE in {result['elapsed_s']:.0f}s")
