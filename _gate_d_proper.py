"""Gate D (proper): Model-level feature screening with full CSI300 sample.

Key differences from _gate_d.py:
- 12 months of data (not 6)
- ALL CSI300 stocks (~200+), not 50
- 40 test days (not ~20% split)
- n_estimators=300 for better importance estimates
- Per-dim-group breakdown
"""
import logging, sys, os, json
import pandas as pd, numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import tempfile, shutil
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
np.random.seed(42)

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.cleaning_pipeline import CleaningPipeline

PANEL = "data/panel_full_enriched_v4_20260729.parquet"

print("=" * 70)
print("Gate D (Proper): Model-Level Feature Screening — Full CSI300")
print("=" * 70)

# ── Step 1: Load CSI300, 12 months, ALL stocks ──
with open("data/csi300_constituents.json") as f:
    csi300 = set(json.load(f))

panel = pd.read_parquet(PANEL)
print(f"\nRaw panel: {len(panel):,} rows, {panel['symbol'].nunique()} stocks, "
      f"dates {panel['date'].min()} ~ {panel['date'].max()}")

cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
panel = panel[panel['symbol'].isin(in_panel)].copy()
print(f"After 12m CSI300 filter: {len(panel):,} rows, {panel['symbol'].nunique()} stocks, "
      f"dates {panel['date'].min()} ~ {panel['date'].max()}")

# ── Step 2: Clean and build features with registry ──
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)
print(f"\nMain board after clean: {len(main_df):,} rows, {main_df['symbol'].nunique()} stocks, "
      f"dates {main_df['date'].min()} ~ {main_df['date'].max()}")

reg_dir = tempfile.mkdtemp()
reg_path = os.path.join(reg_dir, "feature_registry.json")
registry = FeatureRegistry(path=reg_path)

# Seed with a sample per stock
sample = main_df.groupby('symbol', group_keys=False).apply(
    lambda g: g.head(min(30, len(g)))
).reset_index(drop=True)
n_reg = registry._seed(sample)
print(f"Registry seeded: {n_reg} features")

features = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)

df = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
feat_cols = select_features(df, "main", "gate_d", screener, registry=registry)
print(f"Features into model: {len(feat_cols)}")

# ── Step 3: Train/Test split — last 40 days as test ──
dates = sorted(df['date'].unique())
print(f"\nTotal trading days: {len(dates)}, range {dates[0]} ~ {dates[-1]}")

test_dates = set(dates[-40:])
train_dates = set(dates[:-40])

label = "label_pm_1d_net"
if label not in df.columns:
    label = "label_1d_net"
    if label not in df.columns:
        # Fallback: create simple forward return label
        print(f"WARNING: standard labels not found, creating fallback label")
        df[label] = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / x - 1
        )
        df = df.dropna(subset=[label])

train_df = df[df['date'].isin(train_dates)].dropna(subset=[label])
test_df = df[df['date'].isin(test_dates)].dropna(subset=[label])
print(f"Train: {len(train_df):,} rows, {train_df['date'].nunique()} days, "
      f"{train_df['symbol'].nunique()} stocks")
print(f"Test:  {len(test_df):,} rows, {test_df['date'].nunique()} days, "
      f"{test_df['symbol'].nunique()} stocks")

# ── Step 4: Train LightGBM (n_estimators=300 for better importance) ──
X_train = train_df[feat_cols].fillna(0)
y_train = train_df[label]
X_test = test_df[feat_cols].fillna(0)
y_test = test_df[label]

print(f"\nTraining LightGBM (n_estimators=300, max_depth=6) with {len(feat_cols)} features "
      f"on {len(X_train):,} rows...")
model = lgb.LGBMRegressor(
    n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(X_train, y_train)
print("Training complete.")

# ── Step 5: Feature importance ──
importance = pd.DataFrame({
    'feature': feat_cols,
    'gain': model.booster_.feature_importance(importance_type='gain'),
    'split': model.booster_.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)

importance['gain_pct'] = importance['gain'] / importance['gain'].sum()
importance['cum_gain'] = importance['gain_pct'].cumsum()

print(f"\n{'=' * 70}")
print(f"TOP 30 FEATURES BY GAIN IMPORTANCE")
print(f"{'=' * 70}")
for i, row in importance.head(30).iterrows():
    print(f"  {i+1:>2}. {row['feature']:<45} gain={row['gain_pct']:.2%}  cum={row['cum_gain']:.2%}  split={row['split']:.0f}")

print(f"\nCumulative gain thresholds:")
for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    n = (importance['cum_gain'] < threshold).sum() + 1
    print(f"  {threshold:.0%} cumulative gain: {n} features")

# ── Step 6: OOS evaluation ──
def oos_ic(preds, df_test, label_col):
    """Compute daily OOS Rank IC."""
    df_eval = df_test.copy()
    df_eval['pred'] = preds
    ics = []
    for d, g in df_eval.groupby('date'):
        if len(g) < 10:
            continue
        ic, _ = spearmanr(g['pred'], g[label_col])
        if not np.isnan(ic):
            ics.append(ic)
    ics = np.array(ics)
    mean_ic = ics.mean()
    icir = mean_ic / ics.std() if ics.std() > 0 else 0
    return round(mean_ic, 5), round(icir, 4), len(ics)

preds_all = model.predict(X_test)
ic_all, icir_all, n_all_days = oos_ic(preds_all, test_df, label)
print(f"\nModel A (all {len(feat_cols)} features): OOS IC={ic_all:.5f}, ICIR={icir_all:.4f}, n_days={n_all_days}")

# ── Step 7: Forward ablation ──
ablation_ns = [10, 20, 30, 50, 75, 100, 150, 200, len(feat_cols)]
# Deduplicate if len(feat_cols) already in list
ablation_ns = sorted(set(ablation_ns))

print(f"\n{'=' * 70}")
print(f"FORWARD ABLATION (retrain with top-N features)")
print(f"{'=' * 70}")
print(f"  {'n':>3}  {'cum_gain':>10}  {'OOS IC':>10}  {'OOS ICIR':>10}  {'n_days':>6}")
print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")

ablation_results = []
for n_features in ablation_ns:
    top_n = importance.head(n_features)['feature'].tolist()
    cum_gain = importance.head(n_features)['gain_pct'].sum()

    m = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    m.fit(X_train[top_n], y_train)
    preds = m.predict(X_test[top_n])
    ic, icir, n_d = oos_ic(preds, test_df, label)

    ablation_results.append({
        'n': n_features,
        'cum_gain': cum_gain,
        'ic': ic,
        'icir': icir,
        'n_days': n_d,
    })
    print(f"  {n_features:>3}  {cum_gain:>9.1%}  {ic:>+10.5f}  {icir:>+10.4f}  {n_d:>6}")

# ── Step 8: Find saturation point ──
df_a = pd.DataFrame(ablation_results)
best_idx = df_a['icir'].idxmax()
best_icir = df_a.loc[best_idx, 'icir']
best_n = df_a.loc[best_idx, 'n']

# Find first point that reaches 95% of best ICIR
saturation_row = None
for i, row in df_a.iterrows():
    if row['icir'] >= best_icir * 0.95:
        saturation_row = row
        break

# Also find the "diminishing returns" point: where ICIR improvement < 0.01
diminishing_row = None
prev_icir = -np.inf
for i, row in df_a.iterrows():
    if row['icir'] - prev_icir < 0.01 and prev_icir > -np.inf:
        diminishing_row = row
        break
    prev_icir = row['icir']

print(f"\n{'=' * 70}")
print(f"VERDICT")
print(f"{'=' * 70}")
print(f"  All {len(feat_cols)} features:    IC={ic_all:.5f}, ICIR={icir_all:.4f}")
print(f"  Best ablation:           n={best_n:.0f}, IC={df_a.loc[best_idx, 'ic']:.5f}, ICIR={best_icir:.4f}")
if saturation_row is not None:
    print(f"  95% saturation point:    n={saturation_row['n']:.0f}, IC={saturation_row['ic']:.5f}, ICIR={saturation_row['icir']:.4f}")
    print(f"  Feature reduction:       {len(feat_cols)} -> {saturation_row['n']:.0f} "
          f"({saturation_row['n']/len(feat_cols):.0%} of total)")
if diminishing_row is not None:
    print(f"  Diminishing returns at:  n={diminishing_row['n']:.0f}, ICIR={diminishing_row['icir']:.4f}")

# ── Step 9: Per-dim-group importance breakdown ──
print(f"\n{'=' * 70}")
print(f"PER-DIM-GROUP IMPORTANCE BREAKDOWN")
print(f"{'=' * 70}")

# Collect dim_group for each feature
dim_groups = {}
for feat in feat_cols:
    meta = registry.get_meta(feat)
    dg = meta.get('dim_group', 'unknown') if meta else 'unknown'
    dim_groups[feat] = dg

# Aggregate importance by dim_group
importance['dim_group'] = importance['feature'].map(dim_groups)
dim_agg = importance.groupby('dim_group').agg(
    total_gain_pct=('gain_pct', 'sum'),
    n_features=('feature', 'count'),
    avg_gain=('gain_pct', 'mean'),
    top_feature=('gain_pct', 'max'),
).sort_values('total_gain_pct', ascending=False)

print(f"  {'Dim Group':<30} {'Gain %':>8} {'#Feats':>6} {'Avg %':>8} {'Top %':>8} {'Top Feat':>25}")
print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*25}")
for dg, row in dim_agg.iterrows():
    # Find top feature name in this dim_group
    dg_feats = importance[importance['dim_group'] == dg]
    top_feat_name = dg_feats.head(1)['feature'].values[0] if len(dg_feats) > 0 else ''
    print(f"  {dg:<30} {row['total_gain_pct']:>7.2%} {row['n_features']:>6} "
          f"{row['avg_gain']:>7.4%} {row['top_feature']:>7.4%} {top_feat_name[:25]:>25}")

# Top-15 individual features with dim_group
print(f"\nTop-15 features with dim_group:")
print(f"  {'#':>2} {'Feature':<45} {'Gain%':>7} {'Cum%':>7} {'Dim Group':<20} {'Split':>5}")
print(f"  {'-'*2} {'-'*45} {'-'*7} {'-'*7} {'-'*20} {'-'*5}")
for i, row in importance.head(15).iterrows():
    dg = dim_groups.get(row['feature'], 'unknown')
    print(f"  {i+1:>2} {row['feature']:<45} {row['gain_pct']:>6.2%} {row['cum_gain']:>6.2%} {dg:<20} {row['split']:>5.0f}")

# ── Cleanup ──
shutil.rmtree(reg_dir)
print(f"\n{'=' * 70}")
print("Gate D (Proper) complete.")
print(f"{'=' * 70}")
