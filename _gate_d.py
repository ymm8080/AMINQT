"""Gate D: Model-level feature screening — LightGBM importance + AB comparison."""
import logging, sys, os, json
import pandas as pd, numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
np.random.seed(42)

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.cleaning_pipeline import CleaningPipeline
import tempfile, shutil

PANEL = "data/panel_full_enriched_v4_20260729.parquet"

# ── Step 1: Load CSI300, 50 stocks, 6 months ──
print("="*60)
print("Gate D: Model-Level Feature Screening")
print("="*60)

with open("data/csi300_constituents.json") as f:
    csi300 = set(json.load(f))

panel = pd.read_parquet(PANEL)
cutoff = panel['date'].max() - pd.Timedelta(days=180)
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
stocks = np.random.choice(list(in_panel), size=min(50, len(in_panel)), replace=False)
panel = panel[panel['symbol'].isin(stocks)].copy()
print(f"Data: {len(panel):,} rows, {panel['symbol'].nunique()} stocks")

# ── Step 2: Build features with registry ──
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)

reg_dir = tempfile.mkdtemp()
reg_path = os.path.join(reg_dir, "feature_registry.json")
registry = FeatureRegistry(path=reg_path)

sample = main_df.groupby('symbol', group_keys=False).apply(
    lambda g: g.head(min(30, len(g)))
).reset_index(drop=True)
registry._seed(sample)

features = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)

df = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
feat_cols = select_features(df, "main", "D", screener, registry=registry)
print(f"Features into model: {len(feat_cols)}")

# ── Step 3: Train/Test split (time-series: last 20% dates as test) ──
dates = sorted(df['date'].unique())
split_idx = int(len(dates) * 0.8)
train_dates = set(dates[:split_idx])
test_dates = set(dates[split_idx:])

label = "label_pm_1d_net"
if label not in df.columns:
    label = "label_1d_net"
    if label not in df.columns:
        # Fallback: create simple forward return label
        df[label] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1)/x - 1)
        df = df.dropna(subset=[label])

train_df = df[df['date'].isin(train_dates)].dropna(subset=[label])
test_df = df[df['date'].isin(test_dates)].dropna(subset=[label])
print(f"Train: {len(train_df):,} rows, {train_df['date'].nunique()} days")
print(f"Test:  {len(test_df):,} rows, {test_df['date'].nunique()} days")

# ── Step 4: Train LightGBM (Model A: all features) ──
X_train = train_df[feat_cols].fillna(0)
y_train = train_df[label]
X_test = test_df[feat_cols].fillna(0)
y_test = test_df[label]

print(f"\nTraining LightGBM with {len(feat_cols)} features...")
model = lgb.LGBMRegressor(
    n_estimators=200, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1,
)
model.fit(X_train, y_train)

# ── Step 5: Feature importance ──
importance = pd.DataFrame({
    'feature': feat_cols,
    'gain': model.booster_.feature_importance(importance_type='gain'),
    'split': model.booster_.feature_importance(importance_type='split'),
}).sort_values('gain', ascending=False)

# Cumulative importance
importance['gain_pct'] = importance['gain'] / importance['gain'].sum()
importance['cum_gain'] = importance['gain_pct'].cumsum()

print(f"\nTop 20 features by gain importance:")
for i, row in importance.head(20).iterrows():
    print(f"  {row['feature']:<40} gain={row['gain']:.0f}  ({row['gain_pct']:.2%})  cum={row['cum_gain']:.2%}")

# How many features to reach 50%, 80%, 95%, 99% cumulative gain?
for threshold in [0.50, 0.80, 0.90, 0.95, 0.99]:
    n = (importance['cum_gain'] < threshold).sum() + 1
    print(f"  {threshold:.0%} cumulative gain: {n} features")

# ── Step 6: OOS evaluation ──
from scipy.stats import spearmanr

def oos_ic(preds, df_test, label_col):
    """Compute daily OOS Rank IC."""
    df_eval = df_test.copy()
    df_eval['pred'] = preds
    ics = []
    for d, g in df_eval.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[label_col])
        if not np.isnan(ic): ics.append(ic)
    ics = np.array(ics)
    mean_ic = ics.mean()
    icir = mean_ic / ics.std() if ics.std() > 0 else 0
    return round(mean_ic, 5), round(icir, 4), len(ics)

preds_all = model.predict(X_test)
ic_all, icir_all, n_all = oos_ic(preds_all, test_df, label)
print(f"\nModel A (all {len(feat_cols)} features): OOS IC={ic_all:.5f}, ICIR={icir_all:.4f}, n_days={n_all}")

# ── Step 7: Forward ablation — how few features can we use? ──
print(f"\nForward ablation (cumulative gain thresholds):")
ablation_results = []
for n_features in [10, 20, 30, 50, 75, 100, 150, 200, len(feat_cols)]:
    top_n = importance.head(n_features)['feature'].tolist()

    # Retrain with top-N
    m = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, num_leaves=31,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    m.fit(X_train[top_n], y_train)
    preds = m.predict(X_test[top_n])
    ic, icir, n = oos_ic(preds, test_df, label)

    cum_gain = importance.head(n_features)['gain_pct'].sum()
    ablation_results.append({'n': n_features, 'cum_gain': cum_gain, 'ic': ic, 'icir': icir})
    print(f"  Top {n_features:>3}: cum_gain={cum_gain:.1%}  OOS IC={ic:.5f}  ICIR={icir:.4f}")

# ── Step 8: Find saturation point ──
df_a = pd.DataFrame(ablation_results)
# Find where adding more features stops improving ICIR (>5% improvement)
best_icir = df_a['icir'].max()
saturation_n = None
for i, row in df_a.iterrows():
    if row['icir'] >= best_icir * 0.95:
        saturation_n = row['n']
        break

print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")
print(f"  Best OOS ICIR: {best_icir:.4f} (n={df_a.loc[df_a['icir'].idxmax(), 'n']:.0f} features)")
print(f"  Saturation point (95% of best): n={saturation_n}")
print(f"  All-feature ICIR: {icir_all:.4f}")
print(f"  Feature reduction possible: {len(feat_cols)} -> {saturation_n} ({saturation_n/len(feat_cols):.0%})")

shutil.rmtree(reg_dir)
print("\nDone.")
