"""Compare OOS performance: recent production models vs new-code model."""
import pickle, json, warnings, time, os, tempfile, shutil
import pandas as pd, numpy as np
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
np.random.seed(42)

print("=" * 70)
print("PREDICTION QUALITY COMPARISON")
print("=" * 70)

# ── Load production models ──
models_to_test = {}
for path, label in [
    ("models/pipeline1/main_2026W31.pkl", "W31 (production)"),
    ("models/pipeline1/main_2026W30.pkl", "W30 (production)"),
    ("models/pipeline1/main_20260728_132207.pkl", "Jul28 (yesterday)"),
    ("models/pipeline1/main_current.pkl", "current"),
]:
    if os.path.exists(path):
        try:
            m = pickle.load(open(path, "rb"))
            models_to_test[label] = m
            print(f"Loaded {label}: {len(m['feature_cols'])} features, {len(m['models'])} sub-models")
        except Exception as e:
            print(f"Skip {label}: {e}")

# ── Load test data (last 30 trading days of CSI300) ──
with open("data/csi300_constituents.json") as f:
    csi300 = set(json.load(f))

panel = pd.read_parquet("data/panel_full_enriched_v4_20260729.parquet")
cutoff = panel['date'].max() - pd.Timedelta(days=60)  # 2 months
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
panel = panel[panel['symbol'].isin(in_panel)].copy()

from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)

# Build features with NEW code for evaluation
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame
from app.pipeline1.ic_screener import ICScreener

reg_dir = tempfile.mkdtemp()
registry = FeatureRegistry(path=os.path.join(reg_dir, "feature_registry.json"))
sample = main_df.groupby('symbol', group_keys=False).apply(
    lambda g: g.head(min(30, len(g)))
).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
df = prepare_board_frame(main_df, fe, cross_sectional_rank=False, registry=registry)

# ── Test: last 30 trading days ──
label = "label_pm_1d_net"

# If labels missing, compute simple forward return
if label not in df.columns:
    df[label] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1)/x - 1)
    df = df.dropna(subset=[label])

dates = sorted(df['date'].unique())
test_dates = dates[-30:]
test_df = df[df['date'].isin(test_dates)].copy()
print(f"\nTest data: {len(test_df):,} rows, {test_df.symbol.nunique()} stocks, {len(test_dates)} trading days")

# ── Evaluate each model ──
def oos_eval(model_obj, X_test, y_test, test_df):
    """Compute daily Rank IC and ICIR."""
    preds = model_obj.predict(X_test)
    df_e = test_df.copy()
    df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[label])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    ic = a.mean()
    icir = abs(ic) / a.std() if a.std() > 0 else 0
    return round(float(ic), 5), round(float(icir), 4)

print(f"\n{'='*70}")
print(f"OOS Evaluation on last 30 trading days")
print(f"{'='*70}")
print(f"{'Model':<25} {'Features':>8} {'OOS IC':>10} {'OOS ICIR':>10} {'IC>0 days':>10}")
print("-" * 65)

all_results = {}
for model_name, model_data in models_to_test.items():
    feature_cols = model_data['feature_cols']
    # Get the regression model
    reg_key = None
    for k in model_data['models']:
        if 'reg' in k or '1d' in k:
            reg_key = k
            break
    if reg_key is None:
        reg_key = list(model_data['models'].keys())[0]

    lgb_model, _ = model_data['models'][reg_key]

    # Find available features in current panel
    avail_feats = [c for c in feature_cols if c in df.columns]
    missing = len(feature_cols) - len(avail_feats)

    if len(avail_feats) < 5:
        all_results[model_name] = {"error": f"only {len(avail_feats)} features available"}
        print(f"{model_name:<25} {'SKIP':>8}  (only {len(avail_feats)}/{len(feature_cols)} features in panel)")
        continue

    X_test = test_df[avail_feats].fillna(0)
    y_test = test_df[label]

    try:
        ic, icir = oos_eval(lgb_model, X_test, y_test, test_df)
        # Count positive IC days
        df_e = test_df.copy()
        df_e['pred'] = lgb_model.predict(X_test)
        pos_days = 0
        for d, g in df_e.groupby('date'):
            if len(g) < 10: continue
            ic_d, _ = spearmanr(g['pred'], g[label])
            if not np.isnan(ic_d) and ic_d > 0: pos_days += 1

        all_results[model_name] = {"ic": ic, "icir": icir, "pos": pos_days, "n_days": len(test_dates),
                                    "n_feat": len(avail_feats), "missing": missing}
        print(f"{model_name:<25} {len(avail_feats):>8} {ic:>+10.5f} {icir:>10.4f} {pos_days:>8}/{len(test_dates)}")
    except Exception as e:
        all_results[model_name] = {"error": str(e)[:80]}
        print(f"{model_name:<25} {'ERROR':>8}  {str(e)[:60]}")

# ── Train NEW model for comparison ──
print(f"\n{'='*70}")
print(f"Training NEW model (all fixes) for comparison...")
print(f"{'='*70}")

from app.pipeline1.train_runner import select_features
screener = ICScreener(registry_path=reg_dir)
new_feat_cols = select_features(df, "main", "compare", screener, registry=registry)

dates_all = sorted(df['date'].unique())
split = int(len(dates_all) * 0.75)
train_df = df[df['date'].isin(dates_all[:split])].dropna(subset=[label])
test_new_df = df[df['date'].isin(dates_all[split:])].dropna(subset=[label])

import lightgbm as lgb
X_train = train_df[new_feat_cols].fillna(0)
y_train = train_df[label]
X_new_test = test_new_df[new_feat_cols].fillna(0)
y_new_test = test_new_df[label]

t0 = time.time()
new_model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
new_model.fit(X_train, y_train)
train_t = time.time() - t0

ic_new, icir_new = oos_eval(new_model, X_new_test, y_new_test, test_new_df)
print(f"NEW model: {len(new_feat_cols)} features, train {train_t:.0f}s")
print(f"  OOS IC={ic_new:+.5f}, ICIR={icir_new:.4f}")

# Also eval NEW model on same 30-day test as production models
X_new_same = test_df[new_feat_cols].fillna(0)
ic_new2, icir_new2 = oos_eval(new_model, X_new_same, test_df[label], test_df)
print(f"  On same 30d test: IC={ic_new2:+.5f}, ICIR={icir_new2:.4f}")

shutil.rmtree(reg_dir)

# ── Summary ──
print(f"\n{'='*70}")
print(f"SUMMARY: OOS ICIR Comparison (30-day test)")
print(f"{'='*70}")
for name, r in sorted(all_results.items(), key=lambda x: -(x[1].get('icir', -999) if isinstance(x[1], dict) else -999)):
    if isinstance(r, dict) and 'icir' in r:
        print(f"  {name:<25} ICIR={r['icir']:.4f}  ({r['n_feat']} features, {r['pos']}/{r['n_days']} pos days)")
print(f"  {'NEW (our fixes)':<25} ICIR={icir_new2:.4f}  ({len(new_feat_cols)} features)")
