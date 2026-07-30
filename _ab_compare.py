"""A/B Comparison: Before vs After feature engineering fixes.
Uses subprocess isolation — OLD code and NEW code never share a process."""
import subprocess, sys, os, json, shutil, time

SCRIPT = r"""
import pandas as pd, numpy as np, json, tempfile, shutil, time, os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
import lightgbm as lgb
from scipy.stats import spearmanr

PANEL = "data/panel_full_enriched_v4_20260729.parquet"

import importlib
import app.pipeline1.feature_engine_v35 as fe_mod
import app.pipeline1.ic_screener as ic_mod
importlib.reload(fe_mod)
importlib.reload(ic_mod)

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.cleaning_pipeline import CleaningPipeline

# Load data
with open("data/csi300_constituents.json") as f:
    csi300 = set(json.load(f))
panel = pd.read_parquet(PANEL)
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
in_panel = csi300 & set(panel['symbol'].unique())
panel = panel[panel['symbol'].isin(in_panel)].copy()

cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)

t0 = time.time()
reg_dir = tempfile.mkdtemp()
registry = FeatureRegistry(path=os.path.join(reg_dir, "feature_registry.json"))
sample = main_df.groupby('symbol', group_keys=False).apply(lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)
df = prepare_board_frame(main_df, fe, cross_sectional_rank=False, registry=registry)
feat_cols = select_features(df, "main", "ab", screener, registry=registry)
build_time = time.time() - t0
n_built = len(FeatureEngineV35.feature_columns(df))
n_model = len(feat_cols)

# Train/test
label = "label_pm_1d_net"
if label not in df.columns: label = "label_1d_net"
dates = sorted(df['date'].unique())
split = int(len(dates) * 0.75)
train_df = df[df['date'].isin(dates[:split])].dropna(subset=[label])
test_df = df[df['date'].isin(dates[split:])].dropna(subset=[label])

X_train = train_df[feat_cols].fillna(0)
X_test = test_df[feat_cols].fillna(0)

t_train = time.time()
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1)
model.fit(X_train, train_df[label])
train_time = time.time() - t_train

def oos_ic(preds, df_t, lab):
    df_e = df_t.copy(); df_e['pred'] = preds
    ics = []
    for d, g in df_e.groupby('date'):
        if len(g) < 10: continue
        ic, _ = spearmanr(g['pred'], g[lab])
        if not np.isnan(ic): ics.append(ic)
    a = np.array(ics)
    return float(round(a.mean(), 5)), float(round(a.mean()/a.std() if a.std()>0 else 0, 4))

preds = model.predict(X_test)
ic, icir = oos_ic(preds, test_df, label)

s = registry.summary()
shutil.rmtree(reg_dir)

# Print as JSON for parsing
result = {
    "n_built": n_built, "n_model": n_model,
    "n_strong": s["by_grade"]["strong"], "n_weak": s["by_grade"]["weak"],
    "n_dead": s["by_grade"]["dead"],
    "build_time": round(build_time, 1), "train_time": round(train_time, 1),
    "oos_ic": ic, "oos_icir": icir,
    "n_train_rows": len(train_df), "n_test_rows": len(test_df),
    "n_train_days": len(dates[:split]), "n_test_days": len(dates[split:]),
    "n_stocks": main_df.symbol.nunique(),
}
print("RESULT_JSON:", json.dumps(result))
"""

# ── Run A/B ──
print("=" * 70)
print("A/B COMPARISON")
print("=" * 70)

# Save current (NEW) files
for f in ["app/pipeline1/feature_engine_v35.py", "app/pipeline1/ic_screener.py"]:
    shutil.copy(f, f + ".AB_SAVE")

# Stage any current changes so checkout doesn't complain
subprocess.run(["git", "add", "-A"], capture_output=True)

# ── OLD ──
print("\n[1/2] Checking out OLD code (HEAD~1)...")
subprocess.run(["git", "checkout", "HEAD~1", "--",
    "app/pipeline1/feature_engine_v35.py", "app/pipeline1/ic_screener.py"], check=True)

print("Running OLD experiment...")
t0 = time.time()
old_out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
old_time = time.time() - t0

# Parse OLD result
old_result = None
for line in old_out.stdout.split("\n"):
    if "RESULT_JSON:" in line:
        old_result = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
if old_result is None:
    print("OLD FAILED:", old_out.stderr[-500:])
    # Restore NEW files before exiting
    for f in ["app/pipeline1/feature_engine_v35.py", "app/pipeline1/ic_screener.py"]:
        shutil.move(f + ".AB_SAVE", f)
    sys.exit(1)

print(f"OLD: {old_result['n_built']} built, {old_result['n_model']} into model, "
      f"OOS ICIR={old_result['oos_icir']:.4f} ({old_time:.0f}s)")

# ── NEW ──
print("\n[2/2] Restoring NEW code...")
subprocess.run(["git", "checkout", "HEAD", "--",
    "app/pipeline1/feature_engine_v35.py", "app/pipeline1/ic_screener.py"], check=True)
# Restore our working changes
for f in ["app/pipeline1/feature_engine_v35.py", "app/pipeline1/ic_screener.py"]:
    if os.path.exists(f + ".AB_SAVE"):
        shutil.move(f + ".AB_SAVE", f)

# Clear __pycache__ so Python reloads the new .py files
import glob
for cache_dir in glob.glob("app/pipeline1/__pycache__"):
    for f in os.listdir(cache_dir):
        if "feature_engine" in f or "ic_screener" in f:
            os.remove(os.path.join(cache_dir, f))

print("Running NEW experiment...")
t0 = time.time()
new_out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=600)
new_time = time.time() - t0

new_result = None
for line in new_out.stdout.split("\n"):
    if "RESULT_JSON:" in line:
        new_result = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
if new_result is None:
    print("NEW FAILED:", new_out.stderr[-500:])
    sys.exit(1)

print(f"NEW: {new_result['n_built']} built, {new_result['n_model']} into model, "
      f"OOS ICIR={new_result['oos_icir']:.4f} ({new_time:.0f}s)")

# ── REPORT ──
print("\n" + "=" * 70)
print("A/B COMPARISON RESULTS")
print(f"Stocks: {old_result['n_stocks']} | Train: {old_result['n_train_days']}d | Test: {old_result['n_test_days']}d")
print("=" * 70)
print(f"{'Metric':<30} {'OLD (before)':>15} {'NEW (after)':>15} {'Change':>15}")
print("-" * 75)

comparisons = [
    ("Features built", "n_built"),
    ("Features into model", "n_model"),
    ("Strong features", "n_strong"),
    ("Weak features", "n_weak"),
    ("Dead features", "n_dead"),
    ("Build time (s)", "build_time"),
    ("Train time (s)", "train_time"),
    ("OOS IC", "oos_ic"),
    ("OOS ICIR", "oos_icir"),
]

for label, key in comparisons:
    old_v = old_result[key]
    new_v = new_result[key]
    if isinstance(old_v, (int, float)) and old_v != 0:
        pct = (new_v - old_v) / abs(old_v) * 100
        change = f"{pct:+.0f}%"
    else:
        change = ""
    print(f"{label:<30} {str(old_v):>15} {str(new_v):>15} {change:>15}")

# Verdict
old_ir = old_result["oos_icir"]
new_ir = new_result["oos_icir"]
print(f"\nVerdict: ", end="")
if old_ir != 0:
    delta_pct = (new_ir - old_ir) / abs(old_ir) * 100
    if delta_pct > 5:
        print(f"NEW IMPROVES prediction (OOS ICIR {old_ir:.4f} -> {new_ir:.4f}, +{delta_pct:.0f}%)")
    elif delta_pct > -5:
        print(f"PRESERVES prediction quality (OOS ICIR {old_ir:.4f} -> {new_ir:.4f}, {delta_pct:+.0f}%)")
    else:
        print(f"MAY DEGRADE prediction (OOS ICIR {old_ir:.4f} -> {new_ir:.4f}, {delta_pct:.0f}%) — investigate")
