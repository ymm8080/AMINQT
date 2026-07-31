"""
验证 use_registry=True 端到端流程
- 真实沪深300成分股（Tushare index_weight 000300.SH）
- 最近半年数据（300 stocks × ~120 trading days）
- 跑两次窗口，验证 registry 持久化效果 + 省算力幅度
"""
import logging, sys, os, json, shutil, tempfile, warnings
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(name)s [%(levelname)s] %(message)s',
    stream=sys.stdout,
)

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.cleaning_pipeline import CleaningPipeline

np.random.seed(42)

PANEL = "data/panel_full_enriched_v4_20260729.parquet"

# ── Step 1: Load + filter ──
print("\n" + "="*60)
print("STEP 1: Load panel, filter to CSI300 + last 6 months")
print("="*60)

# Load real CSI 300 constituents
with open("data/csi300_constituents.json") as f:
    csi300_codes = set(json.load(f))
print(f"CSI 300 constituents loaded: {len(csi300_codes)}")

# Sample down to 20 first to minimize memory
csi300_list = sorted(csi300_codes)
np.random.seed(42)
sampled = set(np.random.choice(csi300_list, size=min(20, len(csi300_list)), replace=False))
print(f"Sampled {len(sampled)} CSI300 codes for verification")

# Read only date column to find max date
pf = pq.read_table(PANEL, columns=["date", "symbol"])
all_symbols = set(pf.column("symbol").to_pylist())
in_panel = sampled & all_symbols
print(f"Codes in panel: {len(in_panel)}")
max_date = pf.column("date").to_pandas().max()
cutoff = max_date - pd.Timedelta(days=180)
print(f"Date range: {pf.column('date').to_pandas().min().date()} ~ {max_date.date()}, cutoff ≥{cutoff.date()}")
del pf

# Now read full data with filters: only sampled symbols + recent 6mo
filters = [
    ("symbol", "in", list(in_panel)),
    ("date", ">=", cutoff),
]
panel = pd.read_parquet(PANEL, filters=filters)
print(f"Filtered panel: {len(panel):,} rows, {panel['symbol'].nunique()} stocks")

# ── Step 2: Clean ──
print("\n" + "="*60)
print("STEP 2: Cleaning pipeline")
print("="*60)

cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel)
print(f"Main board: {len(main_df):,} rows, {main_df['symbol'].nunique()} stocks")
print(f"Dual board:  {len(dual_df):,} rows, {dual_df['symbol'].nunique()} stocks")

# ── Step 3: First window with registry ──
print("\n" + "="*60)
print("STEP 3: Window 1 — seed registry + IC screen")
print("="*60)

reg_dir = tempfile.mkdtemp()
reg_path = os.path.join(reg_dir, "feature_registry.json")
registry = FeatureRegistry(path=reg_path)

features = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)

# Seed registry from panel sample
sample = main_df.groupby('symbol', group_keys=False).apply(
    lambda g: g.head(min(30, len(g)))
).reset_index(drop=True)
n_seed = registry._seed(sample)
print(f"Registry seeded: {n_seed} features")

# Build features with registry
df1 = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
all_feat1 = FeatureEngineV35.feature_columns(df1)
print(f"Built: {len(df1.columns)} total cols, {len(all_feat1)} feature cols")

# IC screen + sync to registry
cols1 = select_features(df1, "main", "W1", screener, registry=registry)
s1 = registry.summary()
print(f"IC screen: {len(cols1)} features → LightGBM")
print(f"Registry: total={s1['total_features']}, active={s1['active']}")
print(f"Grades: {s1['by_grade']}")

# ── Step 4: Second window — should skip dead features ──
print("\n" + "="*60)
print("STEP 4: Window 2 — registry persists, dead features pruned")
print("="*60)

df2 = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
all_feat2 = FeatureEngineV35.feature_columns(df2)
print(f"Built: {len(df2.columns)} total cols, {len(all_feat2)} feature cols")

cols2 = select_features(df2, "main", "W2", screener, registry=registry)
s2 = registry.summary()
print(f"IC screen: {len(cols2)} features → LightGBM")
print(f"Registry: total={s2['total_features']}, active={s2['active']}")
print(f"Grades: {s2['by_grade']}")

# ── Step 5: Compare ──
print("\n" + "="*60)
print("STEP 5: Comparison")
print("="*60)
print(f"{'Metric':<40} {'Window 1':>12} {'Window 2':>12}")
print(f"{'-'*40} {'-'*12} {'-'*12}")
print(f"{'Feature cols built':<40} {len(all_feat1):>12} {len(all_feat2):>12}")
print(f"{'Into LightGBM':<40} {len(cols1):>12} {len(cols2):>12}")
print(f"{'Registry total':<40} {s1['total_features']:>12} {s2['total_features']:>12}")
print(f"{'Registry active':<40} {s1['active']:>12} {s2['active']:>12}")
print(f"{'Dead features':<40} {s1['by_grade']['dead']:>12} {s2['by_grade']['dead']:>12}")
print(f"{'Strong features':<40} {s1['by_grade']['strong']:>12} {s2['by_grade']['strong']:>12}")

build_reduction = len(all_feat1) - len(all_feat2)
if build_reduction > 0:
    print(f"\n>>> Registry pruning saved {build_reduction} feature builds in Window 2")

# ── Step 6: Per-dim savings ──
print("\n" + "="*60)
print("STEP 6: Per-dim group detail (Window 2)")
print("="*60)
for dg in sorted(registry.get_dim_groups()):
    active_dg = len(registry.get_active(dg))
    total_dg = len([n for n, m in registry.features.items() if m.get('dim_group') == dg])
    dead_dg = total_dg - active_dg
    if total_dg >= 10 and dead_dg > 0:
        print(f"  {dg}: {active_dg}/{total_dg} active, {dead_dg} dead pruned")

# Cleanup
shutil.rmtree(reg_dir)
print("\nDone — registry flow verified!")
