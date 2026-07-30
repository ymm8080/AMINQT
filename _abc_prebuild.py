"""
Pre-build features for ABC test — prepare_board_frame() only, NO IC screening.
Saves full feature pool (all 420+ features) to parquet for all 12 agents to share.
"""
import time, warnings, json, os, tempfile, shutil
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
np.random.seed(42)

PANEL_PATH = "data/panel_full_enriched_v4_20260729.parquet"
CSI300_PATH = "data/csi300_constituents.json"
OUT_DIR = "data/abc_test_results"
os.makedirs(OUT_DIR, exist_ok=True)

t0 = time.time()

# ── Load ──
print("Loading panel...")
panel = pd.read_parquet(PANEL_PATH)
cutoff = panel['date'].max() - pd.Timedelta(days=365)
panel = panel[panel['date'] >= cutoff]
print(f"  Panel: {len(panel):,} rows, {panel.symbol.nunique()} stocks, {panel.date.min().date()}~{panel.date.max().date()}")

from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()

# ═══════════════════════════════════════
# MAIN BOARD
# ═══════════════════════════════════════
print("\n=== MAIN BOARD (CSI300) ===")
with open(CSI300_PATH) as f:
    csi300 = set(json.load(f))
main_panel = panel[panel['symbol'].isin(csi300 & set(panel['symbol'].unique()))].copy()
print(f"  CSI300: {len(main_panel):,} rows, {main_panel.symbol.nunique()} stocks")

t1 = time.time()
main_df, _ = cleaner.run_train(main_panel)
print(f"  Clean: {time.time()-t1:.1f}s -> {len(main_df):,} rows, {main_df.symbol.nunique()} stocks")

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame

t1 = time.time()
reg_dir = tempfile.mkdtemp()
registry = FeatureRegistry(path=os.path.join(reg_dir, "feature_registry.json"))
sample = main_df.groupby('symbol', group_keys=False).apply(lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
registry._seed(sample)

fe = FeatureEngineV35()
df_main = prepare_board_frame(main_df, fe, cross_sectional_rank=False, registry=registry)
n_feat = len(FeatureEngineV35.feature_columns(df_main))
print(f"  Build: {time.time()-t1:.0f}s -> {len(df_main.columns)} total cols, {n_feat} feature cols")

# Check label availability
for lbl in ['label_pm_1d_net', 'label_1d_net', 'label_pm_1d', 'label_1d']:
    if lbl in df_main.columns:
        nn = df_main[lbl].notna().sum()
        print(f"  Label {lbl}: {nn:,} non-NaN ({nn/len(df_main):.0%})")

out_main = os.path.join(OUT_DIR, "prebuilt_main.parquet")
df_main.to_parquet(out_main)
sz_mb = os.path.getsize(out_main)/1024/1024
print(f"  Saved: {out_main} ({sz_mb:.0f} MB)")

shutil.rmtree(reg_dir)

# ═══════════════════════════════════════
# DUAL BOARD
# ═══════════════════════════════════════
print("\n=== DUAL BOARD (random 300 GEM/STAR) ===")
dual_panel = panel[panel['board'].isin(['GEM','STAR'])].copy()
if dual_panel['symbol'].nunique() > 300:
    stocks = np.random.choice(dual_panel['symbol'].unique(), size=300, replace=False)
    dual_panel = dual_panel[dual_panel['symbol'].isin(stocks)]
print(f"  Dual: {len(dual_panel):,} rows, {dual_panel.symbol.nunique()} stocks")

t1 = time.time()
main_d, dual_d = cleaner.run_train(dual_panel)
dual_df = dual_d if len(dual_d) > len(main_d) else main_d
bname = 'dual' if len(dual_d) > len(main_d) else 'main'
print(f"  Clean: {time.time()-t1:.1f}s -> {len(dual_df):,} rows [{bname}], {dual_df.symbol.nunique()} stocks")

t1 = time.time()
reg_dir2 = tempfile.mkdtemp()
registry2 = FeatureRegistry(path=os.path.join(reg_dir2, "feature_registry.json"))
sample2 = dual_df.groupby('symbol', group_keys=False).apply(lambda g: g.head(min(30, len(g)))).reset_index(drop=True)
registry2._seed(sample2)

fe2 = FeatureEngineV35()
df_dual = prepare_board_frame(dual_df, fe2, cross_sectional_rank=True, registry=registry2)
n_feat_d = len(FeatureEngineV35.feature_columns(df_dual))
print(f"  Build: {time.time()-t1:.0f}s -> {len(df_dual.columns)} total cols, {n_feat_d} feature cols")

for lbl in ['label_pm_1d_net', 'label_1d_net', 'label_pm_1d', 'label_1d']:
    if lbl in df_dual.columns:
        nn = df_dual[lbl].notna().sum()
        print(f"  Label {lbl}: {nn:,} non-NaN ({nn/len(df_dual):.0%})")

out_dual = os.path.join(OUT_DIR, "prebuilt_dual.parquet")
df_dual.to_parquet(out_dual)
sz_mb2 = os.path.getsize(out_dual)/1024/1024
print(f"  Saved: {out_dual} ({sz_mb2:.0f} MB)")

shutil.rmtree(reg_dir2)

print(f"\n{'='*60}")
print(f"DONE in {time.time()-t0:.0f}s")
print(f"  Main: {n_feat} features, {sz_mb:.0f} MB")
print(f"  Dual: {n_feat_d} features, {sz_mb2:.0f} MB")
print(f"  Ready for 12-agent ABC test.")
