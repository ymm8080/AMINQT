#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1: Create a test registry + enriched panel for dedup testing."""
import pandas as pd, numpy as np, json, tempfile, shutil, os, sys

np.random.seed(42)
sys.path.insert(0, os.getcwd())

with open("data/csi300_constituents.json") as f:
    csi300 = set(json.load(f))

PANEL = "data/panel_full_enriched_v4_20260729.parquet"
panel = pd.read_parquet(PANEL)
cutoff = panel["date"].max() - pd.Timedelta(days=180)
panel = panel[panel["date"] >= cutoff]
in_panel = csi300 & set(panel["symbol"].unique())
stocks = np.random.choice(list(in_panel), size=min(50, len(in_panel)), replace=False)
panel = panel[panel["symbol"].isin(stocks)].copy()

from app.pipeline1.cleaning_pipeline import CleaningPipeline
cleaner = CleaningPipeline()
main_df, _ = cleaner.run_train(panel)

from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import prepare_board_frame, select_features
from app.pipeline1.ic_screener import ICScreener

reg_dir = tempfile.mkdtemp()
reg_path = os.path.join(reg_dir, "feature_registry.json")
registry = FeatureRegistry(path=reg_path)

# Seed registry with feature discovery on small sample
sample = main_df.groupby("symbol", group_keys=False).apply(
    lambda g: g.head(min(30, len(g)))
).reset_index(drop=True)
registry._seed(sample)

features = FeatureEngineV35()
screener = ICScreener(registry_path=reg_dir)

# Generate full enriched dataframe with features
df = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
feat_cols = select_features(df, "main", "dedup_test", screener, registry=registry)

# Save registry
shutil.copy(reg_path, "data/factor_registry/_dedup_test_registry.json")

# Now run features on a smaller panel sample for the dedup script to use
# Sample 200 stocks x 60 days from the enriched df
all_dates = sorted(df["date"].unique())
recent_dates = all_dates[-60:]
df_recent = df[df["date"].isin(recent_dates)].copy()
all_symbols = df_recent["symbol"].unique()
if len(all_symbols) > 200:
    rng = np.random.RandomState(42)
    picked = rng.choice(all_symbols, 200, replace=False)
    df_recent = df_recent[df_recent["symbol"].isin(picked)]

print(f"Enriched panel sample: {df_recent.shape}, {df_recent['symbol'].nunique()} stocks, {df_recent['date'].nunique()} days")
df_recent.to_parquet("data/factor_registry/_dedup_test_enriched_panel.parquet")

print(f"Registry saved with {len(registry.features)} features, {len(registry.get_active())} active")
print(f"Active dim groups: {len(registry.get_active_dim_groups())}")
shutil.rmtree(reg_dir)
