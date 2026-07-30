#!/usr/bin/env python3
"""Check fina_indicator coverage: identify missing stocks and 2023 gaps."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

# Load v3 stock list
v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=["symbol"])
v3_stocks = set(v3["symbol"].unique())
print(f"V3 stocks: {len(v3_stocks)}")

# Load new cache (from fetch_fina_2023.py)
new_cache = pd.read_parquet(
    "data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet"
)
print(f"New cache columns: {new_cache.columns.tolist()}")
print(f"New cache: {len(new_cache)} rows")
new_cache["symbol"] = new_cache["ts_code"].str.split(".").str[0]
new_stocks = set(new_cache["symbol"].unique())
print(f"New cache stocks: {len(new_stocks)}")

# Check early 2023 coverage
new_cache["ann_dt"] = pd.to_datetime(new_cache["ann_date"], format="%Y%m%d", errors="coerce")
early = new_cache[new_cache["ann_dt"] < "2023-07-01"]
print(f"Pre-2023-07-01 rows: {len(early)}, stocks: {early['ts_code'].nunique()}")

# Missing from new cache
missing_new = v3_stocks - new_stocks
print(f"\nMissing from new cache: {len(missing_new)} stocks")
if missing_new:
    print(f"  {sorted(missing_new)}")

# Also check ALL cache files (per-stock files too)
fina_dir = "data/supply_cache/alt_data/fina_indicator"
all_cache_stocks = set(new_stocks)
per_files = [
    f for f in os.listdir(fina_dir) if f.endswith(".parquet") and not f.startswith("all_")
]
print(f"\nPer-stock files: {len(per_files)}")
for f in per_files:
    try:
        df = pd.read_parquet(os.path.join(fina_dir, f), columns=["symbol"])
        all_cache_stocks.update(df["symbol"].unique())
    except Exception as e:
        print(f"  {f}: ERROR {e}")

print(f"Total cache stocks (all_ + per-stock): {len(all_cache_stocks)}")
missing_all = v3_stocks - all_cache_stocks
print(f"Missing from ALL cache: {len(missing_all)} stocks")
if missing_all:
    print(f"  {sorted(missing_all)}")
