#!/usr/bin/env python3
"""Check which stocks have no 2023 fina_indicator data in the new cache."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

# Load v3
v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=["symbol", "date"])
v3_stocks = set(v3["symbol"].unique())
v3["year"] = v3["date"].dt.year
v3_2023_stocks = set(v3[v3["year"] == 2023]["symbol"].unique())
print(f"V3 stocks with 2023 data: {len(v3_2023_stocks)}")

# Load new cache
cache = pd.read_parquet(
    "data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet"
)
cache["symbol"] = cache["ts_code"].str.split(".").str[0]
cache["ann_dt"] = pd.to_datetime(cache["ann_date"], format="%Y%m%d", errors="coerce")
cache["ann_year"] = cache["ann_dt"].dt.year

# Stocks with 2023 announcements
cache_2023_stocks = set(cache[cache["ann_year"] == 2023]["symbol"].unique())
print(f"Cache stocks with 2023 announcements: {len(cache_2023_stocks)}")

# Stocks in v3 2023 but NOT in cache 2023
missing_2023 = v3_2023_stocks - cache_2023_stocks
print(f"\nV3 2023 stocks with NO 2023 fina data: {len(missing_2023)}")
if missing_2023:
    # Check if these stocks have ANY fina data at all
    for s in sorted(missing_2023):
        sub = cache[cache["symbol"] == s]
        print(f"  {s}: {len(sub)} rows, earliest ann={sub['ann_dt'].min()}, periods={sub['end_date'].tolist()[:3]}")

# Also check: stocks where roe is ALL NaN in cache
print("\n--- Stocks with ALL NaN roe in cache ---")
roe_by_stock = cache.groupby("symbol")["roe"].apply(lambda x: x.notna().sum())
zero_roe = roe_by_stock[roe_by_stock == 0]
print(f"Count: {len(zero_roe)}")
if len(zero_roe) > 0:
    print(f"  {sorted(zero_roe.index.tolist())}")

# Check stocks with very few rows (<=2)
row_counts = cache.groupby("symbol").size()
few_rows = row_counts[row_counts <= 2]
print(f"\n--- Stocks with <=2 fina rows ---")
print(f"Count: {len(few_rows)}")
if len(few_rows) > 0:
    for s in few_rows.index.tolist():
        sub = cache[cache["symbol"] == s]
        print(f"  {s}: {len(sub)} rows, ann_dates={sub['ann_date'].tolist()}, roe={sub['roe'].tolist()}")
