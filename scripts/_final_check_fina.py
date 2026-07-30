#!/usr/bin/env python3
"""Final check: remaining gaps, leaked columns, and the '29 stocks' mystery."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=["symbol", "date", "roe"])
v3["year"] = v3["date"].dt.year
v3["half"] = v3["date"].dt.month.apply(lambda m: "H1" if m <= 6 else "H2")

# 1. Check 3 remaining zero-coverage stocks
print("=== 3 stocks with 0% roe in 2023 ===")
v3_2023 = v3[v3["year"] == 2023]
stock_cov = v3_2023.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
zero_cov = stock_cov[stock_cov == 0].index.tolist()

cache_2022 = pd.read_parquet("data/supply_cache/alt_data/fina_indicator/all_20220101_20230701.parquet")
cache_2022["symbol"] = cache_2022["ts_code"].str.split(".").str[0]
cache_2022["ann_dt"] = pd.to_datetime(cache_2022["ann_date"], format="%Y%m%d", errors="coerce")

cache_2023 = pd.read_parquet("data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet")
cache_2023["symbol"] = cache_2023["ts_code"].str.split(".").str[0]
cache_2023["ann_dt"] = pd.to_datetime(cache_2023["ann_date"], format="%Y%m%d", errors="coerce")

for s in zero_cov:
    c22 = cache_2022[cache_2022["symbol"] == s]
    c23 = cache_2023[cache_2023["symbol"] == s]
    print(f"\n  {s}:")
    print(f"    2022 cache: {len(c22)} rows, roe non-NaN={c22['roe'].notna().sum()}")
    if len(c22) > 0:
        print(f"      ann_dates: {sorted(c22['ann_dt'].dt.strftime('%Y-%m-%d').tolist())}")
        print(f"      roe values: {c22.sort_values('ann_dt')['roe'].tolist()}")
    print(f"    2023 cache: {len(c23)} rows, roe non-NaN={c23['roe'].notna().sum()}")
    if len(c23) > 0:
        print(f"      ann_dates: {sorted(c23['ann_dt'].dt.strftime('%Y-%m-%d').tolist())}")
        print(f"      roe values: {c23.sort_values('ann_dt')['roe'].tolist()}")

# 2. Count stocks with poor coverage (>5% NaN in 2023)
print("\n=== Stocks with >5% roe NaN in 2023 ===")
poor = stock_cov[stock_cov < 95]
print(f"Count: {len(poor)}")
if len(poor) > 0:
    print(f"  {sorted(poor.index.tolist())}")

# 3. Count stocks with >10% NaN in 2023
poor10 = stock_cov[stock_cov < 90]
print(f"\nStocks with >10% roe NaN in 2023: {len(poor10)}")

# 4. Count stocks with >50% NaN in 2023
poor50 = stock_cov[stock_cov < 50]
print(f"Stocks with >50% roe NaN in 2023: {len(poor50)}")
if len(poor50) > 0:
    for s in poor50.index:
        print(f"  {s}: {stock_cov[s]:.1f}% coverage")

# 5. Check leaked columns
print("\n=== Leaked columns check ===")
v3_full = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=None)
leaked = [c for c in ["ts_code", "end_date"] if c in v3_full.columns]
print(f"Leaked columns still in v3: {leaked}")
if leaked:
    for c in leaked:
        print(f"  {c}: {v3_full[c].notna().sum()} non-NaN / {len(v3_full)} total")

# 6. H1 vs H2 breakdown
print("\n=== 2023 H1 vs H2 roe NaN ===")
for half in ["H1", "H2"]:
    sub = v3_2023[v3_2023["half"] == half]
    na = sub["roe"].isna().mean() * 100
    print(f"  {half}: {len(sub):>8,} rows, roe NaN={na:.1f}%")
