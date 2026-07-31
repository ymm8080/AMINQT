#!/usr/bin/env python3
"""Check fina_indicator row count distribution per stock to find sparse stocks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

cache = pd.read_parquet(
    "data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet"
)
cache["symbol"] = cache["ts_code"].str.split(".").str[0]

# Row counts per stock
counts = cache.groupby("symbol").size().sort_values()
print("=== Row count distribution ===")
print(f"Min: {counts.min()}, Max: {counts.max()}, Median: {counts.median()}")
print(f"Stocks with <5 rows: {(counts < 5).sum()}")
print(f"Stocks with <8 rows: {(counts < 8).sum()}")
print(f"Stocks with <10 rows: {(counts < 10).sum()}")

# Show the bottom 30
print("\n=== Bottom 30 stocks by row count ===")
for s, n in counts.head(30).items():
    sub = cache[cache["symbol"] == s]
    anns = sub["ann_date"].tolist()
    print(f"  {s}: {n} rows, ann_dates={anns}")

# Also check: in v3, how many stocks have 0% roe coverage in 2023?
print("\n=== V3 2023 roe coverage ===")
v3 = pd.read_parquet(
    "data/panel_full_enriched_v3.parquet", columns=["symbol", "date", "roe"]
)
v3["year"] = v3["date"].dt.year
v3_2023 = v3[v3["year"] == 2023]
stock_cov = v3_2023.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
zero_cov = stock_cov[stock_cov == 0]
print(f"Stocks with 0% roe coverage in 2023: {len(zero_cov)}")
if len(zero_cov) > 0:
    print(f"  {sorted(zero_cov.index.tolist())}")

low_cov = stock_cov[(stock_cov > 0) & (stock_cov < 50)]
print(f"Stocks with 1-50% roe coverage in 2023: {len(low_cov)}")
