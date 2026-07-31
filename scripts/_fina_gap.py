#!/usr/bin/env python3
"""Analyze fina_indicator NaN gaps by year and by stock."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet(
    "data/panel_full_enriched_v3.parquet",
    columns=["symbol", "date", "roe", "gross_margin", "eps_yoy", "net_margin"],
)

v3["year"] = v3["date"].dt.year

print("=== NaN rate by year ===")
for yr in sorted(v3["year"].unique()):
    sub = v3[v3["year"] == yr]
    n = len(sub)
    roe_na = sub["roe"].isna().mean() * 100
    eps_na = sub["eps_yoy"].isna().mean() * 100
    print(f"  {yr}: {n:>8,} rows | roe {roe_na:5.1f}% NaN | eps_yoy {eps_na:5.1f}% NaN")

# Per-stock: how many stocks have ALL NaN for roe?
print("\n=== Per-stock roe coverage ===")
stock_roe = v3.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
print(f"  Stocks with 0% roe coverage: {(stock_roe == 0).sum()}")
print(f"  Stocks with >0% roe coverage: {(stock_roe > 0).sum()}")
print(f"  Median coverage: {stock_roe.median():.1f}%")

# Those 3 missing stocks
for s in ["300148", "300252", "300390"]:
    sub = v3[v3["symbol"] == s]
    print(f"  {s}: {len(sub)} rows, roe NaN={sub['roe'].isna().mean() * 100:.1f}%")

# The all_ file date range
print("\n=== all_20230701_20260727.parquet date range ===")
all_df = pd.read_parquet(
    "data/supply_cache/alt_data/fina_indicator/all_20230701_20260727.parquet"
)
all_df["ann_date_dt"] = pd.to_datetime(
    all_df["ann_date"], format="%Y%m%d", errors="coerce"
)
print(f"  announce_date: {all_df['ann_date_dt'].min()} ~ {all_df['ann_date_dt'].max()}")
print(f"  report periods: {sorted(all_df['end_date'].unique())[:10]}...")
