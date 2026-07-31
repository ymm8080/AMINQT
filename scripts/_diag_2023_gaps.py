#!/usr/bin/env python3
"""Deep diagnosis: where are the 2023 NaN roe rows? Before or after July 2023?"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet(
    "data/panel_full_enriched_v3.parquet", columns=["symbol", "date", "roe"]
)
v3["year"] = v3["date"].dt.year
v3["half"] = v3["date"].dt.month.apply(lambda m: "H1" if m <= 6 else "H2")

print("=== 2023 roe NaN breakdown ===")
v3_2023 = v3[v3["year"] == 2023]
for half in ["H1", "H2"]:
    sub = v3_2023[v3_2023["half"] == half]
    na = sub["roe"].isna().mean() * 100
    print(
        f"  2023 {half}: {len(sub):>8,} rows, roe NaN={sub['roe'].isna().sum():>7,} ({na:.1f}%)"
    )

# Check how many stocks have data in H1 2023
h1_stocks = v3_2023[v3_2023["half"] == "H1"]["symbol"].nunique()
h2_stocks = v3_2023[v3_2023["half"] == "H2"]["symbol"].nunique()
print(f"\n  H1 stocks: {h1_stocks}, H2 stocks: {h2_stocks}")

# Check date distribution in H1
h1 = v3_2023[v3_2023["half"] == "H1"]
print(f"\n  H1 date range: {h1['date'].min()} ~ {h1['date'].max()}")
print(f"  H1 unique dates: {h1['date'].nunique()}")
print("  H1 rows per date (top 5):")
for d, n in h1.groupby("date").size().sort_values(ascending=False).head(5).items():
    print(f"    {d}: {n} stocks")

# Check the 12 stocks with 0% roe in 2023
print("\n=== 12 stocks with 0% roe in 2023 ===")
stock_cov = v3_2023.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
zero_cov = stock_cov[stock_cov == 0].index.tolist()
for s in zero_cov:
    sub = v3_2023[v3_2023["symbol"] == s]
    print(
        f"  {s}: {len(sub)} rows in 2023, dates {sub['date'].min()} ~ {sub['date'].max()}"
    )

# Check the new cache for these stocks
cache = pd.read_parquet(
    "data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet"
)
cache["symbol"] = cache["ts_code"].str.split(".").str[0]
cache["ann_dt"] = pd.to_datetime(cache["ann_date"], format="%Y%m%d", errors="coerce")

print("\n=== Cache data for 12 zero-coverage stocks ===")
for s in zero_cov:
    sub = cache[cache["symbol"] == s]
    if len(sub) > 0:
        roe_non_nan = sub["roe"].notna().sum()
        print(
            f"  {s}: {len(sub)} rows, roe non-NaN={roe_non_nan}, earliest ann={sub['ann_dt'].min()}, earliest roe={sub.sort_values('ann_dt')['roe'].iloc[0]}"
        )
    else:
        print(f"  {s}: NO cache data")

# Check H1 2023 NaN by stock
print("\n=== H1 2023: stocks with all-NaN roe ===")
h1_data = v3_2023[v3_2023["half"] == "H1"]
h1_cov = h1_data.groupby("symbol")["roe"].apply(lambda x: x.notna().mean() * 100)
h1_zero = h1_cov[h1_cov == 0]
print(f"  Count: {len(h1_zero)}")
if len(h1_zero) > 0:
    print(f"  Stocks: {sorted(h1_zero.index.tolist())[:30]}...")
