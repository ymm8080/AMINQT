#!/usr/bin/env python3
"""Check fina_indicator coverage in v3 vs cache."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet")
v3_stocks = set(v3["symbol"].unique())

# Load all fina cache
fina_dir = "data/supply_cache/alt_data/fina_indicator"
files = sorted(f for f in os.listdir(fina_dir) if f.endswith(".parquet"))
all_stocks = set()
per_stocks = set()
for f in files:
    df = pd.read_parquet(os.path.join(fina_dir, f), columns=["symbol"])
    syms = set(df["symbol"].unique())
    all_stocks.update(syms)
    if not f.startswith("all_"):
        per_stocks.update(syms)

print(f"V3 stocks: {len(v3_stocks)}")
print(f"Cache files: {len(files)}")
print(f"  all_ file stocks: {len(all_stocks - per_stocks)}")
print(f"  per-stock files: {len(per_stocks)}")
print(f"  union: {len(all_stocks)}")

# Missing stocks
missing = v3_stocks - all_stocks
print(f"\nV3 stocks NOT in cache: {len(missing)}")
if missing:
    print(f"  examples: {sorted(missing)[:20]}")

# Per-column NaN rates
fina_cols = [
    "roe",
    "roe_deducted",
    "roa",
    "gross_margin",
    "rev_yoy",
    "debt_ratio",
    "current_ratio",
    "asset_turnover",
    "ar_turnover",
    "inventory_turnover",
    "ocf_to_or",
    "eps_yoy",
    "profit_yoy",
    "net_margin",
    "ocfps",
    "revenue_ps",
    "bps",
    "eps",
    "dt_eps",
    "roe_yoy",
    "q_roe",
    "q_ocf_to_sales",
    "announce_date",
]
print(f"\nPer-column NaN rates (v3 has {len(v3)} rows):")
for c in fina_cols:
    if c in v3.columns:
        na = v3[c].isna().mean() * 100
        print(f"  {c:20s}: {na:5.1f}% NaN")
    else:
        print(f"  {c:20s}: MISSING from v3")
