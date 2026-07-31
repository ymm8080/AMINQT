#!/usr/bin/env python3
"""Final summary of v3 panel after fina_indicator fill."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet")
v3["year"] = v3["date"].dt.year

print("=" * 80)
print("V3 Panel Final Summary")
print("=" * 80)
print(f"Rows: {len(v3):,}, Cols: {len(v3.columns)}, Stocks: {v3['symbol'].nunique()}")
print(f"Date range: {v3['date'].min()} ~ {v3['date'].max()}")
print(f"Has ts_code: {'ts_code' in v3.columns}")
print(f"Has end_date: {'end_date' in v3.columns}")

fina_cols = [
    c
    for c in [
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
        "net_margin",
        "eps_yoy",
        "profit_yoy",
        "ocfps",
        "revenue_ps",
        "bps",
        "eps",
        "dt_eps",
        "roe_yoy",
        "q_roe",
        "q_ocf_to_sales",
    ]
    if c in v3.columns
]
print(f"Fina cols present: {len(fina_cols)}/{22}")

print("\nNaN rate by year (key columns):")
key_cols = ["roe", "eps_yoy", "gross_margin", "net_margin", "profit_yoy", "debt_ratio"]
for yr in sorted(v3["year"].unique()):
    sub = v3[v3["year"] == yr]
    parts = [f"{c}={sub[c].isna().mean() * 100:.1f}%" for c in key_cols]
    print(f"  {yr}: {len(sub):>8,} rows | {' | '.join(parts)}")

print("\nOverall NaN rate (all fina cols):")
for c in fina_cols:
    na = v3[c].isna().mean() * 100
    print(f"  {c:20s}: {na:.1f}% NaN")
