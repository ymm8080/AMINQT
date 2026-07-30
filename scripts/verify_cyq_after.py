#!/usr/bin/env python3
"""Verify CYQ column fill rates after derivation."""
import pandas as pd

df = pd.read_parquet("data/panel_full_enriched_v3.parquet")
print(f"V3: {len(df):,} rows, {len(df.columns)} cols")
print()

cyq = [
    "benefit_part", "avg_cost",
    "pct_70_low", "pct_70_high", "pct_70_con",
    "pct_90_low", "pct_90_high", "pct_90_con",
    "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
    "weight_avg",
]

print(f"{'Column':<15s}  {'NaN%':>7s}  {'non_null':>12s}")
print("-" * 38)
for c in cyq:
    if c in df.columns:
        print(f"{c:<15s}  {df[c].isna().mean()*100:>6.2f}%  {df[c].notna().sum():>12,}")
    else:
        print(f"{c:<15s}  MISSING")
