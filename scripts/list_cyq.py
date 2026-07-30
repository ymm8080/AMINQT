#!/usr/bin/env python3
"""List all CYQ columns in V3 panel with status."""
import pandas as pd

df = pd.read_parquet("data/panel_full_enriched_v3.parquet")

cyq_cols = [
    "benefit_part", "avg_cost", "weight_avg",
    "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
    "pct_70_low", "pct_70_high", "pct_70_con",
    "pct_90_low", "pct_90_high", "pct_90_con",
]

print(f"V3: {len(df):,} rows, {len(df.columns)} cols")
print()
print(f"{'#':>3s}  {'Column':<15s}  {'Source':<10s}  {'NaN%':>7s}  {'non_null':>12s}  {'Type':>8s}")
print("-" * 65)

sources = {
    "benefit_part": "Tushare",
    "avg_cost":     "Tushare",
    "weight_avg":   "Tushare*",
    "cost_5pct":    "Tushare",
    "cost_15pct":   "Tushare",
    "cost_50pct":   "Tushare",
    "cost_85pct":   "Tushare",
    "cost_95pct":   "Tushare",
    "pct_70_low":   "Derived",
    "pct_70_high":  "Derived",
    "pct_70_con":   "Derived",
    "pct_90_low":   "Derived",
    "pct_90_high":  "Derived",
    "pct_90_con":   "Derived",
}

for i, c in enumerate(cyq_cols, 1):
    if c in df.columns:
        src = sources.get(c, "?")
        nan_pct = df[c].isna().mean() * 100
        nn = df[c].notna().sum()
        dtype = str(df[c].dtype)
        print(f"{i:>3d}  {c:<15s}  {src:<10s}  {nan_pct:>6.2f}%  {nn:>12,}  {dtype:>8s}")
    else:
        print(f"{i:>3d}  {c:<15s}  {'MISSING':>10s}")

print()
print("* weight_avg: derived from avg_cost (same Tushare source,")
print("  fetch_cyq_remaining.py mapped Tushare weight_avg → avg_cost)")
print()
print("Derived columns formula:")
print("  pct_70_low  = cost_15pct          (15th percentile price)")
print("  pct_70_high = cost_85pct          (85th percentile price)")
print("  pct_70_con  = (cost_85 - cost_15) / (cost_85 + cost_15)")
print("  pct_90_low  = cost_5pct           (5th percentile price)")
print("  pct_90_high = cost_95pct          (95th percentile price)")
print("  pct_90_con  = (cost_95 - cost_5)  / (cost_95 + cost_5)")
