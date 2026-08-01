# -*- coding: utf-8 -*-
"""Analyze the 31 NaN columns in v3 panel."""
import pyarrow.parquet as pq
import pandas as pd

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"

pairs = [
    ("winner_ratio_x", "winner_ratio_y"),
    ("avg_cost_x", "avg_cost_y"),
    ("cost_5pct_x", "cost_5pct_y"),
    ("cost_15pct_x", "cost_15pct_y"),
    ("cost_50pct_x", "cost_50pct_y"),
    ("cost_85pct_x", "cost_85pct_y"),
    ("cost_95pct_x", "cost_95pct_y"),
    ("pct_70_low_x", "pct_70_low_y"),
    ("pct_70_high_x", "pct_70_high_y"),
    ("pct_70_con_x", "pct_70_con_y"),
    ("pct_90_low_x", "pct_90_low_y"),
    ("pct_90_high_x", "pct_90_high_y"),
    ("pct_90_con_x", "pct_90_con_y"),
    ("weight_avg_x", "weight_avg_y"),
]
cols = []
for x, y in pairs:
    cols.extend([x, y])
cols.extend(["up_limit_raw", "down_limit_raw", "announce_date"])

print("Reading panel...")
t = pq.read_table(PANEL, columns=cols)
df = t.to_pandas()

print(f"{'Col':<20s} {'NaN%':>7s} {'non_null':>12s}")
print("-" * 42)
for x, y in pairs:
    print(f"{x:<20s} {df[x].isna().mean()*100:>6.2f}% {df[x].notna().sum():>12,}")
    print(f"{y:<20s} {df[y].isna().mean()*100:>6.2f}% {df[y].notna().sum():>12,}")
    both = (df[x].notna() & df[y].notna()).sum()
    only_x = (df[x].notna() & df[y].isna()).sum()
    only_y = (df[x].isna() & df[y].notna()).sum()
    print(f"  -> both:{both:,}  only_x:{only_x:,}  only_y:{only_y:,}")
    # Check if values match when both present
    if both > 0:
        diff = (df[x] - df[y]).abs().dropna()
        print(f"  -> max_diff:{diff.max():.6f}  mean_diff:{diff.mean():.6f}")
    print()

for c in ["up_limit_raw", "down_limit_raw", "announce_date"]:
    print(f"{c:<20s} {df[c].isna().mean()*100:>6.2f}% {df[c].notna().sum():>12,}")
