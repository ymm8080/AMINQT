#!/usr/bin/env python3
"""Verify that pct_70_low == cost_15pct etc. in the baostock panel
(which has all CYQ columns from the local calculator)."""
import pandas as pd
import numpy as np

df = pd.read_parquet("data/panel_full.parquet")
df = df.dropna(subset=[
    "cost_15pct", "pct_70_low",
    "cost_85pct", "pct_70_high",
    "cost_5pct", "pct_90_low",
    "cost_95pct", "pct_90_high",
    "weight_avg", "avg_cost",
])
print(f"Rows with all CYQ columns non-null: {len(df):,}")
print()

pairs = [
    ("pct_70_low",  "cost_15pct"),
    ("pct_70_high", "cost_85pct"),
    ("pct_90_low",  "cost_5pct"),
    ("pct_90_high", "cost_95pct"),
    ("weight_avg",  "avg_cost"),
]

for a, b in pairs:
    diff = df[a] - df[b]
    print(f"=== {a}  vs  {b} ===")
    print(f"  mean(diff)     = {diff.mean():.6f}")
    print(f"  max(|diff|)    = {diff.abs().max():.6f}")
    print(f"  correlation    = {df[a].corr(df[b]):.6f}")
    print(f"  exact match    = {(df[a] == df[b]).mean()*100:.2f}%")
    print()

# Also check concentration formulas
print("=== Concentration formula check ===")
denom_70 = df["pct_70_high"] + df["pct_70_low"]
calc_70_con = (df["pct_70_high"] - df["pct_70_low"]) / denom_70.replace(0, np.nan)
print(f"  pct_70_con vs (high-low)/(high+low):")
print(f"    exact match = {(df['pct_70_con'] == calc_70_con).mean()*100:.2f}%")
print(f"    max(|diff|) = {(df['pct_70_con'] - calc_70_con).abs().max():.8f}")

denom_90 = df["pct_90_high"] + df["pct_90_low"]
calc_90_con = (df["pct_90_high"] - df["pct_90_low"]) / denom_90.replace(0, np.nan)
print(f"  pct_90_con vs (high-low)/(high+low):")
print(f"    exact match = {(df['pct_90_con'] == calc_90_con).mean()*100:.2f}%")
print(f"    max(|diff|) = {(df['pct_90_con'] - calc_90_con).abs().max():.8f}")
