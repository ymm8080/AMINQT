#!/usr/bin/env python3
"""Cross-verify CYQ column relationships using akshare + Tushare data.

akshare stock_cyq_em provides: date, winner_ratio, avg_cost,
  pct_90_low, pct_90_high, pct_90_con, pct_70_low, pct_70_high, pct_70_con

Tushare cyq_perf provides: cost_5pct, cost_15pct, cost_50pct,
  cost_85pct, cost_95pct, weight_avg, winner_rate

Verify:
  1. akshare concentration formula: con == (high-low)/(high+low)
  2. akshare pct_70_low vs Tushare cost_15pct
  3. akshare pct_70_high vs Tushare cost_85pct
  4. akshare pct_90_low vs Tushare cost_5pct
  5. akshare pct_90_high vs Tushare cost_95pct
"""

import os
import pandas as pd
import numpy as np

# Load .env for Tushare token
env_path = ".env"
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import akshare as ak
import tushare as ts

SYMBOL = "000001"
TS_CODE = f"{SYMBOL}.SZ"
START = "20260101"
END = "20260729"

# ── 1. Fetch akshare ──
print("=" * 60)
print("1. Fetching akshare stock_cyq_em...")
print("=" * 60)
ak_df = ak.stock_cyq_em(symbol=SYMBOL, adjust="")
# Fix column names (they are Chinese)
ak_cols = list(ak_df.columns)
print(f"akshare columns ({len(ak_cols)}): {ak_cols}")

# Rename to English
ak_df.columns = [
    "date",
    "winner_ratio",
    "avg_cost",
    "pct_90_low",
    "pct_90_high",
    "pct_90_con",
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
]
ak_df["date"] = pd.to_datetime(ak_df["date"])
ak_df = ak_df[ak_df["date"] >= "2026-01-01"].reset_index(drop=True)
print(f"akshare rows: {len(ak_df)}")
print(ak_df.tail(3).to_string())
print()

# ── 2. Fetch Tushare ──
print("=" * 60)
print("2. Fetching Tushare cyq_perf...")
print("=" * 60)
ts.set_token(os.environ["TUSHARE_TOKEN"])
pro = ts.pro_api()
ts_raw = pro.cyq_perf(ts_code=TS_CODE, start_date=START, end_date=END)
print(f"Tushare columns: {list(ts_raw.columns)}")
print(f"Tushare rows: {len(ts_raw)}")

ts_df = pd.DataFrame(
    {
        "date": pd.to_datetime(ts_raw["trade_date"], format="%Y%m%d"),
        "cost_5pct": pd.to_numeric(ts_raw["cost_5pct"]),
        "cost_15pct": pd.to_numeric(ts_raw["cost_15pct"]),
        "cost_50pct": pd.to_numeric(ts_raw["cost_50pct"]),
        "cost_85pct": pd.to_numeric(ts_raw["cost_85pct"]),
        "cost_95pct": pd.to_numeric(ts_raw["cost_95pct"]),
        "weight_avg": pd.to_numeric(ts_raw["weight_avg"]),
        "winner_rate": pd.to_numeric(ts_raw["winner_rate"]),
    }
)
print(ts_df.tail(3).to_string())
print()

# ── 3. Verify akshare concentration formula ──
print("=" * 60)
print("3. Verify akshare concentration formula: con == (high-low)/(high+low)")
print("=" * 60)

for label, lo, hi, con in [
    ("70%", "pct_70_low", "pct_70_high", "pct_70_con"),
    ("90%", "pct_90_low", "pct_90_high", "pct_90_con"),
]:
    calc = (ak_df[hi] - ak_df[lo]) / (ak_df[hi] + ak_df[lo]).replace(0, np.nan)
    diff = (ak_df[con] - calc).abs()
    print(f"\n  {label} concentration:")
    print(f"    rows:           {len(ak_df)}")
    print(f"    max(|diff|):    {diff.max():.8f}")
    print(f"    exact match:    {(diff < 1e-8).mean() * 100:.1f}%")
    print(f"    correlation:    {ak_df[con].corr(calc):.6f}")

# ── 4. Cross-verify akshare vs Tushare ──
print()
print("=" * 60)
print("4. Cross-verify akshare pct columns vs Tushare cost columns")
print("=" * 60)

merged = ak_df.merge(ts_df, on="date", how="inner")
print(f"  Merged rows: {len(merged)}")
print()

pairs = [
    ("akshare pct_70_low", "pct_70_low", "cost_15pct"),
    ("akshare pct_70_high", "pct_70_high", "cost_85pct"),
    ("akshare pct_90_low", "pct_90_low", "cost_5pct"),
    ("akshare pct_90_high", "pct_90_high", "cost_95pct"),
    ("akshare avg_cost", "avg_cost", "cost_50pct"),
    ("akshare avg_cost vs weight_avg", "avg_cost", "weight_avg"),
    ("akshare winner_ratio vs winner_rate/100", "winner_ratio", None),
]

for label, ak_col, ts_col in pairs:
    if ts_col is None:
        # special case: winner_ratio vs winner_rate/100
        ts_vals = merged["winner_rate"] / 100.0
    else:
        ts_vals = merged[ts_col]
    ak_vals = merged[ak_col]
    diff = (ak_vals - ts_vals).abs()
    print(f"  {label}  vs  Tushare {ts_col or 'winner_rate/100'}:")
    print(f"    rows:           {len(merged)}")
    print(f"    mean(diff):     {(ak_vals - ts_vals).mean():.6f}")
    print(f"    max(|diff|):    {diff.max():.6f}")
    print(f"    correlation:    {ak_vals.corr(ts_vals):.6f}")
    print(f"    exact match:    {(diff < 0.001).mean() * 100:.1f}%")
    print()

# Show sample side-by-side
print("=" * 60)
print("5. Sample comparison (last 5 rows)")
print("=" * 60)
cols = [
    "date",
    "pct_70_low",
    "cost_15pct",
    "pct_70_high",
    "cost_85pct",
    "pct_90_low",
    "cost_5pct",
    "pct_90_high",
    "cost_95pct",
    "winner_ratio",
    "winner_rate",
]
print(merged[cols].tail(5).to_string())
