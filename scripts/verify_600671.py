#!/usr/bin/env python3
"""Compare Tushare-derived vs akshare-reported CYQ for 600671 on 2 dates."""

import os
import pandas as pd

# Load Tushare token
with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import tushare as ts
import akshare as ak

TS_CODE = "600671.SH"
SYMBOL = "600671"
TARGET_DATES = ["2026-06-11", "2026-07-28"]

# ── 1. Tushare ──
ts.set_token(os.environ["TUSHARE_TOKEN"])
pro = ts.pro_api()
ts_raw = pro.cyq_perf(ts_code=TS_CODE, start_date="20260601", end_date="20260729")
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

# Derive pct_90_con and pct_70_con from cost columns
ts_df["pct_90_low"] = ts_df["cost_5pct"]
ts_df["pct_90_high"] = ts_df["cost_95pct"]
ts_df["pct_90_con_derived"] = (ts_df["cost_95pct"] - ts_df["cost_5pct"]) / (
    ts_df["cost_95pct"] + ts_df["cost_5pct"]
)
ts_df["pct_70_low"] = ts_df["cost_15pct"]
ts_df["pct_70_high"] = ts_df["cost_85pct"]
ts_df["pct_70_con_derived"] = (ts_df["cost_85pct"] - ts_df["cost_15pct"]) / (
    ts_df["cost_85pct"] + ts_df["cost_15pct"]
)

# ── 2. akshare ──
ak_df = ak.stock_cyq_em(symbol=SYMBOL, adjust="")
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

# ── 3. Compare for target dates ──
print("=" * 80)
print("600671 CYQ Comparison: Tushare-derived vs akshare-reported")
print("=" * 80)

for d in TARGET_DATES:
    dt = pd.to_datetime(d)
    ts_row = ts_df[ts_df["date"] == dt]
    ak_row = ak_df[ak_df["date"] == dt]

    if ts_row.empty:
        print(f"\n{d}: Tushare NO DATA")
        continue
    if ak_row.empty:
        print(f"\n{d}: akshare NO DATA")
        continue

    tr = ts_row.iloc[0]
    ar = ak_row.iloc[0]

    print(f"\n{'─' * 80}")
    print(f"  Date: {d}")
    print(f"{'─' * 80}")

    print(f"\n  {'Metric':<30s}  {'Tushare':>12s}  {'akshare':>12s}  {'diff':>10s}")
    print(f"  {'─' * 68}")

    # 90% concentration
    ts_con90 = tr["pct_90_con_derived"]
    ak_con90 = ar["pct_90_con"]
    print(
        f"  {'90% concentration':<30s}  {ts_con90:>12.6f}  {ak_con90:>12.6f}  {abs(ts_con90 - ak_con90):>10.6f}"
    )

    # 90% low/high
    print(
        f"  {'90% low (price)':<30s}  {tr['cost_5pct']:>12.2f}  {ar['pct_90_low']:>12.2f}  {abs(tr['cost_5pct'] - ar['pct_90_low']):>10.2f}"
    )
    print(
        f"  {'90% high (price)':<30s}  {tr['cost_95pct']:>12.2f}  {ar['pct_90_high']:>12.2f}  {abs(tr['cost_95pct'] - ar['pct_90_high']):>10.2f}"
    )

    # 70% concentration
    ts_con70 = tr["pct_70_con_derived"]
    ak_con70 = ar["pct_70_con"]
    print(
        f"  {'70% concentration':<30s}  {ts_con70:>12.6f}  {ak_con70:>12.6f}  {abs(ts_con70 - ak_con70):>10.6f}"
    )

    # 70% low/high
    print(
        f"  {'70% low (price)':<30s}  {tr['cost_15pct']:>12.2f}  {ar['pct_70_low']:>12.2f}  {abs(tr['cost_15pct'] - ar['pct_70_low']):>10.2f}"
    )
    print(
        f"  {'70% high (price)':<30s}  {tr['cost_85pct']:>12.2f}  {ar['pct_70_high']:>12.2f}  {abs(tr['cost_85pct'] - ar['pct_70_high']):>10.2f}"
    )

    # winner_ratio
    ts_bp = tr["winner_rate"] / 100
    ak_bp = ar["winner_ratio"]
    print(
        f"  {'winner_ratio (winner/100)':<30s}  {ts_bp:>12.6f}  {ak_bp:>12.6f}  {abs(ts_bp - ak_bp):>10.6f}"
    )

    # weight_avg / avg_cost
    print(
        f"  {'weight_avg vs avg_cost':<30s}  {tr['weight_avg']:>12.2f}  {ar['avg_cost']:>12.2f}  {abs(tr['weight_avg'] - ar['avg_cost']):>10.2f}"
    )
