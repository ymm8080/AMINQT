#!/usr/bin/env python3
"""Verify: is the concentration formula the same between akshare and Tushare?

akshare reported pct_90_con  vs  formula(hi-lo)/(hi+lo) from akshare's own low/high
Tushare formula(hi-lo)/(hi+lo) from Tushare's own cost_5pct/cost_95pct
"""

import os
import pandas as pd

with open(".env", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

import akshare as ak
import tushare as ts

# ── Fetch both sources for 600671 ──
ts.set_token(os.environ["TUSHARE_TOKEN"])
pro = ts.pro_api()
ts_raw = pro.cyq_perf(ts_code="600671.SH", start_date="20260601", end_date="20260729")
ts_df = pd.DataFrame(
    {
        "date": pd.to_datetime(ts_raw["trade_date"], format="%Y%m%d"),
        "ts_c5": pd.to_numeric(ts_raw["cost_5pct"]),
        "ts_c95": pd.to_numeric(ts_raw["cost_95pct"]),
    }
)
# Tushare: derive con90 from its own cost percentiles using the formula
ts_df["ts_con90_formula"] = (ts_df["ts_c95"] - ts_df["ts_c5"]) / (
    ts_df["ts_c95"] + ts_df["ts_c5"]
)

ak_df = ak.stock_cyq_em(symbol="600671", adjust="")
ak_df.columns = [
    "date",
    "bp",
    "avg",
    "p90lo",
    "p90hi",
    "p90con",
    "p70lo",
    "p70hi",
    "p70con",
]
ak_df["date"] = pd.to_datetime(ak_df["date"])
# akshare: derive con90 from its own low/high using the SAME formula
ak_df["ak_con90_formula"] = (ak_df["p90hi"] - ak_df["p90lo"]) / (
    ak_df["p90hi"] + ak_df["p90lo"]
)

m = ak_df.merge(ts_df, on="date", how="inner")
m = m.sort_values("date", ascending=False).head(15)

print("=" * 95)
print(
    "600671 — Is the concentration formula (hi-lo)/(hi+lo) the same for both sources?"
)
print("=" * 95)
print()
print(
    f"{'date':>12s}  {'ak_reported':>12s}  {'ak_formula':>12s}  {'ak_diff':>8s}"
    f"  |  {'ts_formula':>12s}  |  {'cross_src_diff':>14s}"
)
print("-" * 95)

for r in m.itertuples():
    ak_diff = abs(r.p90con - r.ak_con90_formula)
    cross_diff = abs(r.ak_con90_formula - r.ts_con90_formula)
    print(
        f"{r.date.strftime('%Y-%m-%d'):>12s}  {r.p90con:>12.6f}  {r.ak_con90_formula:>12.6f}  {ak_diff:>8.6f}"
        f"  |  {r.ts_con90_formula:>12.6f}  |  {cross_diff:>14.6f}"
    )

print()
print("Legend:")
print("  ak_reported    = akshare's pct_90_con (directly from API)")
print(
    "  ak_formula     = (pct_90_high - pct_90_low) / (pct_90_high + pct_90_low)  [from akshare data]"
)
print(
    "  ak_diff        = |ak_reported - ak_formula|  [should be ~0 if formula is correct]"
)
print(
    "  ts_formula     = (cost_95pct - cost_5pct) / (cost_95pct + cost_5pct)      [from Tushare data]"
)
print(
    "  cross_src_diff = |ak_formula - ts_formula|  [will be >0 because distributions differ]"
)
