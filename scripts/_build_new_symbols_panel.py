# -*- coding: utf-8 -*-
"""Build new symbols panel from daily/adj_factor/daily_basic/suspend/stk_limit.

1) Read all daily/adj_factor/daily_basic/suspend/stk_limit files in data/new_symbols_raw.
2) For each symbol:
   - Merge daily OHLCV with adj_factor → forward-adjust price (open/high/low/close/pre_close)
   - Compute vol/amount/adj_ratio, add hfq columns
   - Merge daily_basic (valuation/market cap)
   - Merge stk_limit (up_limit_raw/down_limit_raw)
   - Merge suspend (flag)
   - Derive features: gap_up_5pct_cnt, vol_chg_20d, etc
3) Output: data/new_symbols_raw/panel_new_symbols.parquet (symbol/date OHLCV-hfq/adj_ratio/valuation/limits/suspend/derived)
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# config
OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
PANEL_OUT = "panel_new_symbols.parquet"


def load_parquets(pattern: str) -> pd.DataFrame:
    """Load all parquet files matching pattern, concat."""
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    dfs = []
    for f in fs:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"WARN: {f} bad parquet: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    print("[daily]")
    daily = load_parquets(os.path.join(OUT_DIR, "daily", "*.parquet"))
    print(f"  {len(daily)} rows")
    print("[adj_factor]")
    adj = load_parquets(os.path.join(ALT_DIR, "adj_factor", "adj_*.parquet"))
    print(f"  {len(adj)} rows")
    print("[daily_basic]")
    basic = load_parquets(os.path.join(OUT_DIR, "daily_basic", "*.parquet"))
    print(f"  {len(basic)} rows")
    print("[stk_limit]")
    limit = load_parquets(os.path.join(ALT_DIR, "stk_limit", "*_all__.parquet"))
    print(f"  {len(limit)} rows")
    print("[suspend]")
    susp = load_parquets(os.path.join(OUT_DIR, "suspend", "*.parquet"))
    print(f"  {len(susp)} rows")

    if daily.empty:
        print("ERROR: no daily data")
        return 1
    if adj.empty:
        print("ERROR: no adj_factor")
        return 1

    # ensure dtypes
    for df in [daily, adj, basic, limit, susp]:
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # adj_factor: get latest adj_factor per (symbol, trade_date)
    # There may be multiple rows per symbol per day. Use last.
    adj_latest = adj.groupby(["symbol", "trade_date"], as_index=False)["adj_factor"].last()

    # merge daily + adj_factor
    panel = pd.merge(daily, adj_latest, on=["symbol", "trade_date"], how="left")
    # fill missing adj_factor = 1.0 (no adjustment)
    panel["adj_factor"] = panel["adj_factor"].fillna(1.0)

    # forward-adjust price: open/high/low/close/pre_close
    # OHLCV_hfq = raw * adj_factor
    for c in ["open", "high", "low", "close", "pre_close"]:
        if c in panel.columns:
            panel[f"{c}_hfq"] = panel[c] * panel["adj_factor"]
    # volume/amount not adjusted
    # add adj_ratio = adj_factor / adj_factor.shift(1) for returns
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel["adj_ratio"] = panel.groupby("symbol")["adj_factor"].transform(lambda x: x / x.shift(1).fillna(1.0))

    # merge basic
    if not basic.empty:
        basic_ = basic[[x for x in basic.columns if x not in ("date", "ts_code")]]
        if "trade_date" in basic_.columns:
            panel = pd.merge(panel, basic_, on=["symbol", "trade_date"], how="left")
    # merge limit
    if not limit.empty:
        # limit has down_limit_raw, up_limit_raw
        limit_ = limit[["symbol", "trade_date", "up_limit_raw", "down_limit_raw"]].copy()
        limit_["up_limit_raw"] = pd.to_numeric(limit_["up_limit_raw"], errors="coerce")
        limit_["down_limit_raw"] = pd.to_numeric(limit_["down_limit_raw"], errors="coerce")
        panel = pd.merge(panel, limit_, on=["symbol", "trade_date"], how="left")
    # merge suspend
    if not susp.empty:
        susp_ = susp[["symbol", "trade_date"]].copy()
        susp_["is_suspend"] = 1
        panel = pd.merge(panel, susp_, on=["symbol", "trade_date"], how="left")
        panel["is_suspend"] = panel["is_suspend"].fillna(0)

    # derived features
    # gap_up_5pct_cnt: count consecutive days open >= pre_close * (1 + 0.05)
    # This is a rolling count per symbol
    panel = panel.sort_values(["symbol", "trade_date"])
    panel["gap_up_5pct"] = (panel["open"] >= panel["pre_close"] * 1.05).astype(int)
    panel["gap_up_5pct_cnt"] = panel.groupby("symbol")["gap_up_5pct"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )

    # vol_chg_20d: volume vs 20d rolling mean
    panel["vol_20d"] = panel.groupby("symbol")["vol"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    panel["vol_chg_20d"] = np.where(panel["vol_20d"] > 0, panel["vol"] / panel["vol_20d"] - 1, 0)

    # add columns for safety
    for c in ["close", "open"]:
        panel[f"{c}_hfq"] = panel[f"{c}_hfq"].fillna(panel[c])

    # reorder
    keep = ["symbol", "trade_date",
            "open", "high", "low", "close", "pre_close",
            "open_hfq", "high_hfq", "low_hfq", "close_hfq",
            "vol", "amount", "adj_factor", "adj_ratio",
            "up_limit_raw", "down_limit_raw", "is_suspend",
            "gap_up_5pct", "gap_up_5pct_cnt", "vol_20d", "vol_chg_20d"]
    # add all from merged basic
    for c in panel.columns:
        if c not in keep:
            keep.append(c)
    panel = panel[keep]

    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel.to_parquet(os.path.join(OUT_DIR, PANEL_OUT), index=False)
    print(f"[done] {len(panel)} rows -> {OUT_DIR}/{PANEL_OUT}")
    print(f"  symbols={panel['symbol'].nunique()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())