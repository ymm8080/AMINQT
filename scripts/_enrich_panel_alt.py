"""Enrich panel_cyq with margin/lhb/bt/top_inst."""

import glob
import os
import sys

import pandas as pd

OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
PANEL_IN = "panel_cyq.parquet"
PANEL_OUT = "panel_alt.parquet"


def load_alt(pattern, col_date="date"):
    """Load alt batch files, return DataFrame with trade_date"""
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    dfs = []
    for f in fs:
        try:
            df = pd.read_parquet(f)
            if col_date in df.columns:
                df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
            if "trade_date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
            dfs.append(df)
        except Exception as e:
            print(f"WARN: {f} bad: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    panel = pd.read_parquet(os.path.join(OUT_DIR, PANEL_IN))
    keys = panel[["symbol", "trade_date"]].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce")
    print(f"panel: {len(panel)} rows, {keys['symbol'].nunique()} symbols")

    # margin
    margin = load_alt(os.path.join(ALT_DIR, "margin_b*.parquet"))
    if not margin.empty:
        margin = margin[
            [
                "symbol",
                "date",
                "margin_balance",
                "short_balance",
                "margin_buy_amt",
                "short_sell_vol",
            ]
        ].copy()
        margin.rename(columns={"date": "trade_date"}, inplace=True)
    # lhb (aggregated from top_list)
    lhb = load_alt(os.path.join(ALT_DIR, "lhb", "lhb_b*.parquet"))
    if not lhb.empty:
        lhb = lhb[
            ["symbol", "date", "lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"]
        ].copy()
        lhb.rename(columns={"date": "trade_date"}, inplace=True)
    # bt
    bt = load_alt(os.path.join(ALT_DIR, "block_trade", "bt_b*.parquet"))
    if not bt.empty:
        bt = bt[
            ["symbol", "trade_date", "price", "vol", "amount", "buyer", "seller"]
        ].copy()
    # top_inst
    topinst_files = sorted(
        glob.glob(os.path.join(OUT_DIR, "top_inst", "top_inst_*.parquet"))
    )
    if topinst_files:
        topinst = load_alt(
            os.path.join(OUT_DIR, "top_inst", "top_inst_*.parquet"), "trade_date"
        )
    else:
        topinst = pd.DataFrame()
    if not topinst.empty:
        topinst = topinst[
            ["symbol", "ts_code", "trade_date", "exalter", "buy", "sell"]
        ].copy()

    # Merge in order
    p = panel.copy()
    for _i, (df, name) in enumerate(
        [(margin, "margin"), (lhb, "lhb"), (bt, "bt"), (topinst, "topinst")]
    ):
        if df.empty:
            print(f"[{name}] skip (empty)")
            continue
        df = df.drop_duplicates(subset=["symbol", "trade_date"])
        len(p)
        p = pd.merge(
            p, df, on=["symbol", "trade_date"], how="left", suffixes=("", f"_{name}")
        )
        print(f"[{name}] {len(df)} rows, panel now {len(p)} rows")

    # reorder columns
    keep = [x for x in p.columns if x not in ("date", "ts_code")]
    p = p[keep]
    p.to_parquet(os.path.join(OUT_DIR, PANEL_OUT), index=False)
    print(f"[done] {OUT_DIR}/{PANEL_OUT} ({len(p)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
