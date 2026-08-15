# -*- coding: utf-8 -*-
"""Enrich panel_new_symbols with CYQ筹码分布 18 列.

Read alt_data/cyq_tushare/cyq_b*.parquet, merge into panel
by (symbol, trade_date). Output panel_cyq.parquet.
"""
import glob
import os
import sys

import pandas as pd

OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
PANEL_IN = "panel_new_symbols.parquet"
PANEL_OUT = "panel_cyq.parquet"

def main():
    # read panel keys
    panel = pd.read_parquet(os.path.join(OUT_DIR, PANEL_IN))
    keys = panel[["symbol", "trade_date"]].drop_duplicates()
    keys["trade_date"] = pd.to_datetime(keys["trade_date"], errors="coerce")
    print(f"panel keys: {len(keys)} symbol-date pairs")

    # read cyq batches
    cyq_files = sorted(glob.glob(os.path.join(ALT_DIR, "cyq_tushare", "cyq_b*.parquet")))
    if not cyq_files:
        print("ERROR: no cyq_b*.parquet")
        return 1
    print(f"cyq files: {len(cyq_files)}")

    chunks = []
    for f in cyq_files:
        df = pd.read_parquet(f)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        chunks.append(df)
        print(f"  {f}: {len(df)} rows")
    cyq = pd.concat(chunks, ignore_index=True)
    print(f"cyq full: {len(cyq)} rows")

    # merge keys with cyq
    enriched = pd.merge(keys, cyq, on=["symbol", "trade_date"], how="left")
    print(f"enriched: {len(enriched)} rows")
    print(f"  unique symbols: {enriched['symbol'].nunique()}")

    # merge back into panel
    panel_out = pd.merge(panel, enriched, on=["symbol", "trade_date"], how="left")
    print(f"panel out: {len(panel_out)} rows")

    # add column order
    keep = [x for x in panel_out.columns if x not in ("date", "ts_code")]
    panel_out = panel_out[keep]
    panel_out.to_parquet(os.path.join(OUT_DIR, PANEL_OUT), index=False)
    print(f"[done] {OUT_DIR}/{PANEL_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())