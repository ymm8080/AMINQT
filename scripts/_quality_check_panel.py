# -*- coding: utf-8 -*-
"""Quality check for panel_alt against V3 production.
"""
import os
import sys

import pandas as pd

OUT_DIR = "data/new_symbols_raw"
PANEL_NEW = "panel_alt.parquet"
PANEL_V3 = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"

def main():
    print("[loading]")
    new = pd.read_parquet(os.path.join(OUT_DIR, PANEL_NEW))
    v3 = pd.read_parquet(PANEL_V3, columns=["symbol", "date"])
    new["trade_date"] = pd.to_datetime(new["trade_date"], errors="coerce")
    v3["date"] = pd.to_datetime(v3["date"], errors="coerce")

    print(f"new: {len(new)} rows, {new['symbol'].nunique()} symbols")
    print(f"v3:  {len(v3)} rows, {v3['symbol'].nunique()} symbols")

    # Symbol overlap
    s_new = set(new["symbol"].unique())
    s_v3 = set(v3["symbol"].unique())
    print(f"  new symbols in v3: {len(s_new & s_v3)} / {len(s_new)}")
    print(f"  new-only: {len(s_new - s_v3)} symbols")

    # Date overlap
    d_new = set(new["trade_date"].dropna().dt.date)
    d_v3 = set(v3["date"].dropna().dt.date)
    print(f"  new-trade-dates: {len(d_new)}")
    print(f"  v3-dates: {len(d_v3)}")

    # Coverage per symbol
    cov = new.groupby("symbol").size()
    print(f"  coverage: min={cov.min()} max={cov.max()} mean={cov.mean():.1f}")

    # OHLCV consistency
    ok_price = (
        (new["open"] <= new["high"]) &
        (new["open"] >= new["low"]) &
        (new["close"] <= new["high"]) &
        (new["close"] >= new["low"]) &
        (new["high"] >= new["low"])
    )
    print(f"  OHLCV consistency: {ok_price.sum()} / {len(new)} ({(ok_price.sum()/len(new)*100):.2f}%)")

    # Amount/vol non-negative
    ok_amt = (new["amount"] >= 0) & (new["vol"] >= 0)
    print(f"  non-negative amount/vol: {ok_amt.sum()} / {len(new)} ({(ok_amt.sum()/len(new)*100):.2f}%)")

    # output
    with open(os.path.join(OUT_DIR, "panel_alt_qc.txt"), "w") as f:
        f.write(f"new: {len(new)} rows, {new['symbol'].nunique()} symbols\n")
        f.write(f"v3:  {len(v3)} rows, {v3['symbol'].nunique()} symbols\n")
        f.write(f"new symbols in v3: {len(s_new & s_v3)} / {len(s_new)}\n")
        f.write(f"coverage: min={cov.min()} max={cov.max()} mean={cov.mean():.1f}\n")
        f.write(f"OHLCV consistency: {ok_price.sum()} / {len(new)} ({ok_price.sum()/len(new)*100:.2f}%)\n")
        f.write(f"non-negative amount/vol: {ok_amt.sum()} / {len(new)} ({ok_amt.sum()/len(new)*100:.2f}%)\n")
    print("[done] QC report written")
    return 0


if __name__ == "__main__":
    sys.exit(main())