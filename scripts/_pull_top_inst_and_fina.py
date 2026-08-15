"""Pull top_inst (by LHB dates) and fina (per new symbol)."""

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts

from config import settings

OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
CALL_SLEEP = 0.3
FLUSH_EVERY = 50


def _to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "ts_code" in df.columns:
        df["symbol"] = (
            df["ts_code"]
            .str.replace(".SZ", "", regex=False)
            .str.replace(".SH", "", regex=False)
            .str.replace(".BJ", "", regex=False)
        )
    return df


def _fetch(pro, fn, name, attempts=4, **kw) -> pd.DataFrame:
    for i in range(1, attempts + 1):
        try:
            df = fn(**kw)
            if df is not None and not df.empty:
                return _to_symbol(df)
        except Exception as e:
            print(f"    {name}: FAIL {i}: {e}", flush=True)
        time.sleep(3 * i)
    return pd.DataFrame()


def _save(df, subdir, fname):
    d = os.path.join(OUT_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, fname), index=False)


def pull_top_inst():
    """Extract LHB dates from lhb/*.parquet and pull top_inst for each date."""
    lhb_files = glob.glob(os.path.join(ALT_DIR, "lhb", "lhb_*.parquet"))
    if not lhb_files:
        print("[top_inst] no lhb files")
        return
    dates = set()
    for f in lhb_files:
        try:
            d = pd.read_parquet(f, columns=["date"])
            dates |= set(pd.to_datetime(d["date"]).dt.strftime("%Y%m%d").unique())
        except Exception as e:
            print(f"[top_inst] {f} bad: {e}")
    dates = sorted(dates)
    print(f"[top_inst] LHB dates={len(dates)}", flush=True)

    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    acc = []
    t0 = time.time()
    for i, ds in enumerate(dates):
        t = _fetch(pro, pro.top_inst, f"top_inst {ds}", trade_date=ds)
        if not t.empty:
            keep = [
                x
                for x in ["symbol", "ts_code", "trade_date", "exalter", "buy", "sell"]
                if x in t.columns
            ]
            t2 = t[keep].copy()
            t2["date"] = pd.Timestamp(ds)
            acc.append(t2)
        time.sleep(CALL_SLEEP)
        if (i + 1) % FLUSH_EVERY == 0:
            if acc:
                ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
                _save(
                    pd.concat(acc, ignore_index=True),
                    "top_inst",
                    f"top_inst_{ts_}.parquet",
                )
                acc = []
            rate = (i + 1) / (time.time() - t0) * 3600
            print(f"[top_inst] {i + 1}/{len(dates)} ({rate:.0f}/hr)", flush=True)
    if acc:
        ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save(pd.concat(acc, ignore_index=True), "top_inst", f"top_inst_{ts_}.parquet")
    print("[top_inst] DONE")


def pull_fina():
    """Pull fina_indicator for each new symbol (full history)."""
    universe = pd.read_parquet(
        sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
    )
    ts_map = dict(zip(universe["symbol"], universe["ts_code"]))
    todo_fina = sorted(universe["symbol"].unique())
    print(f"[fina] todo={len(todo_fina)} symbols", flush=True)

    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    acc = []
    t0 = time.time()
    for i, sym in enumerate(todo_fina):
        code = ts_map.get(sym, sym + (".SH" if sym.startswith(("6", "5")) else ".SZ"))
        f = _fetch(
            pro,
            pro.fina_indicator,
            f"fina {sym}",
            ts_code=code,
            start_date="20220101",
            end_date="20260815",
            fields="ts_code,ann_date,end_date,roe,roe_dt,roa,np_margin,gross_margin,eps,ocfps,bps,revenue_ps,eps_yoy,or_yoy,profit_yoy,debt_to_assets,current_ratio,assets_turn,ar_turn,inv_turn,ocf_to_or,dt_eps,roe_yoy,q_roe,q_ocf_to_sales",
        )
        if not f.empty:
            acc.append(f)
        time.sleep(CALL_SLEEP)
        if (i + 1) % FLUSH_EVERY == 0:
            if acc:
                ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
                _save(pd.concat(acc, ignore_index=True), "fina", f"fina_{ts_}.parquet")
                acc = []
            rate = (i + 1) / (time.time() - t0) * 3600
            print(f"[fina] {i + 1}/{len(todo_fina)} ({rate:.0f}/hr)", flush=True)
    if acc:
        ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
        _save(pd.concat(acc, ignore_index=True), "fina", f"fina_{ts_}.parquet")
    print("[fina] DONE")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", choices=["top_inst", "fina", "both"])
    args = p.parse_args()
    if args.src in ("top_inst", "both"):
        pull_top_inst()
    if args.src in ("fina", "both"):
        pull_fina()


if __name__ == "__main__":
    main()
