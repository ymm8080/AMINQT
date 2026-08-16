"""Pull specific sources from panel calendar to alt_data. Idempotent, perdate resume.

Example:
  python _pull_select_sources.py daily daily_basic adj_factor stk_limit suspend
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts  # noqa: E402

from config import settings  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
CALL_SLEEP = 0.2
FLUSH_EVERY = 50


def _panel_calendar() -> pd.DatetimeIndex:
    p = pd.read_parquet(PANEL, columns=["date"])
    return pd.DatetimeIndex(sorted(p["date"].unique()))


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


class Progress:
    def __init__(self, src):
        self.path = os.path.join(OUT_DIR, "progress.json")
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)
        self.key = f"ps_{src}"  # prefix

    def get(self):
        return self.data.get(self.key, "")

    def set(self, val):
        self.data[self.key] = val
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)


def _fetch(pro, fn, name, attempts=4, **kw) -> pd.DataFrame:
    for i in range(1, attempts + 1):
        try:
            df = fn(**kw)
            if df is not None and not df.empty:
                return _to_symbol(df)
        except Exception as e:
            print(f"    {name}: FAIL attempt {i}: {e}", flush=True)
        time.sleep(3 * i)
    print(f"    {name}: GAVE UP after {attempts} attempts", flush=True)
    return pd.DataFrame()


def _save_alt(df: pd.DataFrame, fname: str, subdir: str):
    d = os.path.join(ALT_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, fname), index=False)


def pull_daily(pro, ds):
    d = _fetch(pro, pro.daily, f"daily {ds}", trade_date=ds)
    if not d.empty:
        keep = [
            x
            for x in [
                "ts_code",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "amount",
                "vol",
                "symbol",
            ]
            if x in d.columns
        ]
        _save_alt(d[keep], f"daily_{ds}.parquet", "daily")
        return True
    return False


def pull_basic(pro, ds):
    d = _fetch(pro, pro.daily_basic, f"basic {ds}", trade_date=ds)
    if not d.empty:
        keep = [
            x
            for x in [
                "ts_code",
                "trade_date",
                "turnover_rate",
                "turnover_rate_f",
                "volume_ratio",
                "pe_ttm",
                "pb",
                "ps_ttm",
                "total_share",
                "float_share",
                "free_share",
                "total_mv",
                "circ_mv",
                "dv_ratio",
                "dv_ttm",
                "symbol",
            ]
            if x in d.columns
        ]
        _save_alt(d[keep], f"daily_basic_{ds}.parquet", "daily_basic")
        return True
    return False


def pull_adj(pro, ds):
    a = _fetch(pro, pro.adj_factor, f"adj {ds}", trade_date=ds)
    if not a.empty:
        keep = ["ts_code", "trade_date", "adj_factor", "symbol"]
        _save_alt(a[keep], f"adj_{ds}.parquet", "adj_factor")
        return True
    return False


def pull_limit(pro, ds):
    limit_data = _fetch(pro, pro.stk_limit, f"limit {ds}", trade_date=ds)
    if not limit_data.empty:
        l2 = limit_data[["symbol", "trade_date", "up_limit", "down_limit"]].rename(
            columns={"up_limit": "up_limit_raw", "down_limit": "down_limit_raw"}
        )
        l2["date"] = pd.Timestamp(ds)
        _save_alt(l2, f"{ds}_all__.parquet", "stk_limit")
        return True
    return False


def pull_suspend(pro, ds):
    s = _fetch(pro, pro.suspend_d, f"susp {ds}", trade_date=ds)
    if not s.empty:
        keep = ["symbol", "trade_date"]
        _save_alt(s[keep], f"suspend_{ds}.parquet", "suspend")
        return True
    return False


def main():
    src_map = {
        "daily": pull_daily,
        "daily_basic": pull_basic,
        "adj_factor": pull_adj,
        "stk_limit": pull_limit,
        "suspend": pull_suspend,
    }
    p = argparse.ArgumentParser()
    p.add_argument("srcs", nargs="+", choices=list(src_map.keys()))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cal = _panel_calendar()
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    print(f"[plan] sources={args.srcs} | dates={len(cal)} (panel)", flush=True)
    if args.dry_run:
        print("[dry-run] bye")
        return

    for src in args.srcs:
        prog = Progress(src)
        last_done = prog.get()
        todo = [d for d in cal if d.strftime("%Y%m%d") > last_done]
        print(
            f"[{src}] todo={len(todo)} (resume after {last_done or 'none'})", flush=True
        )
        t0 = time.time()
        for i, d in enumerate(todo):
            ds = d.strftime("%Y%m%d")
            ok = src_map[src](pro, ds)
            if ok:
                prog.set(ds)
            time.sleep(CALL_SLEEP)
            if (i + 1) % FLUSH_EVERY == 0:
                rate = (i + 1) / (time.time() - t0) * 3600
                print(f"[{src}] {i + 1}/{len(todo)} ({rate:.0f}/hr) @ {ds}", flush=True)
        print(f"[{src}] DONE", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
