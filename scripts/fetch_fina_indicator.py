# -*- coding: utf-8 -*-
"""Bulk fetch fina_indicator with per-thread API clients for thread-safety."""

import os
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
from app.pipeline1.data_supply import DataSupplyChain

# Get stock list (do this once in main thread)
supply = DataSupplyChain()
pro = supply._tushare_pro()

panel = pd.read_parquet("data/panel_full_enriched.parquet", columns=["symbol"])
panel_symbols = set(panel["symbol"].unique())
stocks = pro.stock_basic()
stocks["symbol"] = stocks["ts_code"].str.split(".").str[0]
ak = stocks[stocks["symbol"].isin(panel_symbols)].copy()
ak = ak.drop_duplicates("ts_code")
ts_list = ak["ts_code"].tolist()
total = len(ts_list)
print(f"Total: {total}", flush=True)

fields = (
    "ts_code,ann_date,end_date,"
    "roe,roa,gross_margin,netprofit_margin,"
    "dt_eps_yoy,or_yoy,netprofit_yoy,"
    "debt_to_assets,current_ratio,assets_turn,"
    "ocfps,revenue_ps,bps,eps,dt_eps,roe_yoy,q_roe,q_ocf_to_sales"
)

out_dir = "data/supply_cache/alt_data/fina_indicator"
os.makedirs(out_dir, exist_ok=True)
out_path = f"{out_dir}/all_20230701_20260727.parquet"

# Thread-safe state
import threading

results = []
errors = []
done = 0
_lock = threading.Lock()
_start = time.time()


def fetch_one(ts):
    """Create a per-thread API client to avoid thread-safety issues."""
    global done
    try:
        local_pro = DataSupplyChain()._tushare_pro()
        raw = local_pro.fina_indicator(
            ts_code=ts, start_date="20230701", end_date="20260727", fields=fields
        )
        if len(raw) > 0:
            raw["symbol"] = ts.split(".")[0]
        return raw
    finally:
        with _lock:
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - _start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                ok_r = sum(len(r) for r in results)
                print(
                    f"[{done}/{total}] {len(results)} OK, {ok_r} rows, "
                    f"{len(errors)} err | {elapsed:.0f}s / ETA {eta:.0f}s",
                    flush=True,
                )


with ThreadPoolExecutor(max_workers=4) as exe:
    futs = {exe.submit(fetch_one, ts): ts for ts in ts_list}
    for f in as_completed(futs):
        try:
            r = f.result()
            if r is not None and len(r) > 0:
                with _lock:
                    results.append(r)
        except Exception as e:
            with _lock:
                errors.append((futs[f], str(e)))

elapsed = time.time() - _start
print(
    f"\nDone in {elapsed:.0f}s | {len(results)}/{total} OK | {len(errors)} err",
    flush=True,
)

if errors:
    print(f"\nErrors (showing {min(20, len(errors))}):")
    for ts, e in errors[:20]:
        print(f"  {ts}: {e}", flush=True)

if results:
    df = pd.concat(results, ignore_index=True)
    df = df.drop_duplicates(["ts_code", "ann_date", "end_date"])
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"Rows: {len(df)}, Stocks: {df['ts_code'].nunique()}")
    print(f"Columns: {list(df.columns)}")
    print(df.head(3).to_string())
else:
    print("NO DATA!")
    sys.exit(1)
