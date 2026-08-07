"""Fetch fina_indicator for 2022 Q3 + 2022 annual reports (announced Oct 2022 ~ Apr 2023).

These announcements are needed to forward-fill H1 2023 (Jan-Mar) trading days
in the v3 panel. The existing cache starts from Apr 2023, leaving Jan-Mar 2023
rows with NaN roe.

Strategy: per-stock fetch, start_date=20220101 end_date=20230701, all 3244 stocks.
Save to all_20220101_20230701.parquet (WORM: no overwrite of old file).
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

import pandas as pd  # noqa: E402

# Load .env manually (avoid dotenv dependency)
_env_path = os.path.join(_project_root, ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

# ── Config ──
START_DATE = "20220101"
END_DATE = "20230701"
OUT_DIR = "data/supply_cache/alt_data/fina_indicator"
OUT_PATH = f"{OUT_DIR}/all_20220101_20230701.parquet"

FIELDS = (
    "ts_code,ann_date,end_date,"
    "roe,roa,gross_margin,netprofit_margin,"
    "dt_eps_yoy,or_yoy,netprofit_yoy,"
    "debt_to_assets,current_ratio,assets_turn,"
    "ocfps,revenue_ps,bps,eps,dt_eps,roe_yoy,q_roe,q_ocf_to_sales"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Get v3 stock list ──
panel_path = os.path.join(_project_root, "data", "panel_full_enriched_v3.parquet")
panel = pd.read_parquet(panel_path, columns=["symbol"])
panel_symbols = set(panel["symbol"].unique())
supply = DataSupplyChain()
pro = supply._tushare_pro()
stocks = pro.stock_basic()
stocks["symbol"] = stocks["ts_code"].str.split(".").str[0]
ak = stocks[stocks["symbol"].isin(panel_symbols)].copy()
ak = ak.drop_duplicates("ts_code")
ts_list = ak["ts_code"].tolist()
total = len(ts_list)
print(f"Total stocks to fetch: {total}", flush=True)
print(f"Date range: {START_DATE} ~ {END_DATE}", flush=True)

# ── Thread-safe state ──
results = []
errors = []
done = 0
_lock = threading.Lock()
_start = time.time()


def fetch_one(ts):
    global done
    try:
        local_pro = DataSupplyChain()._tushare_pro()
        raw = local_pro.fina_indicator(
            ts_code=ts, start_date=START_DATE, end_date=END_DATE, fields=FIELDS
        )
        if len(raw) > 0:
            raw["symbol"] = ts.split(".")[0]
        return raw
    except Exception as e:
        return ("ERROR", ts, str(e))
    finally:
        with _lock:
            done += 1
            if done % 200 == 0:
                elapsed = time.time() - _start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                ok_r = sum(len(r) for r in results if isinstance(r, pd.DataFrame))
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
            if isinstance(r, pd.DataFrame) and len(r) > 0:
                with _lock:
                    results.append(r)
            elif isinstance(r, tuple) and r[0] == "ERROR":
                with _lock:
                    errors.append((r[1], r[2]))
        except Exception as e:
            with _lock:
                errors.append((futs[f], str(e)))

elapsed = time.time() - _start
print(
    f"\nDone in {elapsed:.0f}s | {len(results)}/{total} OK | {len(errors)} err",
    flush=True,
)

if errors:
    print(f"\nErrors ({min(20, len(errors))} shown):")
    for ts, e in errors[:20]:
        print(f"  {ts}: {e}", flush=True)

if results:
    df = pd.concat(results, ignore_index=True)
    df = df.drop_duplicates(["ts_code", "ann_date", "end_date"])
    df = df.sort_values(["ts_code", "end_date"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}", flush=True)
    print(f"Rows: {len(df)}, Stocks: {df['ts_code'].nunique()}", flush=True)
    # Check date range
    df["ann_dt"] = pd.to_datetime(df["ann_date"], format="%Y%m%d", errors="coerce")
    print(
        f"Announce date range: {df['ann_dt'].min()} ~ {df['ann_dt'].max()}", flush=True
    )
    # Check pre-2023-04-08 coverage (the gap we're filling)
    pre_apr = df[df["ann_dt"] < "2023-04-08"]
    print(
        f"Pre-2023-04-08 rows: {len(pre_apr)}, stocks: {pre_apr['ts_code'].nunique()}",
        flush=True,
    )
else:
    print("NO DATA!", flush=True)
    sys.exit(1)
