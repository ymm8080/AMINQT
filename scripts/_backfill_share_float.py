"""Backfill the full 限售股解禁 (Tushare `share_float`) calendar into a raw cache.

The FINAL STOCK SCAN reads this cache to exclude candidates whose cumulative
future unlock ratio exceeds a threshold (解禁 → 抛压 → 不买). Unlike `block_trade`
(past executed trades), `share_float` is a FORWARD calendar: each row is one
(holder, share_type) unlock announced on `ann_date` for `float_date`, carrying
`float_ratio` = 解禁股份占总股本比例 (%).

Data completeness: Tushare `share_float` truncates every call at 6000 rows, and
mass-batch unlock dates (e.g. 2026-08-10 = 16,365 rows) exceed it — so each
float_date is fetched by paging `limit=6000, offset=0,6000,...` until empty.
Verified: offset pagination recovers all rows (16,365 recovered for 08-10).

Units (verified vs daily_basic): `float_share` in 股, `float_ratio` in %.

This is a ONE-TIME historical backfill — deliberately NOT in `_daily_fetch.py`.
The daily refresh (which captures newly-announced unlocks) lives in `_daily_fetch`
as a separate ann_date sweep block.

WORM: backs up an existing cache before overwriting; writes atomically.
"""

import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow.parquet as pq
import tushare as ts
from dotenv import load_dotenv

load_dotenv()

from config.settings import PANEL_V3_PATH

ROOT = Path(__file__).resolve().parent.parent
PANEL = Path(PANEL_V3_PATH)
CACHE = (
    ROOT
    / "data"
    / "supply_cache"
    / "alt_data"
    / "share_float"
    / "share_float_full.parquet"
)
KEEP = [
    "symbol",
    "ts_code",
    "ann_date",
    "float_date",
    "float_share",
    "float_ratio",
    "holder_name",
    "share_type",
]
START_D = "20230101"  # backtest window start (matches bt_ backfill)
FUTURE_DAYS = 120  # forward schedule to pre-load
SLEEP_S = 0.4
RETRY_N = 3


def _symbol_of(ts_code: str) -> str:
    for suf in (".SH", ".SZ", ".BJ"):
        if ts_code.endswith(suf):
            return ts_code[: -len(suf)]
    return ts_code


def _fetch_date(pro, dd: str) -> pd.DataFrame:
    """Page float_date=dd until empty (Tushare 6000-row cap). Returns df or empty."""
    chunks, offset, pages = [], 0, 0
    while True:
        for attempt in range(RETRY_N):
            try:
                pg = pro.share_float(float_date=dd, limit=6000, offset=offset)
                break
            except Exception:
                if attempt == RETRY_N - 1:
                    raise
                time.sleep(1.0 * (attempt + 1))
        if len(pg) == 0:
            break
        chunks.append(pg)
        offset += len(pg)
        pages += 1
        if pages >= 30:
            print(f"    share_float {dd}: page fuse hit at {pages} pages, stopping")
            break
        time.sleep(SLEEP_S)
    if not chunks:
        return pd.DataFrame(columns=KEEP)
    df = pd.concat(chunks, ignore_index=True)
    df["symbol"] = df["ts_code"].map(_symbol_of)
    return df[[c for c in KEEP if c in df.columns]]


def main():
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        print("FATAL: No Tushare token found.")
        return
    pro = ts.pro_api(token)

    if not PANEL.exists():
        print(f"FATAL: panel not found: {PANEL}")
        return

    # ── 1. Date list: past trading days from panel + future calendar days ──
    pdate = pq.read_table(PANEL, columns=["date"]).to_pandas()
    pdate["date"] = pd.to_datetime(pdate["date"])
    past = sorted(
        pdate.loc[pdate["date"] <= pd.Timestamp.today(), "date"]
        .dt.strftime("%Y%m%d")
        .unique()
    )
    future = [
        (pd.Timestamp.today() + pd.Timedelta(days=i)).strftime("%Y%m%d")
        for i in range(0, FUTURE_DAYS + 1)
    ]
    days = sorted(set(past) | set(future))
    days = [d for d in days if START_D <= d <= future[-1]]
    print(f"Dates to fetch: {len(days)}  ({days[0]}..{days[-1]})")

    # ── 2. Fetch + page each date ──
    frames, n_pages, t0 = [], 0, time.time()
    for i, dd in enumerate(days):
        try:
            df = _fetch_date(pro, dd)
        except Exception as e:
            print(f"  {dd}: FETCH ERROR {type(e).__name__}: {e}")
            continue
        n_pages += max(1, len(df) // 6000 + 1)
        if len(df):
            frames.append(df)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(
                f"  ...{i + 1}/{len(days)}  rows_so_far={sum(len(f) for f in frames):,}  "
                f"elapsed={el / 60:.1f}min"
            )
    if not frames:
        print("FATAL: no data fetched.")
        return

    full = pd.concat(frames, ignore_index=True)
    full = (
        full.drop_duplicates(keep="last")
        .sort_values(["symbol", "float_date"])
        .reset_index(drop=True)
    )
    full = full.dropna(subset=["float_date"]).reset_index(drop=True)
    full = full[KEEP]
    print(f"\nFetched: {len(full):,} rows, {full['symbol'].nunique():,} symbols")
    print(
        f"  float_date {full['float_date'].min()}..{full['float_date'].max()} | "
        f"ann_date {full['ann_date'].min()}..{full['ann_date'].max()}"
    )

    # ── 3. WORM backup + atomic write ──
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        bak = CACHE.with_name(
            f"share_float_full_prebackfill_{time.strftime('%Y%m%d_%H%M%S')}.parquet"
        )
        shutil.copy2(CACHE, bak)
        print(f"Cache WORM backup: {bak}")
    tmp = CACHE.with_name(CACHE.name + ".tmp")
    full.to_parquet(tmp, index=False, compression="snappy")
    os.replace(tmp, CACHE)
    print(f"Cache written: {CACHE} ({CACHE.stat().st_size / 1e6:.1f} MB)")

    # ── 4. Verify ──
    v = pd.read_parquet(CACHE)
    print(f"\nVerify: {len(v):,} rows, {len(v.columns)} cols")
    for c in KEEP:
        print(f"  {c:14s} non-null={v[c].notna().sum():,}")
    dup = v.duplicated().sum()
    print(f"  duplicate rows: {dup}")
    # spot check a mass-batch date is complete (>6000)
    for chk in ("20260810", "20260715"):
        n = (v["float_date"] == chk).sum()
        print(f"  float_date {chk}: {n:,} rows (expect >6000 for mass batches)")


if __name__ == "__main__":
    main()
