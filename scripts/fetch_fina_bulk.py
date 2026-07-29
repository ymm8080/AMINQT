#!/usr/bin/env python3
"""
Bulk fetch fina_indicator for ALL stocks in panel_3y.parquet using per-stock
Tushare queries.  Handles rate limiting, caching, and error recovery.

Usage:
    python scripts/fetch_fina_bulk.py

Output:
    data/supply_cache/alt_data/fina_indicator/all_20240102_20260727.parquet
"""
from __future__ import annotations

import logging
import os
import sys
import time

import pandas as pd

# ── path bootstrap ───────────────────────────────────────────────
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJ not in sys.path:
    sys.path.insert(0, PROJ)

from app.pipeline1.data_supply import DataSupplyChain

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_fina_bulk")

START_DATE = "20240102"
END_DATE = "20260727"
OUTPUT = os.path.join(
    PROJ,
    "data/supply_cache/alt_data/fina_indicator",
    f"all_{START_DATE}_{END_DATE}.parquet",
)
PANEL_PATH = os.path.join(PROJ, "data/panel_3y.parquet")
RATE_LIMIT_S = 0.05  # 50 ms between calls → ~20 calls/sec
PROGRESS_INTERVAL = 50


def ts_code_of(symbol: str) -> str:
    """Map 6-digit symbol → Tushare ts_code."""
    suffix = "SZ" if symbol.startswith(("0", "3", "1")) else "SH"
    return f"{symbol}.{suffix}"


def main() -> None:
    # 1. Load panel → get stock list
    log.info("Loading panel from %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    symbols = sorted(panel["symbol"].unique())
    log.info("Panel has %d unique symbols, date range %s ~ %s",
             len(symbols), panel["date"].min(), panel["date"].max())

    supply = DataSupplyChain(
        cache_dir=os.path.join(PROJ, "data/supply_cache")
    )

    # 2. Test a bare-period query (no ts_code) — faster if it works
    log.info("Testing bare-period query (no ts_code)...")
    try:
        df_test = supply.fetch_fina_indicator(
            period=None, start_date=START_DATE, end_date=END_DATE,
        )
        if df_test is not None and len(df_test) > 0:
            log.info("Bare-period query returned %d rows — saving directly!", len(df_test))
            os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
            df_test.to_parquet(OUTPUT, index=False)
            _report(df_test)
            return
    except Exception as exc:
        log.info("Bare-period query failed (%s), falling back to per-stock.", exc)

    # Alternate: try period='20241231' (single-period) without ts_code
    try:
        df_test2 = supply.fetch_fina_indicator(
            period="20241231", start_date=None, end_date=None,
        )
        if df_test2 is not None and len(df_test2) > 0:
            log.info("Single-period query returned %d rows!", len(df_test2))
    except Exception:
        log.info("Single-period query also failed.")

    # 3. Per-stock fetch
    log.info("Starting per-stock fetch for %d symbols ...", len(symbols))
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    skipped = 0

    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        tsc = ts_code_of(sym)

        # Check if per-stock cache already exists (fetch_fina_indicator handles this internally)
        try:
            df_one = supply.fetch_fina_indicator(
                ts_code=tsc,
                start_date=START_DATE,
                end_date=END_DATE,
                refresh=False,
            )
        except Exception as exc:
            # Try without .SZ/.SH suffix (just 6-digit code)
            try:
                log.warning("  [%d/%d] %s failed with suffix, trying bare code: %s",
                           i, len(symbols), tsc, exc)
                df_one = supply.fetch_fina_indicator(
                    ts_code=sym,
                    start_date=START_DATE,
                    end_date=END_DATE,
                    refresh=False,
                )
            except Exception as exc2:
                log.warning("  [%d/%d] %s also failed: %s", i, len(symbols), sym, exc2)
                errors.append(tsc)
                df_one = pd.DataFrame()

        if df_one is not None and len(df_one) > 0:
            frames.append(df_one)
        else:
            skipped += 1

        if i % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            have = sum(len(f) for f in frames)
            log.info("  Progress %d/%d (%.0f%%) — %d rows collected, %d errors, %.1f stocks/sec",
                     i, len(symbols), 100 * i / len(symbols),
                     have, len(errors), rate)

        time.sleep(RATE_LIMIT_S)

    # 4. Combine & save
    if not frames:
        log.error("No data fetched at all!")
        sys.exit(1)

    result = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    result.to_parquet(OUTPUT, index=False)
    log.info("Saved %d rows to %s", len(result), OUTPUT)

    _report(result)
    if errors:
        log.warning("Errors (%d): %s", len(errors), errors[:10])


def _report(df: pd.DataFrame) -> None:
    """Print coverage stats."""
    n_stocks = df["symbol"].nunique()
    log.info("=" * 50)
    log.info("COVERAGE REPORT")
    log.info("=" * 50)
    log.info("Stocks with fina_indicator data: %d", n_stocks)
    log.info("Total rows: %d", len(df))
    log.info("Columns: %s", list(df.columns))
    if "announce_date" in df.columns:
        ad = df["announce_date"].dropna()
        log.info("Announce date range: %s ~ %s", ad.min(), ad.max())
    if "report_period" in df.columns:
        rp = df["report_period"].dropna()
        log.info("Report period range: %s ~ %s", rp.min(), rp.max())

    # Show sample stocks with data
    top = df["symbol"].value_counts().head(5)
    log.info("Top 5 stocks by row count:\n%s", top)


if __name__ == "__main__":
    main()
