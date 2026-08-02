# -*- coding: utf-8 -*-
"""Fetch historical SW daily index data for L1/L2/L3 from Tushare.

Reads sw_stock_classification.csv to get all unique SW index codes,
then fetches sw_daily(ts_code=xxx, start_date, end_date) for each.

Output: data/processed/sw_daily_history.parquet
  Columns: ts_code, trade_date, name, open, low, high, close,
           change, pct_change, vol, amount, pe, pb, float_mv, total_mv,
           level (L1/L2/L3)

Usage:
  python scripts/fetch_sw_daily_history.py                    # Full backfill
  python scripts/fetch_sw_daily_history.py --start 20230101   # Custom range
  python scripts/fetch_sw_daily_history.py --incremental      # Append new dates only
"""

import os
import sys
import time
import logging
import argparse
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

PROCESSED_DIR = os.getenv("PROCESSED_DIR", "data/processed")
SW_CSV = os.path.join(PROCESSED_DIR, "sw_stock_classification.csv")
OUTPUT = os.path.join(PROCESSED_DIR, "sw_daily_history.parquet")

DEFAULT_START = "20180101"
DEFAULT_END = "20260731"
API_DELAY = 0.15  # seconds between API calls


def get_all_sw_codes():
    """Read classification CSV, return dict {level: set of codes}."""
    df = pd.read_csv(SW_CSV, encoding="utf-8-sig", dtype=str)
    codes = {"L1": set(), "L2": set(), "L3": set()}
    for _, row in df.iterrows():
        for level, col in [
            ("L1", "sw_l1_code"),
            ("L2", "sw_l2_code"),
            ("L3", "sw_l3_code"),
        ]:
            c = str(row.get(col, "")).strip()
            if c and c != "nan":
                codes[level].add(c)
    logger.info(
        "SW codes: L1=%d, L2=%d, L3=%d",
        len(codes["L1"]),
        len(codes["L2"]),
        len(codes["L3"]),
    )
    return codes


def fetch_one_index(pro, ts_code, start_date, end_date):
    """Fetch sw_daily for one index code."""
    try:
        df = pro.sw_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is not None and len(df) > 0:
            return df
        return None
    except Exception as e:
        logger.warning("  Failed %s: %s", ts_code, e)
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch SW daily history")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYYMMDD")
    parser.add_argument("--end", default=DEFAULT_END, help="End date YYYYMMDD")
    parser.add_argument(
        "--incremental", action="store_true", help="Only fetch dates after existing max"
    )
    args = parser.parse_args()

    if not os.path.exists(SW_CSV):
        logger.error("SW classification CSV not found: %s", SW_CSV)
        sys.exit(1)

    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        logger.error("TUSHARE_TOKEN not set")
        sys.exit(1)
    pro = ts.pro_api(token)

    codes_map = get_all_sw_codes()

    # ── Incremental mode: find existing max date ──
    existing_df = None
    if args.incremental and os.path.exists(OUTPUT):
        existing_df = pd.read_parquet(OUTPUT)
        if "trade_date" in existing_df.columns and len(existing_df) > 0:
            max_date = existing_df["trade_date"].max()
            args.start = max_date
            logger.info("Incremental: fetching from %s", max_date)

    # ── Fetch all indices ──
    all_codes = []
    for level in ["L1", "L2", "L3"]:
        for code in sorted(codes_map[level]):
            all_codes.append((level, code))

    logger.info(
        "Fetching %d indices from %s to %s ...", len(all_codes), args.start, args.end
    )

    frames = []
    for i, (level, code) in enumerate(all_codes):
        df = fetch_one_index(pro, code, args.start, args.end)
        if df is not None:
            df["level"] = level
            frames.append(df)
        time.sleep(API_DELAY)
        if (i + 1) % 50 == 0:
            logger.info(
                "  Progress: %d/%d indices, %d rows so far",
                i + 1,
                len(all_codes),
                sum(len(f) for f in frames),
            )

    if not frames:
        logger.error("No data fetched!")
        sys.exit(1)

    new_df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Fetched: %d rows, %d indices", len(new_df), new_df["ts_code"].nunique()
    )

    # ── Merge with existing if incremental ──
    if existing_df is not None and len(existing_df) > 0:
        # Drop overlapping dates from existing
        new_dates = set(new_df["trade_date"].unique())
        existing_df = existing_df[~existing_df["trade_date"].isin(new_dates)]
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
        combined = combined.sort_values(["level", "ts_code", "trade_date"]).reset_index(
            drop=True
        )
        logger.info(
            "Combined: %d rows (existing %d + new %d)",
            len(combined),
            len(existing_df),
            len(new_df),
        )
        out_df = combined
    else:
        out_df = new_df.sort_values(["level", "ts_code", "trade_date"]).reset_index(
            drop=True
        )

    # ── Save ──
    out_df.to_parquet(OUTPUT, index=False)
    logger.info(
        "Saved to %s: %d rows, %d indices",
        OUTPUT,
        len(out_df),
        out_df["ts_code"].nunique(),
    )

    # ── Audit ──
    for level in ["L1", "L2", "L3"]:
        sub = out_df[out_df["level"] == level]
        logger.info(
            "  %s: %d indices, %d dates (%s ~ %s)",
            level,
            sub["ts_code"].nunique(),
            sub["trade_date"].nunique(),
            sub["trade_date"].min() if len(sub) else "N/A",
            sub["trade_date"].max() if len(sub) else "N/A",
        )


if __name__ == "__main__":
    main()
