# -*- coding: utf-8 -*-
"""Bulk-fetch missing alt data from Tushare and rebuild panel v4.

Fixes:
  1. pre_close  — compute from close / (1 + pctChg/100)  [no API call]
  2. north_*    — fix merge bug (broadcast market-level by date)
  3. margin_*   — fetch ALL trading days (not sampled)
  4. holder_count — consolidate per-stock cache + merge_asof forward-fill
  5. sh_*       — fetch stk_holdertrade full range + merge_asof
  6. lhb_*      — merge existing cache (inherently sparse, no fix needed)

Usage:
    python scripts/fetch_tushare_bulk.py [--skip-margin] [--skip-holdertrade]
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# ── Setup ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_bulk")

import tushare as ts  # noqa: E402
from config import settings  # noqa: E402

pro = ts.pro_api(settings.TUSHARE_TOKEN)

PANEL_V3 = "data/panel_full_enriched_v3.parquet"
PANEL_V4 = f"data/panel_full_enriched_v4_{datetime.now().strftime('%Y%m%d')}.parquet"
CACHE_DIR = "data/supply_cache/alt_data"


def coverage_report(df: pd.DataFrame, columns: list[str], label: str = "") -> None:
    """Print NaN coverage for specified columns."""
    print(f"\n{'=' * 60}")
    print(f"  Coverage Report {label}")
    print(f"{'=' * 60}")
    for c in columns:
        if c in df.columns:
            pct = (1 - df[c].isna().mean()) * 100
            print(f"  {c:<30s}: {pct:6.1f}%  ({df[c].notna().sum():>8,} / {len(df):,})")
        else:
            print(f"  {c:<30s}: MISSING COLUMN")
    print()


# ════════════════════════════════════════════════════════════════
# STEP 0: Load panel + trading dates
# ════════════════════════════════════════════════════════════════
logger.info("STEP 0: Loading V3 panel...")
df = pd.read_parquet(PANEL_V3)
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

TARGET_COLS = [
    "pre_close",
    "pctChg",
    "margin_buy_amt",
    "short_sell_vol",
    "holder_count",
    "lhb_net_buy",
    "lhb_buy_amt",
    "lhb_sell_amt",
    "sh_net_change_sign",
    "sh_change_amt_total",
    "sh_change_vol",
    "sh_change_amt",
    "sh_net_sign",
]

coverage_report(df, TARGET_COLS, "(BEFORE)")

all_dates = sorted(df["date"].dropna().unique())
date_strs = [d.strftime("%Y%m%d") for d in all_dates]
logger.info(
    "Panel: %d rows, %d symbols, %d dates (%s ~ %s)",
    len(df),
    df["symbol"].nunique(),
    len(all_dates),
    date_strs[0],
    date_strs[-1],
)


# ════════════════════════════════════════════════════════════════
# STEP 1: Fix pre_close — compute from close & pctChg (no API call)
# ════════════════════════════════════════════════════════════════
logger.info("STEP 1: Fix pre_close from close / (1 + pctChg/100)...")

mask = df["pre_close"].isna() & df["close"].notna() & df["pctChg"].notna()
fixed_count = mask.sum()
df.loc[mask, "pre_close"] = df.loc[mask, "close"] / (1 + df.loc[mask, "pctChg"] / 100)
logger.info(
    "  Fixed %d rows (%.1f%% of NaN)",
    fixed_count,
    fixed_count / max(df["pre_close"].isna().sum() + fixed_count, 1) * 100,
)


# ════════════════════════════════════════════════════════════════
# STEP 2: Northbound — SKIPPED (not used as training data)
# ════════════════════════════════════════════════════════════════
logger.info("STEP 2: Northbound SKIPPED (not used as training data)")


# ════════════════════════════════════════════════════════════════
# STEP 3: Fetch margin_detail for ALL trading days
# ════════════════════════════════════════════════════════════════
skip_margin = "--skip-margin" in sys.argv
if not skip_margin:
    logger.info("STEP 3: Fetch margin_detail for all trading days...")

    # Load existing margin cache
    mg_cache = "data/supply_cache/alt_data/margin/all_margin_full.parquet"
    mg_existing = pd.DataFrame()
    if os.path.exists(mg_cache):
        mg_existing = pd.read_parquet(mg_cache)
        mg_existing["date"] = pd.to_datetime(mg_existing["date"])
        logger.info(
            "  Existing cache: %d rows, %d dates",
            len(mg_existing),
            mg_existing["date"].nunique(),
        )

    # Find missing dates
    existing_dates = (
        set(mg_existing["date"].dt.strftime("%Y%m%d").unique())
        if len(mg_existing)
        else set()
    )
    missing_dates = [d for d in date_strs if d not in existing_dates]
    logger.info("  Missing dates: %d (of %d total)", len(missing_dates), len(date_strs))

    mg_frames = [mg_existing] if len(mg_existing) else []
    for i, dt_str in enumerate(missing_dates):
        try:
            raw = pro.margin_detail(trade_date=dt_str)
            if raw is not None and len(raw) > 0:
                parsed = pd.DataFrame(
                    {
                        "symbol": raw["ts_code"]
                        .str.replace(".SZ", "")
                        .str.replace(".SH", ""),
                        "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
                        "margin_balance": pd.to_numeric(
                            raw.get("rzye", np.nan), errors="coerce"
                        ),
                        "short_balance": pd.to_numeric(
                            raw.get("rqye", np.nan), errors="coerce"
                        ),
                        "margin_buy_amt": pd.to_numeric(
                            raw.get("rzmre", np.nan), errors="coerce"
                        ),
                        "short_sell_vol": pd.to_numeric(
                            raw.get("rqmcl", np.nan), errors="coerce"
                        ),
                    }
                )
                mg_frames.append(parsed)
        except Exception as e:
            logger.debug("  margin %s failed: %s", dt_str, e)
        if i % 50 == 0:
            logger.info(
                "    ... %d/%d (cumulative: %d dates)",
                i,
                len(missing_dates),
                sum(
                    f["date"].nunique()
                    for f in mg_frames
                    if isinstance(f, pd.DataFrame)
                ),
            )
        time.sleep(0.2)

    if mg_frames:
        mg_all = pd.concat(mg_frames, ignore_index=True).drop_duplicates(
            subset=["symbol", "date"]
        )
        mg_all.to_parquet(mg_cache, index=False)
        logger.info(
            "  Total margin: %d rows, %d dates, %d symbols",
            len(mg_all),
            mg_all["date"].nunique(),
            mg_all["symbol"].nunique(),
        )

        # Merge into panel
        for c in [
            "margin_balance",
            "short_balance",
            "margin_buy_amt",
            "short_sell_vol",
        ]:
            if c in df.columns:
                df = df.drop(columns=c)
        df = df.merge(
            mg_all[
                [
                    "symbol",
                    "date",
                    "margin_balance",
                    "short_balance",
                    "margin_buy_amt",
                    "short_sell_vol",
                ]
            ],
            on=["symbol", "date"],
            how="left",
        )
else:
    logger.info("STEP 3: SKIPPED (--skip-margin)")


# ════════════════════════════════════════════════════════════════
# STEP 4: Consolidate holder_count + merge_asof forward-fill
# ════════════════════════════════════════════════════════════════
logger.info("STEP 4: Consolidate holder_count + merge_asof...")

hn_dir = os.path.join(CACHE_DIR, "holdernumber")
hn_cache = os.path.join(CACHE_DIR, "holdernumber_all_consolidated.parquet")

if os.path.exists(hn_cache):
    hn_all = pd.read_parquet(hn_cache)
    logger.info("  Loaded consolidated cache: %d rows", len(hn_all))
else:
    hn_files = (
        [f for f in os.listdir(hn_dir) if f.endswith(".parquet")]
        if os.path.exists(hn_dir)
        else []
    )
    logger.info("  Consolidating %d per-stock files...", len(hn_files))
    hn_frames = []
    for f in hn_files:
        try:
            tmp = pd.read_parquet(os.path.join(hn_dir, f))
            if "holder_count" in tmp.columns and len(tmp) > 0:
                hn_frames.append(tmp[["symbol", "date", "holder_count"]].copy())
        except Exception:
            pass
    if hn_frames:
        hn_all = pd.concat(hn_frames, ignore_index=True).drop_duplicates(
            subset=["symbol", "date"]
        )
        hn_all = hn_all.sort_values(["symbol", "date"]).reset_index(drop=True)
        hn_all.to_parquet(hn_cache, index=False)
        logger.info(
            "  Consolidated: %d rows, %d symbols",
            len(hn_all),
            hn_all["symbol"].nunique(),
        )
    else:
        hn_all = pd.DataFrame()

# Also fetch missing stocks via bulk stk_holdernumber (by period)
if len(hn_all) > 0:
    existing_hn_symbols = set(hn_all["symbol"].unique())
    panel_symbols = set(df["symbol"].unique())
    missing_symbols = panel_symbols - existing_hn_symbols
    logger.info(
        "  Missing symbols in holdernumber: %d (of %d)",
        len(missing_symbols),
        len(panel_symbols),
    )

    if missing_symbols:
        # Fetch by ts_code for missing stocks (batch)
        missing_list = sorted(missing_symbols)
        for i, sym in enumerate(missing_list):
            ts_code = sym + (".SH" if sym.startswith("6") else ".SZ")
            try:
                raw = pro.stk_holdernumber(ts_code=ts_code)
                if raw is not None and len(raw) > 0:
                    tmp = pd.DataFrame(
                        {
                            "symbol": raw["ts_code"]
                            .str.replace(".SZ", "")
                            .str.replace(".SH", ""),
                            "date": pd.to_datetime(
                                raw.get("ann_date", raw.get("end_date")),
                                format="%Y%m%d",
                                errors="coerce",
                            ),
                            "holder_count": pd.to_numeric(
                                raw.get("holder_num", np.nan), errors="coerce"
                            ),
                        }
                    )
                    hn_all = pd.concat([hn_all, tmp], ignore_index=True)
            except Exception:
                pass
            if i % 100 == 0 and i > 0:
                logger.info("    ... %d/%d missing symbols", i, len(missing_list))
            time.sleep(0.12)

        hn_all = (
            hn_all.drop_duplicates(subset=["symbol", "date"])
            .sort_values(["symbol", "date"])
            .reset_index(drop=True)
        )
        hn_all.to_parquet(hn_cache, index=False)
        logger.info(
            "  After fetch: %d rows, %d symbols",
            len(hn_all),
            hn_all["symbol"].nunique(),
        )

# Merge with forward-fill (manual: merge exact then ffill per symbol)
if len(hn_all) > 0:
    hn_all = hn_all.dropna(subset=["date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    if "holder_count" in df.columns:
        df = df.drop(columns="holder_count")
    df = df.merge(
        hn_all[["symbol", "date", "holder_count"]], on=["symbol", "date"], how="left"
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["holder_count"] = df.groupby("symbol")["holder_count"].ffill()
    logger.info("  holder_count merged with forward-fill")


# ════════════════════════════════════════════════════════════════
# STEP 5: Fetch stk_holdertrade full range + merge_asof
# ════════════════════════════════════════════════════════════════
skip_holdertrade = "--skip-holdertrade" in sys.argv
if not skip_holdertrade:
    logger.info("STEP 5: Fetch stk_holdertrade full range...")

    ht_cache = os.path.join(CACHE_DIR, "holdertrade_all_full.parquet")
    if os.path.exists(ht_cache):
        ht_all = pd.read_parquet(ht_cache)
        logger.info("  Loaded cache: %d rows", len(ht_all))
    else:
        # Fetch by year to keep pagination manageable
        year_ranges = [
            ("20230101", "20231231"),
            ("20240101", "20241231"),
            ("20250101", "20251231"),
            ("20260101", "20260728"),
        ]
        ht_frames = []
        for start, end in year_ranges:
            logger.info("  Fetching %s ~ %s...", start, end)
            offset = 0
            while True:
                try:
                    raw = pro.stk_holdertrade(
                        start_date=start, end_date=end, limit=3000, offset=offset
                    )
                    if raw is None or len(raw) == 0:
                        break
                    ht_frames.append(raw)
                    if len(raw) < 3000:
                        break
                    offset += 3000
                    time.sleep(0.3)
                except Exception as e:
                    logger.warning(
                        "  holdertrade %s-%s offset=%d failed: %s",
                        start,
                        end,
                        offset,
                        e,
                    )
                    break
            logger.info("    %s~%s: %d pages", start, end, len(ht_frames))

        if ht_frames:
            raw_all = pd.concat(ht_frames, ignore_index=True)
            # Parse
            change_vol = pd.to_numeric(
                raw_all.get("change_vol", 0), errors="coerce"
            ).fillna(0)
            avg_price = pd.to_numeric(
                raw_all.get("avg_price", 0), errors="coerce"
            ).fillna(0)
            change_amt = change_vol * avg_price
            in_de = raw_all.get("in_de", "")
            ht_all = pd.DataFrame(
                {
                    "symbol": raw_all["ts_code"]
                    .str.replace(".SZ", "")
                    .str.replace(".SH", ""),
                    "date": pd.to_datetime(
                        raw_all.get("ann_date", raw_all.get("trade_date", "")),
                        format="%Y%m%d",
                        errors="coerce",
                    ),
                    "sh_change_vol": change_vol,
                    "sh_change_amt": change_amt,
                    "sh_change_type": in_de,
                }
            )
            # Aggregate by symbol+date (multiple holders on same day)
            ht_all["sh_net_sign"] = in_de.apply(
                lambda x: (
                    1
                    if str(x).upper() == "IN"
                    else (-1 if str(x).upper() == "DE" else 0)
                )
            )
            ht_agg = (
                ht_all.groupby(["symbol", "date"])
                .agg(
                    sh_change_vol=("sh_change_vol", "sum"),
                    sh_change_amt=("sh_change_amt", "sum"),
                    sh_net_change_sign=("sh_net_sign", "sum"),  # +1 per IN, -1 per DE
                    sh_change_amt_total=("sh_change_amt", "sum"),
                )
                .reset_index()
            )
            # Net sign: >0 = net increase, <0 = net decrease
            ht_agg["sh_net_sign"] = ht_agg["sh_net_change_sign"].apply(
                lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
            )
            ht_all = ht_agg.sort_values(["symbol", "date"]).reset_index(drop=True)
            ht_all.to_parquet(ht_cache, index=False)
            logger.info(
                "  holdertrade: %d rows, %d symbols, %d dates",
                len(ht_all),
                ht_all["symbol"].nunique(),
                ht_all["date"].nunique(),
            )
        else:
            ht_all = pd.DataFrame()

    # Merge with forward-fill (manual: merge exact then ffill per symbol)
    if len(ht_all) > 0:
        ht_all = ht_all.dropna(subset=["date"]).drop_duplicates(
            subset=["symbol", "date"], keep="last"
        )
        ht_cols = [
            "sh_change_vol",
            "sh_change_amt",
            "sh_net_change_sign",
            "sh_change_amt_total",
            "sh_net_sign",
        ]
        for c in ht_cols:
            if c in df.columns:
                df = df.drop(columns=c)
        df = df.merge(
            ht_all[["symbol", "date"] + ht_cols], on=["symbol", "date"], how="left"
        )
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        for c in ht_cols:
            df[c] = df.groupby("symbol")[c].ffill()
        logger.info("  holdertrade merged with forward-fill")
else:
    logger.info("STEP 5: SKIPPED (--skip-holdertrade)")


# ════════════════════════════════════════════════════════════════
# STEP 6: Merge LHB (existing cache, inherently sparse)
# ════════════════════════════════════════════════════════════════
logger.info("STEP 6: Merge LHB from existing cache...")
lhb_cache = os.path.join(CACHE_DIR, "lhb", "all_20240102_20260727.parquet")
if os.path.exists(lhb_cache):
    lhb = pd.read_parquet(lhb_cache)
    lhb["date"] = pd.to_datetime(lhb["date"])
    lhb_cols = ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]
    for c in lhb_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.merge(lhb[["symbol", "date"] + lhb_cols], on=["symbol", "date"], how="left")
    logger.info("  LHB merged: %d rows in cache (inherently sparse)", len(lhb))


# ════════════════════════════════════════════════════════════════
# STEP 7: Save + final report
# ════════════════════════════════════════════════════════════════
logger.info("STEP 7: Save V4 panel...")
df.to_parquet(PANEL_V4, index=False)
logger.info("  Saved: %s (%d rows, %d cols)", PANEL_V4, len(df), len(df.columns))

coverage_report(df, TARGET_COLS, "(AFTER)")
logger.info("DONE. Output: %s", PANEL_V4)
