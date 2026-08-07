"""Fill short_sell_vol and sh_change_vol in V3 panel from Tushare API.

Sources:
  - short_sell_vol  <- Tushare margin_detail (rqmcl), daily per-stock
  - sh_change_vol   <- Tushare stk_holdertrade (change_vol), event-based + ffill

Usage:
    python scripts/fill_short_sell_vol_and_sh_change_vol.py
    python scripts/fill_short_sell_vol_and_sh_change_vol.py --skip-fetch
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fill_v3")

import tushare as ts  # noqa: E402

from config import settings  # noqa: E402

pro = ts.pro_api(settings.TUSHARE_TOKEN)

V3_PATH = PROJECT_ROOT / "data" / "panel_full_enriched_v3.parquet"
MARGIN_CACHE = (
    PROJECT_ROOT
    / "data"
    / "supply_cache"
    / "alt_data"
    / "margin"
    / "all_margin_full.parquet"
)
HOLDERTRADE_CACHE = (
    PROJECT_ROOT / "data" / "supply_cache" / "alt_data" / "holdertrade_all_full.parquet"
)
THROTTLE = 0.2


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


# ====================================================================
# STEP 0: Backup V3 (iron rule #11)
# ====================================================================
backup_name = (
    f"panel_full_enriched_v3_bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
)
backup_path = PROJECT_ROOT / "data" / backup_name
logger.info("STEP 0: Backing up V3 -> %s", backup_name)
shutil.copy2(V3_PATH, backup_path)
logger.info("  Backup done.")

# ====================================================================
# STEP 1: Load V3 panel
# ====================================================================
logger.info("STEP 1: Loading V3 panel...")
df = pd.read_parquet(V3_PATH)
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
all_dates = sorted(df["date"].dropna().unique())
date_strs = [d.strftime("%Y%m%d") for d in all_dates]
logger.info(
    "  Panel: %d rows, %d symbols, %d dates (%s ~ %s)",
    len(df),
    df["symbol"].nunique(),
    len(all_dates),
    date_strs[0],
    date_strs[-1],
)
TARGET_COLS = ["short_sell_vol", "sh_change_vol"]
coverage_report(df, TARGET_COLS, "(BEFORE)")

skip_fetch = "--skip-fetch" in sys.argv

# ====================================================================
# STEP 2: Load / fetch margin_detail cache for short_sell_vol
# ====================================================================
logger.info("STEP 2: Load / fetch margin_detail (short_sell_vol)...")
mg_existing = pd.DataFrame()
if MARGIN_CACHE.exists():
    mg_existing = pd.read_parquet(MARGIN_CACHE)
    mg_existing["date"] = pd.to_datetime(mg_existing["date"])
    logger.info(
        "  Existing cache: %d rows, %d dates, %d symbols",
        len(mg_existing),
        mg_existing["date"].nunique(),
        mg_existing["symbol"].nunique(),
    )

if not skip_fetch:
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
        if i % 50 == 0 and i > 0:
            cum = sum(
                f["date"].nunique() for f in mg_frames if isinstance(f, pd.DataFrame)
            )
            logger.info(
                "    ... %d/%d (cumulative: %d dates)", i, len(missing_dates), cum
            )
        time.sleep(THROTTLE)

    if mg_frames:
        mg_all = pd.concat(mg_frames, ignore_index=True).drop_duplicates(
            subset=["symbol", "date"]
        )
        MARGIN_CACHE.parent.mkdir(parents=True, exist_ok=True)
        mg_all.to_parquet(MARGIN_CACHE, index=False)
        logger.info(
            "  Total margin cache: %d rows, %d dates, %d symbols",
            len(mg_all),
            mg_all["date"].nunique(),
            mg_all["symbol"].nunique(),
        )
    else:
        mg_all = mg_existing
else:
    mg_all = mg_existing
    logger.info("  Skipped fetch (--skip-fetch), using cache only.")

# ====================================================================
# STEP 3: Load / fetch stk_holdertrade cache for sh_change_vol
# ====================================================================
logger.info("STEP 3: Load / fetch stk_holdertrade (sh_change_vol)...")
ht_existing = pd.DataFrame()
if HOLDERTRADE_CACHE.exists():
    ht_existing = pd.read_parquet(HOLDERTRADE_CACHE)
    ht_existing["date"] = pd.to_datetime(ht_existing["date"])
    logger.info(
        "  Existing cache: %d rows, %d dates, %d symbols",
        len(ht_existing),
        ht_existing["date"].nunique(),
        ht_existing["symbol"].nunique(),
    )

if not skip_fetch:
    year_ranges = [
        ("20230101", "20231231"),
        ("20240101", "20241231"),
        ("20250101", "20251231"),
        ("20260101", "20260728"),
    ]
    ht_frames_raw: list[pd.DataFrame] = []
    for start, end in year_ranges:
        if len(ht_existing) > 0:
            covered = ht_existing[
                (ht_existing["date"] >= pd.to_datetime(start, format="%Y%m%d"))
                & (ht_existing["date"] <= pd.to_datetime(end, format="%Y%m%d"))
            ]
            if len(covered) > 0:
                logger.info(
                    "  %s~%s: cache has %d rows, skipping", start, end, len(covered)
                )
                continue
        logger.info("  Fetching %s ~ %s...", start, end)
        offset = 0
        _PAGE = 3000
        while True:
            try:
                raw = pro.stk_holdertrade(
                    start_date=start, end_date=end, limit=_PAGE, offset=offset
                )
                if raw is None or len(raw) == 0:
                    break
                ht_frames_raw.append(raw)
                if len(raw) < _PAGE:
                    break
                offset += _PAGE
                time.sleep(0.3)
            except Exception as e:
                logger.warning(
                    "  holdertrade %s-%s offset=%d failed: %s", start, end, offset, e
                )
                break
        logger.info("    %s~%s: %d pages", start, end, len(ht_frames_raw))

    if ht_frames_raw:
        raw_all = pd.concat(ht_frames_raw, ignore_index=True)
        change_vol = pd.to_numeric(
            raw_all.get("change_vol", 0), errors="coerce"
        ).fillna(0)
        avg_price = pd.to_numeric(raw_all.get("avg_price", 0), errors="coerce").fillna(
            0
        )
        change_amt = change_vol * avg_price
        in_de = raw_all.get("in_de", "")
        ht_new = pd.DataFrame(
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
        ht_new["sh_net_sign"] = in_de.apply(
            lambda x: (
                1 if str(x).upper() == "IN" else (-1 if str(x).upper() == "DE" else 0)
            )
        )
        ht_agg = (
            ht_new.groupby(["symbol", "date"])
            .agg(
                sh_change_vol=("sh_change_vol", "sum"),
                sh_change_amt=("sh_change_amt", "sum"),
                sh_net_change_sign=("sh_net_sign", "sum"),
                sh_change_amt_total=("sh_change_amt", "sum"),
            )
            .reset_index()
        )
        ht_agg["sh_net_sign"] = ht_agg["sh_net_change_sign"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        if len(ht_existing) > 0:
            ht_all = pd.concat([ht_existing, ht_agg], ignore_index=True)
            ht_all = ht_all.drop_duplicates(subset=["symbol", "date"], keep="last")
        else:
            ht_all = ht_agg
        ht_all = ht_all.sort_values(["symbol", "date"]).reset_index(drop=True)
        HOLDERTRADE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        ht_all.to_parquet(HOLDERTRADE_CACHE, index=False)
        logger.info(
            "  Total holdertrade cache: %d rows, %d symbols, %d dates",
            len(ht_all),
            ht_all["symbol"].nunique(),
            ht_all["date"].nunique(),
        )
    else:
        ht_all = ht_existing
        logger.info("  No new data fetched, using existing cache.")
else:
    ht_all = ht_existing
    logger.info("  Skipped fetch (--skip-fetch), using cache only.")

# ====================================================================
# STEP 4: Merge margin (short_sell_vol) into V3
# ====================================================================
logger.info("STEP 4: Merge short_sell_vol from margin cache into V3...")
if len(mg_all) > 0:
    margin_cols = [
        "margin_balance",
        "short_balance",
        "margin_buy_amt",
        "short_sell_vol",
    ]
    for c in margin_cols:
        if c in df.columns:
            df = df.drop(columns=c)
    df = df.merge(
        mg_all[["symbol", "date"] + margin_cols],
        on=["symbol", "date"],
        how="left",
    )
    logger.info("  short_sell_vol merged from margin_detail cache.")
else:
    logger.warning("  Margin cache empty, short_sell_vol not updated.")

# ====================================================================
# STEP 5: Merge holdertrade (sh_change_vol) into V3 with forward-fill
# ====================================================================
logger.info(
    "STEP 5: Merge sh_change_vol from holdertrade cache into V3 (with forward-fill)..."
)
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
        ht_all[["symbol", "date"] + ht_cols],
        on=["symbol", "date"],
        how="left",
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    for c in ht_cols:
        df[c] = df.groupby("symbol")[c].ffill()
    logger.info("  sh_change_vol merged with forward-fill.")
else:
    logger.warning("  Holdertrade cache empty, sh_change_vol not updated.")

# ====================================================================
# STEP 6: Save V3 + final coverage report
# ====================================================================
logger.info("STEP 6: Saving V3 panel...")
df.to_parquet(V3_PATH, index=False)
logger.info("  Saved: %s (%d rows, %d cols)", V3_PATH.name, len(df), len(df.columns))
coverage_report(df, TARGET_COLS, "(AFTER)")
logger.info("DONE. Backup at: %s", backup_name)
