"""Retry stk_holdertrade fetch + merge into V4 panel."""

import logging
import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("holdertrade")

import tushare as ts

from config import settings

pro = ts.pro_api(settings.TUSHARE_TOKEN)

CACHE_DIR = "data/supply_cache/alt_data"

# Step 1: Fetch stk_holdertrade by year
ht_cache = os.path.join(CACHE_DIR, "holdertrade_all_full.parquet")
if os.path.exists(ht_cache):
    ht_all = pd.read_parquet(ht_cache)
    logger.info("Loaded existing cache: %d rows", len(ht_all))
else:
    year_ranges = [
        ("20230101", "20231231"),
        ("20240101", "20241231"),
        ("20250101", "20251231"),
        ("20260101", "20260728"),
    ]
    ht_frames = []
    for start, end in year_ranges:
        logger.info("Fetching %s ~ %s...", start, end)
        offset = 0
        while True:
            try:
                raw = pro.stk_holdertrade(
                    start_date=start, end_date=end, limit=3000, offset=offset
                )
                if raw is None or len(raw) == 0:
                    break
                ht_frames.append(raw)
                logger.info("  page offset=%d: %d rows", offset, len(raw))
                if len(raw) < 3000:
                    break
                offset += 3000
                time.sleep(0.3)
            except Exception as e:
                logger.warning("  failed: %s", e)
                time.sleep(2)
                # retry once
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
                except Exception as e2:
                    logger.error("  retry also failed: %s", e2)
                    break
        logger.info("  %s~%s: %d pages so far", start, end, len(ht_frames))

    if ht_frames:
        raw_all = pd.concat(ht_frames, ignore_index=True)
        change_vol = pd.to_numeric(
            raw_all.get("change_vol", 0), errors="coerce"
        ).fillna(0)
        avg_price = pd.to_numeric(raw_all.get("avg_price", 0), errors="coerce").fillna(
            0
        )
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
            }
        )
        ht_all["sh_net_sign_raw"] = in_de.apply(
            lambda x: (
                1 if str(x).upper() == "IN" else (-1 if str(x).upper() == "DE" else 0)
            )
        )
        ht_agg = (
            ht_all.groupby(["symbol", "date"])
            .agg(
                sh_change_vol=("sh_change_vol", "sum"),
                sh_change_amt=("sh_change_amt", "sum"),
                sh_net_change_sign=("sh_net_sign_raw", "sum"),
                sh_change_amt_total=("sh_change_amt", "sum"),
            )
            .reset_index()
        )
        ht_agg["sh_net_sign"] = ht_agg["sh_net_change_sign"].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        )
        ht_all = ht_agg.sort_values(["symbol", "date"]).reset_index(drop=True)
        ht_all.to_parquet(ht_cache, index=False)
        logger.info(
            "holdertrade: %d rows, %d symbols, %d dates",
            len(ht_all),
            ht_all["symbol"].nunique(),
            ht_all["date"].nunique(),
        )
    else:
        logger.error("No holdertrade data fetched!")
        sys.exit(1)

# Step 2: Load V4 panel and merge
import glob

v4_files = sorted(glob.glob("data/panel_full_enriched_v4_*.parquet"))
if not v4_files:
    logger.error("No V4 panel found!")
    sys.exit(1)
v4_path = v4_files[-1]
logger.info("Loading %s...", v4_path)
df = pd.read_parquet(v4_path)

# Merge holdertrade with forward-fill
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
df = df.merge(ht_all[["symbol", "date"] + ht_cols], on=["symbol", "date"], how="left")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
for c in ht_cols:
    df[c] = df.groupby("symbol")[c].ffill()
logger.info("holdertrade merged with forward-fill")

# Save
df.to_parquet(v4_path, index=False)
logger.info("Saved: %s (%d rows, %d cols)", v4_path, len(df), len(df.columns))

# Coverage report
print("\n=== sh_* Coverage (AFTER) ===")
for c in ht_cols:
    if c in df.columns:
        pct = (1 - df[c].isna().mean()) * 100
        print(f"  {c:<25s}: {pct:6.1f}%  ({df[c].notna().sum():>8,} / {len(df):,})")
