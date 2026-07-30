#!/usr/bin/env python3
"""Diagnose why fina_indicator fill didn't improve 2023 coverage."""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diag")

V3_PATH = "data/panel_full_enriched_v3.parquet"

RAW_TO_NORM = {
    "ann_date": "announce_date",
    "end_date": "report_period",
    "ts_code": "_ts_code",
    "roe_dt": "roe_deducted",
    "np_margin": "net_margin",
    "netprofit_margin": "net_margin",
    "dt_eps_yoy": "eps_yoy",
    "or_yoy": "rev_yoy",
    "netprofit_yoy": "profit_yoy",
    "cf_sales": "op_cf_ratio",
    "debt_to_assets": "debt_ratio",
    "assets_turn": "asset_turnover",
    "ar_turn": "ar_turnover",
    "inv_turn": "inventory_turnover",
}


def normalize_fina(df):
    if "ann_date" in df.columns and "announce_date" not in df.columns:
        df = df.rename(columns=RAW_TO_NORM)
        logger.info("  raw Tushare format -> normalized (%d cols)", len(df.columns))
    if "symbol" not in df.columns and "_ts_code" in df.columns:
        df["symbol"] = df["_ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
    if "announce_date" in df.columns:
        df["announce_date"] = pd.to_datetime(df["announce_date"], format="mixed", errors="coerce")
    if "report_period" in df.columns:
        df["report_period"] = pd.to_datetime(df["report_period"], format="mixed", errors="coerce")
    for c in df.columns:
        if c not in ("symbol", "_ts_code", "announce_date", "report_period"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# Step 1: Check what the new cache looks like BEFORE normalization
logger.info("=== Step 1: Inspect new cache file ===")
new_raw = pd.read_parquet("data/supply_cache/alt_data/fina_indicator/all_20230101_20260728.parquet")
logger.info("New cache: %d rows, cols=%s", len(new_raw), new_raw.columns.tolist())
logger.info("ann_date dtype: %s, sample: %s", new_raw["ann_date"].dtype, new_raw["ann_date"].head(3).tolist())
new_norm = normalize_fina(new_raw.copy())
logger.info("After normalize: announce_date dtype=%s, sample=%s",
            new_norm["announce_date"].dtype, new_norm["announce_date"].head(3).tolist())
logger.info("New cache date range: %s ~ %s",
            new_norm["announce_date"].min(), new_norm["announce_date"].max())

# Step 2: Check per-stock file format
logger.info("\n=== Step 2: Inspect a per-stock file ===")
fina_dir = "data/supply_cache/alt_data/fina_indicator"
per_files = sorted(f for f in os.listdir(fina_dir) if f.endswith(".parquet") and not f.startswith("all_"))
if per_files:
    sample = pd.read_parquet(os.path.join(fina_dir, per_files[0]))
    logger.info("Per-stock file %s: %d rows, cols=%s", per_files[0], len(sample), sample.columns.tolist())
    if "ann_date" in sample.columns:
        logger.info("  has ann_date, dtype=%s", sample["ann_date"].dtype)
    if "announce_date" in sample.columns:
        logger.info("  has announce_date, dtype=%s, sample=%s",
                    sample["announce_date"].dtype, sample["announce_date"].head(3).tolist())
    sample_norm = normalize_fina(sample.copy())
    logger.info("  after normalize: cols=%s", sample_norm.columns.tolist())

# Step 3: Simulate the full merge process
logger.info("\n=== Step 3: Simulate full merge ===")
files = sorted(f for f in os.listdir(fina_dir) if f.endswith(".parquet"))
frames = []
for f in files:
    try:
        df = pd.read_parquet(os.path.join(fina_dir, f))
        if len(df) == 0:
            continue
        df = normalize_fina(df)
        if "symbol" in df.columns:
            frames.append(df)
    except Exception as e:
        logger.debug("  %s skip: %s", f, e)

fina = pd.concat(frames, ignore_index=True)
logger.info("Concatenated: %d rows", len(fina))

# Check announce_date dtype
logger.info("announce_date dtype: %s", fina["announce_date"].dtype)
logger.info("announce_date range: %s ~ %s", fina["announce_date"].min(), fina["announce_date"].max())

# Check for NaT in announce_date
nat_count = fina["announce_date"].isna().sum()
logger.info("NaT in announce_date: %d (%.1f%%)", nat_count, nat_count / len(fina) * 100)

fina = fina.sort_values(["symbol", "announce_date"])
fina = fina.groupby(["symbol", "announce_date"], sort=False).first().reset_index()
logger.info("After groupby: %d rows, %d stocks", len(fina), fina["symbol"].nunique())

# Step 4: Check 2023 H1 data
fina["date"] = pd.to_datetime(fina["announce_date"])
fina = fina.dropna(subset=["date"])
early_2023 = fina[(fina["date"] >= "2023-01-01") & (fina["date"] < "2023-07-01")]
logger.info("\n=== Step 4: Early 2023 fina data ===")
logger.info("Pre-2023-07-01 rows: %d, stocks: %d", len(early_2023), early_2023["symbol"].nunique())
if len(early_2023) > 0:
    logger.info("Sample: %s", early_2023[["symbol", "date", "roe", "eps_yoy"]].head(5).to_string())

# Step 5: Check v3 date format
logger.info("\n=== Step 5: V3 date format ===")
v3 = pd.read_parquet(V3_PATH, columns=["symbol", "date", "roe"])
logger.info("V3 date dtype: %s, range: %s ~ %s", v3["date"].dtype, v3["date"].min(), v3["date"].max())

# Step 6: Check a specific stock - 000620 (was 100% NaN)
logger.info("\n=== Step 6: Check stock 000620 ===")
stock_data = fina[fina["symbol"] == "000620"]
logger.info("Fina data for 000620: %d rows", len(stock_data))
if len(stock_data) > 0:
    logger.info("  dates: %s", stock_data["date"].tolist())
    logger.info("  roe: %s", stock_data["roe"].tolist())

v3_stock = v3[v3["symbol"] == "000620"]
logger.info("V3 data for 000620: %d rows, roe NaN=%d (%.1f%%)",
            len(v3_stock), v3_stock["roe"].isna().sum(), v3_stock["roe"].isna().mean() * 100)
