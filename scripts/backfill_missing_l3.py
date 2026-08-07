"""Backfill missing L3 SW daily indices."""

import logging
import os
import time

import pandas as pd
import tushare as ts
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api(token)

OUT = "data/processed/sw_daily_history.parquet"

existing = pd.read_parquet(OUT)
l3 = existing[existing["level"] == "L3"]
existing_codes = set(l3["ts_code"].unique())

ic = pro.index_classify(level="L3", src="SW2021")
classify_codes = set(ic["index_code"].unique())
missing = sorted(classify_codes - existing_codes)

logger.info("Fetching %d missing L3 indices...", len(missing))

frames = []
ok = 0
fail = 0
for i, code in enumerate(missing):
    try:
        df = pro.sw_daily(ts_code=code, start_date="20180101", end_date="20260731")
        if df is not None and len(df) > 0:
            df["level"] = "L3"
            frames.append(df)
            ok += 1
            logger.info("  %s: %d rows", code, len(df))
        else:
            fail += 1
            logger.info("  %s: EMPTY", code)
    except Exception as e:
        fail += 1
        logger.warning("  %s: ERROR %s", code, e)
    time.sleep(0.15)
    if (i + 1) % 20 == 0:
        logger.info("  Progress: %d/%d (ok=%d, fail=%d)", i + 1, len(missing), ok, fail)

if frames:
    new = pd.concat(frames, ignore_index=True)
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["ts_code", "trade_date"])
    combined = combined.sort_values(["level", "ts_code", "trade_date"]).reset_index(
        drop=True
    )
    combined.to_parquet(OUT, index=False)
    l3_new = combined[combined["level"] == "L3"]["ts_code"].nunique()
    logger.info(
        "Saved: %d rows, %d total indices, L3=%d",
        len(combined),
        combined["ts_code"].nunique(),
        l3_new,
    )
else:
    logger.info("No new data fetched")

logger.info("Done: ok=%d, fail=%d", ok, fail)
