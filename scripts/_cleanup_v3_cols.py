#!/usr/bin/env python3
"""Remove leaked ts_code and end_date columns from v3 (from previous _fill_fina run)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cleanup")

V3_PATH = "data/panel_full_enriched_v3.parquet"

logger.info("Loading v3...")
v3 = pd.read_parquet(V3_PATH)
logger.info("v3: %d rows %d cols", len(v3), len(v3.columns))

# Check for leaked columns
leaked = [c for c in ["ts_code", "end_date"] if c in v3.columns]
if not leaked:
    logger.info("No leaked columns found. Nothing to do.")
    sys.exit(0)

logger.info("Removing leaked columns: %s", leaked)
v3 = v3.drop(columns=leaked)
logger.info("v3 after cleanup: %d rows %d cols", len(v3), len(v3.columns))

logger.info("Saving v3...")
v3.to_parquet(V3_PATH, index=False)
logger.info("Done.")
