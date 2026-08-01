#!/usr/bin/env python3
"""Fix winner_ratio unit: Tushare winner_rate (0-100) stored directly as winner_ratio.

Only divide values > 1 by 100 (Tushare values in 0-100 range).
Values <= 1 are already correct (either baostock 0-1 or rare Tushare <1%).
Safe write: temp + atomic rename.
"""

import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fix_bp")

V3_PATH = os.path.join("data", "panel_full_enriched_v3.parquet")


def main():
    logger.info("Loading V3: %s", V3_PATH)
    v3 = pd.read_parquet(V3_PATH)

    bp = v3["winner_ratio"]
    nn = bp.notna().sum()
    over1 = (bp > 1).sum()
    under1 = (bp <= 1).sum() & bp.notna()
    logger.info("Before fix:")
    logger.info("  non_null:    %d", nn)
    logger.info("  > 1 (Tushare 0-100): %d  → will ÷100", over1)
    logger.info("  ≤ 1 (correct 0-1):  %d  → keep as-is", under1)
    logger.info("  range: %.4f ~ %.4f", bp.min(), bp.max())

    # Divide only values > 1 by 100
    mask = bp > 1
    v3.loc[mask, "winner_ratio"] = v3.loc[mask, "winner_ratio"] / 100.0

    bp = v3["winner_ratio"]
    logger.info("After fix:")
    logger.info("  range: %.4f ~ %.4f", bp.min(), bp.max())
    logger.info("  > 1: %d (should be 0)", (bp > 1).sum())
    logger.info("  > 1.01: %d (tolerance for rounding)", (bp > 1.01).sum())

    # Safe write
    tmp = V3_PATH + ".tmp"
    logger.info("Writing to temp: %s", tmp)
    v3.to_parquet(tmp, index=False)
    os.replace(tmp, V3_PATH)
    logger.info("V3 saved: %d rows, %d cols", len(v3), len(v3.columns))


if __name__ == "__main__":
    main()
