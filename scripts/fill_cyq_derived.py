#!/usr/bin/env python3
"""Fill remaining CYQ columns in V3 using Tushare cost percentile data.

Derivation (verified 100% exact within same source):
  pct_70_low  = cost_15pct
  pct_70_high = cost_85pct
  pct_70_con  = (cost_85pct - cost_15pct) / (cost_85pct + cost_15pct)
  pct_90_low  = cost_5pct
  pct_90_high = cost_95pct
  pct_90_con  = (cost_95pct - cost_5pct) / (cost_95pct + cost_5pct)
  weight_avg  = avg_cost  (same Tushare source, fetch_cyq_remaining mapped weight_avg→avg_cost)

Only fills NaN values; existing non-NaN data is preserved.
Safe write: temp file + atomic rename.
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fill_cyq")

V3_PATH = os.path.join("data", "panel_full_enriched_v3.parquet")

DERIVE_MAP = {
    "pct_70_low": ("cost_15pct", None),
    "pct_70_high": ("cost_85pct", None),
    "pct_70_con": ("cost_85pct", "cost_15pct"),  # (hi, lo) → (hi-lo)/(hi+lo)
    "pct_90_low": ("cost_5pct", None),
    "pct_90_high": ("cost_95pct", None),
    "pct_90_con": ("cost_95pct", "cost_5pct"),  # (hi, lo) → (hi-lo)/(hi+lo)
}

# weight_avg: same Tushare source as avg_cost (fetch_cyq_remaining mapped weight_avg→avg_cost)
WEIGHT_AVG_SRC = "avg_cost"


def main():
    logger.info("Loading V3: %s", V3_PATH)
    v3 = pd.read_parquet(V3_PATH)
    logger.info("  shape=%s, cols=%d", v3.shape, len(v3.columns))

    # ── Before stats ──
    all_cols = list(DERIVE_MAP.keys()) + [WEIGHT_AVG_SRC]
    logger.info("Before fill:")
    for c in all_cols:
        if c in v3.columns:
            nn = v3[c].notna().sum()
            logger.info(
                "  %-15s  NaN=%6.2f%%  non_null=%d", c, v3[c].isna().mean() * 100, nn
            )
        else:
            logger.info("  %-15s  NOT IN PANEL", c)

    # ── Derive pct_70/pct_90 ──
    for target, (hi_col, lo_col) in DERIVE_MAP.items():
        if hi_col not in v3.columns:
            logger.warning("  %s not in panel, skip %s", hi_col, target)
            continue

        before_nn = v3[target].notna().sum() if target in v3.columns else 0
        if target not in v3.columns:
            v3[target] = np.nan

        if lo_col is None:
            # Direct copy (low/high price = cost percentile)
            mask = v3[target].isna() & v3[hi_col].notna()
            v3.loc[mask, target] = v3.loc[mask, hi_col]
        else:
            # Concentration: (hi - lo) / (hi + lo)
            mask = v3[target].isna() & v3[hi_col].notna() & v3[lo_col].notna()
            hi = v3.loc[mask, hi_col]
            lo = v3.loc[mask, lo_col]
            denom = (hi + lo).replace(0, np.nan)
            v3.loc[mask, target] = (hi - lo) / denom

        after_nn = v3[target].notna().sum()
        filled = after_nn - before_nn
        logger.info(
            "  %s: filled %d rows (%d → %d non-null, %.2f%% NaN remains)",
            target,
            filled,
            before_nn,
            after_nn,
            v3[target].isna().mean() * 100,
        )

    # ── weight_avg from avg_cost ──
    if WEIGHT_AVG_SRC in v3.columns and "weight_avg" in v3.columns:
        before_nn = v3["weight_avg"].notna().sum()
        mask = v3["weight_avg"].isna() & v3[WEIGHT_AVG_SRC].notna()
        v3.loc[mask, "weight_avg"] = v3.loc[mask, WEIGHT_AVG_SRC]
        after_nn = v3["weight_avg"].notna().sum()
        logger.info(
            "  weight_avg: filled %d rows (%d → %d non-null, %.2f%% NaN remains)",
            after_nn - before_nn,
            before_nn,
            after_nn,
            v3["weight_avg"].isna().mean() * 100,
        )

    # ── After stats ──
    logger.info("After fill:")
    for c in all_cols:
        nn = v3[c].notna().sum()
        logger.info(
            "  %-15s  NaN=%6.2f%%  non_null=%d", c, v3[c].isna().mean() * 100, nn
        )

    # ── Safe write ──
    tmp_path = V3_PATH + ".tmp"
    logger.info("Writing to temp: %s", tmp_path)
    v3.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, V3_PATH)
    logger.info("V3 saved: %d rows, %d cols", len(v3), len(v3.columns))


if __name__ == "__main__":
    main()
