#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enrich v3 panel with baostock HFQ data, CYQ fill, and bias features.

1. Merge baostock HFQ (open_hfq/high_hfq/low_hfq/close_hfq) into v3
2. Fallback raw close for 2023 dates not in baostock
3. Fill CYQ chip columns from baostock (baostock has only 30% NaN vs 83% in v3)
4. Compute bias features (bias_5/10/20/60/120/250) + derived (cross, amplitude, vol ratio)
5. Save back to panel_full_enriched_v3.parquet

Usage:
    python scripts/enrich_v3_from_baostock.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("enrich_v3")

V3_PATH = ROOT / "data" / "panel_full_enriched_v3.parquet"
BS_PATH = ROOT / "data" / "panel_full.parquet"

HFQ_COLS = ["open_hfq", "high_hfq", "low_hfq", "close_hfq"]
CYQ_COLS = [
    "benefit_part", "avg_cost",
    "pct_70_low", "pct_70_high", "pct_70_con",
    "pct_90_low", "pct_90_high", "pct_90_con",
    "cost_5pct", "cost_15pct", "cost_50pct",
    "cost_85pct", "cost_95pct", "weight_avg",
]
BIAS_WINDOWS = [5, 10, 20, 60, 120, 250]


def main():
    # ── 1. Load ──
    logger.info("Loading v3 panel: %s", V3_PATH)
    v3 = pd.read_parquet(V3_PATH)
    logger.info("  shape=%s, symbols=%d, date_range=%s ~ %s",
                v3.shape, v3["symbol"].nunique(), v3["date"].min(), v3["date"].max())

    logger.info("Loading baostock panel: %s", BS_PATH)
    bs = pd.read_parquet(BS_PATH)
    logger.info("  shape=%s, symbols=%d, date_range=%s ~ %s",
                bs.shape, bs["symbol"].nunique(), bs["date"].min(), bs["date"].max())

    # ── 2. Pre-merge HFQ / CYQ NaN stats ──
    for col in HFQ_COLS:
        pct = v3[col].isna().mean() * 100
        logger.info("  v3 %s NaN: %.2f%%", col, pct)
    for col in CYQ_COLS:
        pct = v3[col].isna().mean() * 100
        bs_pct = bs[col].isna().mean() * 100
        logger.info("  %s NaN: v3=%.2f%%  bs=%.2f%%", col, pct, bs_pct)

    # ── 3. Merge baostock HFQ + CYQ into v3 ──
    logger.info("Merging baostock HFQ + CYQ data into v3...")
    bs_fill = bs[["symbol", "date"] + HFQ_COLS + CYQ_COLS].copy()

    # Ensure same dtypes for merge keys
    v3["symbol"] = v3["symbol"].astype(str)
    bs_fill["symbol"] = bs_fill["symbol"].astype(str)

    # Left merge (v3 keeps all rows)
    before = v3.shape[0]
    v3 = v3.merge(bs_fill, on=["symbol", "date"], how="left", suffixes=("", "_bs"))
    assert before == v3.shape[0], "Merge should not change row count"

    # Fill v3 columns with baostock values
    fill_cols = HFQ_COLS + CYQ_COLS
    for col in fill_cols:
        bs_col = col + "_bs"
        if bs_col in v3.columns:
            filled = v3[col].isna().sum()
            v3[col] = v3[col].fillna(v3[bs_col])
            still_na = v3[col].isna().sum()
            filled_now = filled - still_na
            if filled_now > 0:
                logger.info("  %s: filled %d rows from baostock, %d still NaN",
                            col, filled_now, still_na)
            v3.drop(columns=[bs_col], inplace=True)

    # ── 4. Fallback: for 2023 dates, close_hfq ≈ close ──
    logger.info("HFQ fallback for 2023 (dates not in baostock)...")
    bs_dates = set(bs["date"].unique())
    v3_dates = set(v3["date"].unique())
    missing_dates = v3_dates - bs_dates
    logger.info("  %d dates in v3 not in baostock (all 2023 era)", len(missing_dates))

    # close_hfq: fallback to close
    before = v3["close_hfq"].isna().sum()
    v3["close_hfq"] = v3["close_hfq"].fillna(v3["close"])
    after = v3["close_hfq"].isna().sum()
    logger.info("  close_hfq: filled %d more rows from close, %d still NaN",
                before - after, after)

    # open_hfq: fallback to open
    for raw, hfq in [("open", "open_hfq"), ("high", "high_hfq"), ("low", "low_hfq")]:
        before = v3[hfq].isna().sum()
        v3[hfq] = v3[hfq].fillna(v3[raw])
        after = v3[hfq].isna().sum()
        logger.info("  %s: filled %d rows from %s, %d still NaN",
                    hfq, before - after, raw, after)

    # ── 5. Verify HFQ fill ──
    logger.info("HFQ fill summary:")
    for col in HFQ_COLS:
        pct = v3[col].isna().mean() * 100
        logger.info("  %s NaN: %.4f%%", col, pct)

    # ── 6. Compute bias features ──
    logger.info("Computing bias features (per stock rolling MAs)...")
    v3 = v3.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Pre-compute MAs for all windows at once using groupby rolling
    for w in BIAS_WINDOWS:
        col = f"bias_{w}"
        ma = v3.groupby("symbol")["close_hfq"].transform(
            lambda x: x.rolling(window=w, min_periods=1).mean()
        )
        v3[col] = v3["close_hfq"] / ma - 1
        logger.info("  bias_%d done", w)

    # ── 7. Derivative features ──
    logger.info("Computing derivative features...")

    # bias_5_20_cross: bias_5 - bias_20
    v3["bias_5_20_cross"] = v3["bias_5"] - v3["bias_20"]
    logger.info("  bias_5_20_cross done")

    # bias_20_60_cross: bias_20 - bias_60
    v3["bias_20_60_cross"] = v3["bias_20"] - v3["bias_60"]
    logger.info("  bias_20_60_cross done")

    # ma_vol_ratio_5_20: volume MA5 / volume MA20
    vol_ma5 = v3.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    vol_ma20 = v3.groupby("symbol")["volume"].transform(
        lambda x: x.rolling(window=20, min_periods=1).mean()
    )
    # Avoid division by zero on MA20
    vol_ma20_safe = vol_ma20.replace(0, np.nan)
    v3["ma_vol_ratio_5_20"] = vol_ma5 / vol_ma20_safe
    logger.info("  ma_vol_ratio_5_20 done")

    # amplitude_5d: (high.rolling(5).max() / low.rolling(5).min() - 1) * 100
    high_max5 = v3.groupby("symbol")["high"].transform(
        lambda x: x.rolling(window=5, min_periods=1).max()
    )
    low_min5 = v3.groupby("symbol")["low"].transform(
        lambda x: x.rolling(window=5, min_periods=1).min()
    )
    low_min5_safe = low_min5.replace(0, np.nan)
    v3["amplitude_5d"] = (high_max5 / low_min5_safe - 1) * 100
    logger.info("  amplitude_5d done")

    # ── 8. Print final summary ──
    new_cols = [f"bias_{w}" for w in BIAS_WINDOWS] + [
        "bias_5_20_cross", "bias_20_60_cross",
        "ma_vol_ratio_5_20", "amplitude_5d",
    ]
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("  Shape: %s", v3.shape)
    logger.info("  Columns: %d", len(v3.columns))
    logger.info("  Symbols: %d", v3["symbol"].nunique())
    logger.info("  Date range: %s ~ %s", v3["date"].min(), v3["date"].max())

    logger.info("New/improved columns NaN rates:")
    for col in HFQ_COLS + CYQ_COLS + new_cols:
        pct = v3[col].isna().mean() * 100
        if col in HFQ_COLS:
            logger.info("  %-25s  %.4f%%", col, pct)
        else:
            logger.info("  %-25s  %.2f%%", col, pct)

    # ── 9. Save ──
    logger.info("Saving to %s", V3_PATH)
    v3.to_parquet(V3_PATH, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
