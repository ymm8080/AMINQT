#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble final enriched panel from per-source partials.

Reads panel_full_enriched_v3.parquet as base, left-joins each
data/enrich_parts/{source}.parquet, and writes the full enriched output.

Usage:
  python scripts/assemble_enriched.py [--output data/panel_full_enriched_v4.parquet]
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_PANEL = str(ROOT / "data" / "panel_full_enriched_v3.parquet")
PARTS_DIR = str(ROOT / "data" / "enrich_parts")
DEFAULT_OUTPUT = str(ROOT / "data" / "panel_full_enriched_v4.parquet")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("assemble")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble enriched panel from partials"
    )
    parser.add_argument("--panel", default=DEFAULT_PANEL, help="Base panel path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output path")
    parser.add_argument(
        "--parts-dir", default=PARTS_DIR, help="Directory with source partials"
    )
    args = parser.parse_args()

    # ── Load base panel ──
    logger.info("Loading base panel: %s", args.panel)
    panel = pd.read_parquet(args.panel)
    len(panel)
    panel["symbol"].nunique()
    base_cols = len(panel.columns)

    # Deduplicate / clean base
    stale = [c for c in panel.columns if c.endswith("_x") or c.endswith("_y")]
    if stale:
        logger.info("Dropping %d stale _x/_y columns", len(stale))
        panel = panel.drop(columns=stale)
    dupes = panel.columns[panel.columns.duplicated()].tolist()
    if dupes:
        logger.info("Dropping %d duplicate columns: %s", len(dupes), dupes)
        panel = panel.loc[:, ~panel.columns.duplicated()]

    logger.info(
        "Base panel: %d rows, %d symbols, %d cols (loaded %d, cleaned)",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
        base_cols,
    )

    # ── Merge each partial ──
    parts = sorted(glob.glob(os.path.join(args.parts_dir, "*.parquet")))
    if not parts:
        logger.warning("No partial files found in %s", args.parts_dir)
    else:
        logger.info(
            "Found %d partial files: %s",
            len(parts),
            [os.path.basename(p) for p in parts],
        )

    total_new_cols = 0
    for part_path in parts:
        source = os.path.splitext(os.path.basename(part_path))[0]
        logger.info("--- Merging %s ---", source)
        try:
            part = pd.read_parquet(part_path)
            logger.info("  %d rows, %d cols", len(part), len(part.columns))
        except Exception as exc:
            logger.warning("  Failed to read %s: %s", part_path, exc)
            continue

        if "symbol" not in part.columns or "date" not in part.columns:
            logger.warning("  Missing symbol/date keys, skipping")
            continue

        # Get new columns only (exclude symbol/date)
        new_cols = [c for c in part.columns if c not in ("symbol", "date")]
        # Avoid overwriting existing columns
        existing_overlap = [c for c in new_cols if c in panel.columns]
        if existing_overlap:
            logger.info("  Columns already in panel (skip): %s", existing_overlap)
            new_cols = [c for c in new_cols if c not in panel.columns]

        if not new_cols:
            logger.info("  No new columns to add")
            continue

        # Ensure date types match
        if part["date"].dtype != panel["date"].dtype:
            part["date"] = pd.to_datetime(part["date"])

        before = len(panel.columns)
        panel = panel.merge(
            part[["symbol", "date"] + new_cols],
            on=["symbol", "date"],
            how="left",
        )
        added = len(panel.columns) - before
        total_new_cols += added
        logger.info("  +%d new cols: %s", added, sorted(new_cols))

    # ── Coverage report ──
    logger.info("=" * 60)
    logger.info("ASSEMBLY COMPLETE")
    logger.info("  Rows:    %d", len(panel))
    logger.info("  Symbols: %d", panel["symbol"].nunique())
    logger.info(
        "  Columns: %d (base: %d, +%d new)",
        len(panel.columns),
        base_cols - len(stale) - len(dupes),
        total_new_cols,
    )
    logger.info(
        "  NaN%%:   %.2f%%",
        panel.isna().sum().sum() / (len(panel) * len(panel.columns)) * 100,
    )

    # Check key markers
    markers = {
        "northbound": "north_net_buy_sh",
        "margin": "margin_balance",
        "fina": "roe",
        "lhb": "lhb_net_buy",
        "holdernumber": "holder_count",
        "holdertrade": "sh_net_change_sign",
        "sector_index": "sw_ret_1d",
        "daily_basic": "pe_ttm",
        "stk_limit": "up_limit_raw",
        "cyq_tushare": "winner_ratio",
    }
    present = {k: v for k, v in markers.items() if v in panel.columns}
    missing = {k: v for k, v in markers.items() if v not in panel.columns}
    logger.info(
        "  Markers present (%d): %s",
        len(present),
        ", ".join(f"{k}({v})" for k, v in present.items()),
    )
    if missing:
        logger.warning(
            "  Markers MISSING (%d): %s",
            len(missing),
            ", ".join(f"{k}({v})" for k, v in missing.items()),
        )

    # ── Save ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    panel.to_parquet(args.output, index=False)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    logger.info("Saved: %s (%.1f MB)", args.output, size_mb)


if __name__ == "__main__":
    main()
