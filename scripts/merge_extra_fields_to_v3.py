#!/usr/bin/env python3
"""将 baostock 新增字段合并回 v3 面板.

合并顺序:
  1. pctChg (本地计算, close 的日收益率)
  2. pcfNcfTTM (baostock K-line)
  3. 季度基本面 (baostock, forward-filled)

Usage:
    python scripts/merge_extra_fields_to_v3.py
    python scripts/merge_extra_fields_to_v3.py --dry-run
"""

import argparse
import logging
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

V3_PANEL = "data/panel_full_enriched_v3.parquet"
PCF_PATH = "data/bs_pcfNcfTTM.parquet"
PCTG_PATH = "data/bs_pctChg.parquet"
FF_PATH = "data/bs_fundamentals_ff.parquet"

NEW_COLS = [
    "pctChg",
    "pcfNcfTTM",
    "quickRatio",
    "cashRatio",
    "assetToEquity",
    "tangibleAssetToAsset",
    "ebitToInterest",
    "CFOToNP",
    "CFOToGr",
]


def dry_run_check():
    """检查所有输入文件是否存在."""
    for f in [PCF_PATH, PCTG_PATH, FF_PATH]:
        if os.path.exists(f):
            df = pd.read_parquet(f, columns=["symbol"] if f != FF_PATH else None)
            n = len(df) if f != FF_PATH else pd.read_parquet(f).shape[0]
            logger.info(f"  {os.path.basename(f)}: EXISTS ({n} rows)")
        else:
            logger.info(f"  {os.path.basename(f)}: MISSING")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--backup", action="store_true", help="合并前创建 v3 备份 (默认跳过)"
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN: checking input files...")
        dry_run_check()
        return

    # ── 加载 v3 ──
    logger.info(f"Loading v3 panel: {V3_PANEL}")
    panel = pd.read_parquet(V3_PANEL)
    orig_shape = panel.shape
    logger.info(f"  Original: {orig_shape[0]} rows, {orig_shape[1]} cols")

    # ── 备份 (可选) ──
    if args.backup:
        backup = V3_PANEL.replace(".parquet", "_backup.parquet")
        panel.to_parquet(backup, index=False)
        logger.info(f"Backup saved: {backup}")

    # ── 1. 合并 pctChg ──
    if os.path.exists(PCTG_PATH):
        pct = pd.read_parquet(PCTG_PATH)
        logger.info(f"pctChg: {len(pct)} rows, coverage={pct['pctChg'].notna().sum()}")
        old_len = len(panel)
        panel = panel.merge(pct, on=["symbol", "date"], how="left")
        logger.info(f"  After merge: {len(panel)} rows (was {old_len})")
    else:
        logger.warning(f"  {PCTG_PATH} not found, skipping pctChg")

    # ── 2. 合并 pcfNcfTTM ──
    if os.path.exists(PCF_PATH):
        pcf = pd.read_parquet(PCF_PATH)
        logger.info(
            f"pcfNcfTTM: {len(pcf)} rows, coverage={pcf['pcfNcfTTM'].notna().sum()}"
        )
        panel = panel.merge(pcf, on=["symbol", "date"], how="left")
    else:
        logger.warning(f"  {PCF_PATH} not found, skipping pcfNcfTTM")

    # ── 3. 合并季度基本面 ──
    if os.path.exists(FF_PATH):
        ff = pd.read_parquet(FF_PATH)
        logger.info(f"Fundamentals ff: {len(ff)} rows, {ff['symbol'].nunique()} stocks")
        panel = panel.merge(ff, on=["symbol", "date"], how="left")
    else:
        logger.warning(f"  {FF_PATH} not found, skipping fundamentals")

    # ── 统计 ──
    new_shape = panel.shape
    logger.info(
        f"Final: {new_shape[0]} rows, {new_shape[1]} cols "
        f"(added {new_shape[1] - orig_shape[1]} cols)"
    )

    for col in NEW_COLS:
        if col in panel.columns:
            nna = panel[col].notna().sum()
            logger.info(f"  {col}: {nna} ({nna / len(panel) * 100:.1f}%)")
        else:
            logger.info(f"  {col}: NOT PRESENT")

    # ── 保存 ──
    panel.to_parquet(V3_PANEL, index=False)
    logger.info(f"Saved: {V3_PANEL}")


if __name__ == "__main__":
    main()
