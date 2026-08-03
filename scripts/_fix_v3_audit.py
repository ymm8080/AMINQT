#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 V3 面板审计发现的数据质量问题 (2026-08-02):
  1. volume 为 null 的 2,250 行是停牌日 (open==high==low==close), 但 is_suspended 错误为 False.
     修复: 置 is_suspended=True, volume=0, amount=0, turnover_rate=0 (满足 OHLCV 校验).
  2. board 列有 10 种大小写/中文变体, 归一化为 main/GEM/STAR (board_of 标准).
  3. announce_date 仅 28 个 symbol 有值 (merge gap). 从 fina backfill 缓存 (3,244 文件) 按
     merge_asof(backward) 补全, 供公告事件特征 (feature_engine dim_announcement) 使用.

WORM: 改写前备份, os.replace 原子写. 严禁与其他面板写脚本并发.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fix_v3_audit")

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
FINA_DIR = (
    ROOT
    / "data"
    / "supply_cache"
    / "alt_data"
    / "fina_indicator"
    / "backfill_20260802_full"
)

from app.pipeline1.cleaning_pipeline import board_of  # noqa: E402


def fix_suspension(df: pd.DataFrame) -> pd.DataFrame:
    """volume null + 平盘 OHLC → 停牌日: is_suspended=True, volume/amount/turnover=0."""
    mask = (
        df["volume"].isna()
        & (df["open"] == df["close"])
        & (df["high"] == df["low"])
        & (df["open"] == df["high"])
    )
    n = int(mask.sum())
    if n:
        df.loc[mask, "is_suspended"] = True
        for c in ("volume", "amount", "turnover_rate"):
            if c in df.columns:
                df.loc[mask, c] = 0
    logger.info("停牌修复: %d 行 is_suspended=True, volume/amount/turnover=0", n)
    return df


def normalize_board(df: pd.DataFrame) -> pd.DataFrame:
    """board 归一化: 全部按 symbol 前缀 → main/GEM/STAR."""
    valid = {"main", "GEM", "STAR"}
    bad = ~df["board"].isin(valid)
    n = int(bad.sum())
    if n:
        df.loc[bad, "board"] = df.loc[bad, "symbol"].map(board_of)
    logger.info("board 归一化: %d 行变体 → main/GEM/STAR", n)
    return df


def load_fina_announce_dates() -> pd.DataFrame:
    """从 fina backfill 缓存提取 (symbol, announce_date) 唯一对."""
    if not FINA_DIR.is_dir():
        logger.warning("fina backfill 缓存不存在: %s", FINA_DIR)
        return pd.DataFrame()
    rows = []
    for f in sorted(FINA_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        if "symbol" not in df.columns or "ann_date" not in df.columns:
            continue
        sym = df["symbol"].iloc[0] if df["symbol"].nunique() == 1 else None
        if sym is None:
            continue
        ann = pd.to_datetime(df["ann_date"], format="mixed", errors="coerce").dropna()
        if len(ann):
            rows.append(pd.DataFrame({"symbol": sym, "announce_date": ann}))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["symbol", "announce_date"]).sort_values(
        ["symbol", "announce_date"]
    )
    logger.info("fina announce_date: %d 行, %d 股", len(out), out["symbol"].nunique())
    return out


def fix_announce_date(panel: pd.DataFrame, ann: pd.DataFrame) -> pd.DataFrame:
    """按 merge_asof(backward) 补全 announce_date (PIT 语义, 与 panel_builder 一致)."""
    if len(ann) == 0:
        return panel
    # 只补 announce_date 为空的行; 已有值的 (28 股) 不动
    fill_mask = panel["announce_date"].isna()
    if not fill_mask.any():
        logger.info("announce_date 已全覆盖, 跳过")
        return panel
    panel_sorted = panel.sort_values("date")
    right = ann.sort_values("announce_date").rename(
        columns={"announce_date": "right_ann"}
    )
    merged = pd.merge_asof(
        panel_sorted.loc[fill_mask, ["symbol", "date"]],
        right,
        left_on="date",
        right_on="right_ann",
        by="symbol",
        direction="backward",
    )
    merged = merged.rename(columns={"right_ann": "announce_date"})
    filled = merged[merged["announce_date"].notna()]
    n = len(filled)
    if n:
        panel.loc[filled.index, "announce_date"] = filled["announce_date"].values
    logger.info("announce_date 补全: %d 行 (%.1f%% 覆盖率)", n, n / len(panel) * 100)
    return panel


def main():
    t0 = time.time()
    if not os.path.exists(PANEL):
        logger.error("面板不存在: %s", PANEL)
        return
    logger.info("加载面板: %s", PANEL)
    panel = pd.read_parquet(PANEL)
    logger.info(
        "面板: %d 行 %d 列 %d 股, %s ~ %s",
        len(panel),
        len(panel.columns),
        panel["symbol"].nunique(),
        panel["date"].min(),
        panel["date"].max(),
    )

    before = {
        "volume_null": int(panel["volume"].isna().sum()),
        "is_suspended_true": int(panel["is_suspended"].astype(bool).sum()),
        "board_variants": int(panel["board"].nunique()),
        "announce_date_nn": int(panel["announce_date"].notna().sum()),
    }
    logger.info("修复前: %s", before)

    # ── 1. 停牌标记 ──
    panel = fix_suspension(panel)
    # ── 2. board 归一化 ──
    panel = normalize_board(panel)
    # ── 3. announce_date 补全 ──
    ann = load_fina_announce_dates()
    panel = fix_announce_date(panel, ann)

    after = {
        "volume_null": int(panel["volume"].isna().sum()),
        "is_suspended_true": int(panel["is_suspended"].astype(bool).sum()),
        "board_variants": int(panel["board"].nunique()),
        "announce_date_nn": int(panel["announce_date"].notna().sum()),
    }
    logger.info("修复后: %s", after)

    # ── 4. WORM 备份 + 原子写 ──
    backup = PANEL.replace(
        ".parquet",
        "_preauditfix_{}.parquet".format(pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")),
    )
    shutil.copy2(PANEL, backup)
    logger.info("备份: %s", backup)
    tmp = PANEL + ".tmp"
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, PANEL)
    logger.info("完成: 面板已写回, 耗时 %.1f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
