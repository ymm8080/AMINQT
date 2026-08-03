#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回填 V3 面板 up_limit_raw/down_limit_raw (2023-2025 + 2026 缺口).
审计发现: up/down_limit_raw 仅 2026 有 97% 覆盖, 2023/2024 ≈0%, 2025 27%.
按 panel 中 up_limit_raw 为 NaN 的交易日, 复用 DataSupplyChain.fetch_stk_limit
(per-date, 走缓存) 拉取, 只回填 NaN 单元格, 不动已有值.

WORM: 改写前日期后缀备份, os.replace 原子写. 严禁与其他面板写脚本并发.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("refill_stk_limit")

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import _parallel_fetch  # noqa: E402


def main():
    t0 = time.time()
    if not os.path.exists(PANEL):
        logger.error("面板不存在: %s", PANEL)
        return
    logger.info("加载面板: %s", PANEL)
    panel = pd.read_parquet(PANEL)
    logger.info(
        "面板: %d 行 %d 列 %d 股, %s ~ %s",
        len(panel), len(panel.columns), panel["symbol"].nunique(),
        panel["date"].min(), panel["date"].max(),
    )

    if "up_limit_raw" not in panel.columns:
        logger.error("面板无 up_limit_raw 列")
        return

    missing_mask = panel["up_limit_raw"].isna()
    n_missing = int(missing_mask.sum())
    dates = panel.loc[missing_mask, "date"].drop_duplicates().sort_values()
    logger.info("需回填 %d 行, 涉及 %d 个交易日", n_missing, len(dates))
    if n_missing == 0:
        logger.info("up_limit_raw 已全覆盖, 跳过")
        return

    supply = DataSupplyChain()

    def _fetch_one(d):
        ds = d.strftime("%Y%m%d")
        df_one = supply.fetch_stk_limit(trade_date=ds, refresh=False)
        return df_one if len(df_one) else None

    logger.info("拉取 stk_limit (per-date, %d 日)...", len(dates))
    frames = _parallel_fetch(
        _fetch_one,
        dates.tolist(),
        desc="stk_limit_refill",
    )
    frames = [f for f in frames if f is not None]
    if not frames:
        logger.error("全部拉取失败, 中止 (不写盘)")
        return
    fetch_df = pd.concat(frames, ignore_index=True)
    fetch_df = fetch_df.drop_duplicates(subset=["symbol", "date"])
    logger.info(
        "拉取完成: %d 行, %d 个交易日, %d 股",
        len(fetch_df), fetch_df["date"].nunique(), fetch_df["symbol"].nunique(),
    )

    # 只回填 NaN 单元格: 先记录命中率, 匹配率过低于中止
    merged = panel[["symbol", "date"]].merge(
        fetch_df[["symbol", "date", "up_limit_raw", "down_limit_raw"]],
        on=["symbol", "date"], how="left",
    )
    hit = merged.loc[missing_mask, "up_limit_raw"].notna()
    hit_rate = hit.mean() if len(hit) else 0.0
    logger.info("NaN 单元格匹配率: %.1f%%", hit_rate * 100)
    if hit_rate < 0.95:
        logger.error("匹配率 <95%%, 中止 (防陈旧缓存静默覆盖)")
        return

    panel.loc[missing_mask, "up_limit_raw"] = merged.loc[missing_mask, "up_limit_raw"]
    panel.loc[missing_mask, "down_limit_raw"] = merged.loc[missing_mask, "down_limit_raw"]

    after_missing = int(panel["up_limit_raw"].isna().sum())
    logger.info("回填后 up_limit_raw 剩余 NaN: %d (覆盖率 %.2f%%)",
                after_missing, (len(panel) - after_missing) / len(panel) * 100)

    backup = PANEL.replace(".parquet", "_prestklimit_{}.parquet".format(
        pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")))
    shutil.copy2(PANEL, backup)
    logger.info("备份: %s", backup)
    tmp = PANEL + ".tmp"
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, PANEL)
    logger.info("完成: 面板已写回, 耗时 %.1f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
