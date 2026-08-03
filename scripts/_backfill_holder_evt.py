# -*- coding: utf-8 -*-
"""一次性脚本: 给现存 V3 面板回填 sh_evt_start_date / sh_evt_end_date.

Surgical merge — 只并 2 列事件窗口日期, 不复跑 enrich_alt_data:
面板已有 sh_net_change_sign / sh_change_amt_total, 整源重跑会生成 _x/_y 孪生列.
聚合逻辑与 panel_builder.holdertrade 分支完全一致 (min/max 窗口).

WORM: 先备份 panel_full_enriched_v3_preholder_evt_<ts>.parquet 再写回.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("backfill_holder_evt")

sys.path.insert(0, ".")

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

PANEL_PATH = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
EVT_COLS = ["sh_evt_start_date", "sh_evt_end_date"]


def main():
    # 1. 加载面板
    logger.info("加载面板: %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    logger.info(
        "面板: %d 行, %d symbols, 日期 %s ~ %s, 列数 %d",
        panel.shape[0],
        panel["symbol"].nunique(),
        panel["date"].min().strftime("%Y%m%d"),
        panel["date"].max().strftime("%Y%m%d"),
        len(panel.columns),
    )

    # 幂等: 已有 evt 列则先删, 重新 merge
    for c in EVT_COLS:
        if c in panel.columns:
            logger.info("面板已有 %s, 删除后重新回填", c)
            panel = panel.drop(columns=[c])

    # 2. 全量拉取 holdertrade (bulk 分页 + 限流, 带 evt 日期)
    supply = DataSupplyChain()
    start_date = panel["date"].min().strftime("%Y%m%d")
    end_date = panel["date"].max().strftime("%Y%m%d")
    logger.info("拉取 holdertrade (%s ~ %s) ...", start_date, end_date)
    t0 = time.time()
    df = supply.fetch_holdertrade(
        start_date=start_date,
        end_date=end_date,
        refresh=False,
    )
    # 旧缓存可能缺 evt 列或全为 NaT (改字段名前的缓存) → 强制重拉
    if (
        "evt_start_date" not in df.columns
        or "evt_end_date" not in df.columns
        or df["evt_start_date"].isna().all()
    ):
        logger.warning("缓存缺 evt 列, 强制刷新拉取 ...")
        df = supply.fetch_holdertrade(
            start_date=start_date,
            end_date=end_date,
            refresh=True,
        )
    elapsed = time.time() - t0
    logger.info(
        "拉取完成: %d records, %.1f 秒, 含 evt 列: %s / %s",
        len(df),
        elapsed,
        "evt_start_date" in df.columns,
        "evt_end_date" in df.columns,
    )

    # 3. 按公告日聚合事件窗口 (与 panel_builder 一致: start=min, end=max)
    agg_map = {}
    if "evt_start_date" in df.columns:
        agg_map["sh_evt_start_date"] = ("evt_start_date", "min")
    if "evt_end_date" in df.columns:
        agg_map["sh_evt_end_date"] = ("evt_end_date", "max")
    daily = (
        df.groupby(["symbol", "announce_date"]).agg(**agg_map).reset_index()
        if agg_map
        else df[["symbol", "announce_date"]].drop_duplicates()
    )
    daily = daily.rename(columns={"announce_date": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    logger.info(
        "聚合: %d (symbol, date) 事件行, evt_start 非空 %d, evt_end 非空 %d",
        len(daily),
        daily.get("sh_evt_start_date", pd.Series(dtype="datetime64[ns]")).notna().sum(),
        daily.get("sh_evt_end_date", pd.Series(dtype="datetime64[ns]")).notna().sum(),
    )

    # 4. 核对聚合结果与面板已有 sh_net_change_sign 覆盖一致
    sign_cov = panel["sh_net_change_sign"].notna().mean()
    logger.info("面板已有 sh_net_change_sign 覆盖率: %.1f%%", 100 * sign_cov)
    evt_rows = daily["sh_evt_start_date"].notna().mean()
    logger.info("事件行 evt_start 覆盖率: %.1f%%", 100 * evt_rows)

    # 5. WORM 备份
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{os.path.splitext(PANEL_PATH)[0]}_preholder_evt_{ts}.parquet"
    logger.info("备份: %s", backup)
    panel.to_parquet(backup, index=False)

    # 6. 只并 evt 2 列
    panel = panel.merge(
        daily[["symbol", "date"] + EVT_COLS], on=["symbol", "date"], how="left"
    )

    # 7. 写回
    logger.info("写回: %s (列数 %d)", PANEL_PATH, len(panel.columns))
    panel.to_parquet(PANEL_PATH, index=False)

    # 8. 验证
    for c in EVT_COLS:
        cov = panel[c].notna().mean()
        logger.info("  %s 覆盖率: %.1f%%", c, 100 * cov)


if __name__ == "__main__":
    main()
