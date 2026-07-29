#!/usr/bin/env python3
"""从 Tushare API 拉取 cyq_perf 筹码分布数据并填充到 v3."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import logging  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cyq_fetch")

V3_PATH = "data/panel_full_enriched_v3.parquet"


def main():
    from app.pipeline1.data_supply import DataSupplyChain

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("拉取 cyq_tushare 筹码分布数据并填充 v3")
    logger.info("=" * 60)

    # 1. 加载 v3, 获取股票列表和日期范围
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    symbols = sorted(v3["symbol"].unique().tolist())
    start_date = v3["date"].min().strftime("%Y%m%d")
    end_date = v3["date"].max().strftime("%Y%m%d")
    logger.info(
        "v3: %d 行, %d 股, %s ~ %s", len(v3), len(symbols), start_date, end_date
    )

    # 2. 拉取 cyq_perf 数据
    logger.info("初始化 DataSupplyChain...")
    chain = DataSupplyChain()

    logger.info("开始批量拉取 cyq_perf (预计 ~16 分钟)...")
    try:
        cyq = chain.fetch_chip_distribution_batch(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            throttle=0.3,
        )
    except Exception as exc:
        logger.error("拉取失败: %s", exc)
        return

    logger.info("cyq_perf 拉取完成: %d 行 %d 列", len(cyq), len(cyq.columns))
    logger.info("cyq 列: %s", cyq.columns.tolist())
    logger.info("cyq 日期范围: %s ~ %s", cyq["date"].min(), cyq["date"].max())

    # 3. 列名映射: Tushare -> v3
    rename_map = {
        "winner_rate": "benefit_part",  # 获利比例
        "weight_avg": "avg_cost",  # 加权平均成本 -> 平均成本
    }
    cyq = cyq.rename(columns=rename_map)
    logger.info("列名映射: %s", rename_map)

    # 4. 删除 v3 中的旧 cyq 列 (避免 _x/_y 冲突)
    cyq_cols_in_v3 = [
        "benefit_part",
        "avg_cost",
        "pct_70_low",
        "pct_70_high",
        "pct_70_con",
        "pct_90_low",
        "pct_90_high",
        "pct_90_con",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "weight_avg",
    ]
    # 只删除 cyq 中实际有的列
    cols_to_drop = [c for c in cyq_cols_in_v3 if c in v3.columns and c in cyq.columns]
    if cols_to_drop:
        v3 = v3.drop(columns=cols_to_drop)
        logger.info("删除 %d 列旧 cyq 数据", len(cols_to_drop))

    # 5. Merge
    merge_cols = ["symbol", "date"]
    data_cols = [c for c in cyq.columns if c not in merge_cols]
    v3 = v3.merge(cyq[merge_cols + data_cols], on=merge_cols, how="left")
    logger.info("merge 完成: %d 列", len(data_cols))

    # 6. 保存
    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %s (%d 行 %d 列)", V3_PATH, len(v3), len(v3.columns))
    logger.info(
        "总耗时: %.1f 秒 (%.1f 分钟)", time.time() - t0, (time.time() - t0) / 60
    )


if __name__ == "__main__":
    main()
