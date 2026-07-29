# -*- coding: utf-8 -*-
"""一次性脚本: 将 holdertrade (股东增减持) 数据 enrich 到 panel_full_enriched.parquet"""

from __future__ import annotations

import logging
import sys
import time

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("enrich_holdertrade")

sys.path.insert(0, ".")

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import enrich_alt_data  # noqa: E402


def main():
    panel_path = "data/panel_full_enriched.parquet"

    # 1. 加载已有面板
    logger.info("加载面板: %s", panel_path)
    panel = pd.read_parquet(panel_path)
    logger.info(
        "面板: %s 行, %s symbols, 日期 %s ~ %s",
        panel.shape[0],
        panel["symbol"].nunique(),
        panel["date"].min().strftime("%Y%m%d"),
        panel["date"].max().strftime("%Y%m%d"),
    )

    # 检查是否已有 holdertrade 列
    existing_ht_cols = [c for c in panel.columns if c.startswith("sh_")]
    if existing_ht_cols:
        logger.info("面板已有 holdertrade 列: %s, 将覆盖更新", existing_ht_cols)

    # 2. 日期范围
    start_date = panel["date"].min().strftime("%Y%m%d")
    end_date = panel["date"].max().strftime("%Y%m%d")

    # 3. 数据供应链
    supply = DataSupplyChain()

    # 4. 执行 enrich (holdertrade only)
    logger.info("开始 enrich holdertrade (%s ~ %s) ...", start_date, end_date)
    t0 = time.time()

    panel = enrich_alt_data(
        panel,
        supply,
        sources=["holdertrade"],
        start_date=start_date,
        end_date=end_date,
        refresh=False,
    )

    elapsed = time.time() - t0
    logger.info("enrich 完成, 耗时 %.1f 秒 (%.1f 分钟)", elapsed, elapsed / 60)

    # 5. 验证
    new_ht_cols = [c for c in panel.columns if c.startswith("sh_")]
    logger.info("新增 holdertrade 列: %s", new_ht_cols)

    for col in ["sh_net_change_sign", "sh_change_amt_total"]:
        if col in panel.columns:
            nz = (panel[col].notna() & (panel[col] != 0)).sum()
            total = len(panel)
            logger.info("  %s: 非零 %d / %d (%.1f%%)", col, nz, total, 100 * nz / total)
        else:
            logger.error("  %s: 列缺失!", col)

    # 6. 保存
    logger.info("保存: %s", panel_path)
    panel.to_parquet(panel_path, index=False)
    logger.info("完成! 最终列数: %d, 总行数: %d", len(panel.columns), len(panel))


if __name__ == "__main__":
    main()
