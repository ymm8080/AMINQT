#!/usr/bin/env python3
"""将已缓存的 cyq 数据 merge 到 v3."""
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import logging  # noqa: E402
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge")

V3_PATH = "data/panel_full_enriched_v3.parquet"
CACHE_PATH = "data/supply_cache/alt_data/cyq_tushare/cyq_full.parquet"

def main():
    t0 = time.time()
    logger.info("加载 cyq 缓存...")
    cyq = pd.read_parquet(CACHE_PATH)
    logger.info("cyq: %d 行, %d 股, %s ~ %s",
                len(cyq), cyq["symbol"].nunique(),
                cyq["date"].min(), cyq["date"].max())

    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    logger.info("v3: %d 行, %d 列", len(v3), len(v3.columns))

    # 删除旧 cyq 列
    cyq_cols = ["benefit_part", "avg_cost", "cost_5pct", "cost_15pct",
                "cost_50pct", "cost_85pct", "cost_95pct"]
    old = [c for c in cyq_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("删除 %d 列旧 cyq 数据", len(old))

    # Merge
    data_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
    v3 = v3.merge(cyq[["symbol", "date"] + data_cols], on=["symbol", "date"], how="left")
    logger.info("merge 完成: %d 列", len(data_cols))

    # 覆盖率
    for c in data_cols:
        nn = v3[c].notna().sum()
        logger.info("  %s: %d/%d (%.1f%%)", c, nn, len(v3), nn / len(v3) * 100)

    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %s (%d 行 %d 列), 耗时 %.1f 秒", V3_PATH, len(v3), len(v3.columns), time.time() - t0)

if __name__ == "__main__":
    main()