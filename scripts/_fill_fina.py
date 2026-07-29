#!/usr/bin/env python3
"""填充 fina_indicator: 季度数据, 用 announce_date forward-fill 到交易日."""

import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fina_fill")

V3_PATH = "data/panel_full_enriched_v3.parquet"


def main():
    t0 = time.time()
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    logger.info("v3: %d 行 %d 列", len(v3), len(v3.columns))

    # 加载 fina_indicator 缓存
    fina_dir = "data/supply_cache/alt_data/fina_indicator"
    if not os.path.isdir(fina_dir):
        logger.error("fina_indicator 缓存不存在")
        return

    files = sorted(f for f in os.listdir(fina_dir) if f.endswith(".parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(fina_dir, f))
            if len(df) and "symbol" in df.columns:
                frames.append(df)
        except:
            pass
    if not frames:
        logger.error("无 fina_indicator 数据")
        return

    fina = pd.concat(frames, ignore_index=True)
    # 去重: 同一 symbol + announce_date 保留最新
    fina = fina.drop_duplicates(subset=["symbol", "announce_date"], keep="last")
    logger.info(
        "fina_indicator: %d 行 %d 列, %d 股",
        len(fina),
        len(fina.columns),
        fina["symbol"].nunique(),
    )

    # 用 announce_date 作为 date, 然后 forward-fill
    fina["date"] = pd.to_datetime(fina["announce_date"])
    fina = fina.dropna(subset=["date"])
    fina = fina.sort_values(["symbol", "date"])

    # 只保留需要的列
    skip = {"symbol", "_ts_code", "announce_date", "report_period", "date"}
    data_cols = [c for c in fina.columns if c not in skip]
    logger.info("数据列: %s", data_cols)

    # 删除 v3 中已有的同名列 (避免冲突)
    old = [c for c in data_cols if c in v3.columns]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("删除 %d 列旧数据", len(old))

    # merge_asof: 对每只股票, 找到 <= 交易日的最近一条财报
    fina_clean = fina[["symbol", "date"] + data_cols].drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    # merge_asof 要求 on 列全局排序
    v3_sorted = v3.sort_values("date").reset_index(drop=True)
    fina_sorted = fina_clean.sort_values("date").reset_index(drop=True)
    v3 = pd.merge_asof(
        v3_sorted,
        fina_sorted,
        on="date",
        by="symbol",
        direction="backward",
    )
    logger.info("merge_asof 完成")

    # 保存
    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info(
        "完成: %d 行 %d 列, 耗时 %.1f 秒", len(v3), len(v3.columns), time.time() - t0
    )


if __name__ == "__main__":
    main()
