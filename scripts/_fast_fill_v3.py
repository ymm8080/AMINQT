#!/usr/bin/env python3
"""快速填充 v3: 从缓存填充所有数据源."""

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
logger = logging.getLogger("fast_fill")

V3_PATH = "data/panel_full_enriched_v3.parquet"

# 每个数据源的配置: merge_type + 已知列名 (用于删除旧列)
SOURCES = [
    # (name, merge_type, known_cols)
    ("stk_limit", "symbol+date", ["up_limit_raw", "down_limit_raw"]),
    (
        "margin",
        "symbol+date",
        ["margin_balance", "margin_buy_amt", "short_balance", "short_sell_vol"],
    ),
    (
        "daily_basic",
        "symbol+date",
        [
            "turnover_rate_f",
            "volume_ratio_y",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "dv_ratio",
            "dv_ttm_y",
            "total_mv",
            "circ_mv",
            "total_share",
            "float_share",
            "free_share",
        ],
    ),
    ("sector_index", "date_broadcast", ["sw_ret_1d"]),
    ("holdernumber", "symbol+date", ["holder_count"]),
    ("holdertrade", "symbol+date", ["sh_change_vol", "sh_change_amt", "sh_net_sign"]),
    (
        "northbound",
        "date",
        [
            "north_net_buy_sh",
            "north_net_buy_sz",
            "north_buy_amt_sh",
            "north_sell_amt_sh",
            "north_buy_amt_sz",
            "north_sell_amt_sz",
        ],
    ),
    ("lhb", "symbol+date", ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]),
    # fina_indicator 是季度数据, 无 date 列, 暂不处理
]


def load_cache(src):
    """加载数据源的所有缓存文件并合并."""
    d = f"data/supply_cache/alt_data/{src}"
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if not files:
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(d, f))
            if len(df):
                frames.append(df)
        except Exception as e:
            logger.debug("  %s %s 跳过: %s", src, f, e)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    if "symbol" in out.columns and "date" in out.columns:
        out = out.drop_duplicates(subset=["symbol", "date"])
    elif "date" in out.columns:
        out = out.drop_duplicates(subset=["date"])
    logger.info(
        "  %s: %d 行 %d 列, %s ~ %s",
        src,
        len(out),
        len(out.columns),
        out["date"].min() if "date" in out.columns else "?",
        out["date"].max() if "date" in out.columns else "?",
    )
    return out


def merge_source(v3, src_name, merge_type, known_cols, cache_df):
    """将缓存数据 merge 到 v3 面板."""
    # 1. 删除旧列
    old = [c for c in v3.columns if c in known_cols]
    if old:
        v3 = v3.drop(columns=old)
        logger.info("  删除 %d 列旧数据", len(old))

    # 2. 确定要 merge 的列
    if merge_type == "date_broadcast":
        # sector_index 特殊处理: 按日期广播指数回报
        if "ret_pct" in cache_df.columns and "date" in cache_df.columns:
            sw = cache_df.groupby("date")["ret_pct"].mean().round(6).reset_index()
            sw.columns = ["date", "sw_ret_1d"]
            v3 = v3.merge(sw, on="date", how="left")
            logger.info("  merge 1 列 (date_broadcast)")
        return v3

    # 通用 merge
    key_cols = ["symbol", "date"] if merge_type == "symbol+date" else ["date"]
    if not all(k in cache_df.columns for k in key_cols):
        logger.info("  缺少 key 列 %s, 跳过", key_cols)
        return v3

    # 排除非数据列
    skip_cols = {
        "symbol",
        "date",
        "_ts_code",
        "announce_date",
        "report_period",
        "index_code",
        "index_name",
        "sh_holder_name",
        "sh_change_type",
    }
    data_cols = [c for c in cache_df.columns if c not in skip_cols]
    if not data_cols:
        logger.info("  无数据列")
        return v3

    merge_cols = key_cols + data_cols
    v3 = v3.merge(cache_df[merge_cols], on=key_cols, how="left")
    logger.info("  merge %d 列 (%s)", len(data_cols), "+".join(key_cols))
    return v3


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("快速填充 v3: 从缓存填充所有数据源")
    logger.info("=" * 60)

    if not os.path.exists(V3_PATH):
        logger.error("v3 不存在!")
        return
    logger.info("加载 v3...")
    v3 = pd.read_parquet(V3_PATH)
    logger.info(
        "v3: %d 行 %d 列 %d 股, %s ~ %s",
        len(v3),
        len(v3.columns),
        v3["symbol"].nunique(),
        v3["date"].min(),
        v3["date"].max(),
    )

    for src_name, merge_type, known_cols in SOURCES:
        logger.info("--- %s ---", src_name)
        cache_df = load_cache(src_name)
        if cache_df is None or len(cache_df) == 0:
            logger.info("  无缓存, 跳过")
            continue
        v3 = merge_source(v3, src_name, merge_type, known_cols, cache_df)

    logger.info("保存 v3...")
    v3.to_parquet(V3_PATH, index=False)
    logger.info("完成: %s (%d 行 %d 列)", V3_PATH, len(v3), len(v3.columns))
    logger.info("耗时: %.1f 秒", time.time() - t0)


if __name__ == "__main__":
    main()
