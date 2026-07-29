#!/usr/bin/env python3
"""修复 v3: 重命名被 merge 冲突的列, 删除无关的 lhb 列."""

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fix")

V3_PATH = "data/panel_full_enriched_v3.parquet"

logger.info("加载 v3...")
df = pd.read_parquet(V3_PATH)
logger.info("修复前: %d 列", len(df.columns))

# 1. 重命名被冲突的列: _x -> 原名
renames = {
    "close_x": "close",
    "amount_x": "amount",
}
# turnover_rate_x 可能是原有的, 检查
if "turnover_rate_x" in df.columns and "turnover_rate" not in df.columns:
    renames["turnover_rate_x"] = "turnover_rate"
# volume_ratio -> 已存在 (无后缀), volume_ratio_x 是旧的
# dv_ttm -> 已存在 (无后缀), dv_ttm_x 是旧的

df = df.rename(columns=renames)
logger.info("重命名: %s", renames)

# 2. 删除 lhb merge 带来的无关列
drop_cols = [
    # _y 后缀 (来自 lhb 缓存)
    "close_y",
    "amount_y",
    "turnover_rate_y",
    # lhb 原始字段
    "trade_date",
    "ts_code",
    "name",
    "pct_change",
    "l_sell",
    "l_buy",
    "l_amount",
    "net_amount",
    "net_rate",
    "amount_rate",
    "float_values",
    "reason",
]

# 删除所有中文列名
for c in df.columns:
    try:
        c.encode("ascii")
    except (UnicodeEncodeError, UnicodeDecodeError):
        drop_cols.append(c)

# 只删除实际存在的列
drop_cols = [c for c in drop_cols if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    logger.info("删除 %d 列: %s", len(drop_cols), drop_cols[:10])

logger.info("保存 v3...")
df.to_parquet(V3_PATH, index=False)
logger.info("修复后: %d 列", len(df.columns))
logger.info("完成!")
