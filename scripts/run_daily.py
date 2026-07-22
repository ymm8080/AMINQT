# -*- coding: utf-8 -*-
"""
每日选股清单生成入口 (16:00, APScheduler 或手工)
=====================================================
用法: python scripts/run_daily.py [YYYYMMDD]
生产: akshare 数据供应链 + models/pipeline1/current_{main,dual}.pkl
失败: 自动走 ListDeliveryGuard 三档降级 (沿用/只卖/人工)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_daily")

BUNDLES = {"main": "models/pipeline1/main_current.pkl",
           "dual": "models/pipeline1/dual_current.pkl"}


def main(trade_date: str | None = None) -> dict:
    trade_date = trade_date or time.strftime("%Y%m%d")
    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    result = pipe.run(trade_date)   # panel=None → 生产装配路径
    lst = result.get("list")
    n = 0 if lst is None else len(lst)
    logger.info("清单生成完成: mode=%s, %d 只, schema=%s",
                result.get("mode"), n, result.get("schema_version", "-"))
    if n:
        print(lst.to_string(index=False))
    return result


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
