# -*- coding: utf-8 -*-
"""Pipeline-1 每日预测入口 (清单生成)
=====================================================
用法:
    python scripts/predict_pipeline1.py --symbols-file config/universe.txt
    python scripts/predict_pipeline1.py --symbols 600519 601318 --trade-date 20260724

流程: 装配面板 (最近 3 年 akshare, 含当日) → 加载最新模型包 →
DailySelectionPipeline.run → 清单 parquet 落盘 data/lists/list_<date>.parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import assemble_panel  # noqa: E402
from app.pipeline1.predict_runner import find_bundles, run_prediction  # noqa: E402
from scripts.train_pipeline1 import load_symbols  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("predict_pipeline1")


def main() -> dict:
    parser = argparse.ArgumentParser(description="Pipeline-1 每日清单预测")
    parser.add_argument("--symbols", nargs="*", default=None, help="股票代码列表")
    parser.add_argument("--symbols-file", default=None, help="universe 文件 (每行一代码)")
    parser.add_argument("--trade-date", default=None, help="YYYYMMDD (默认今天)")
    parser.add_argument("--years", type=int, default=3, help="回看年数 (默认 3)")
    parser.add_argument("--tag", default=None, help="模型标签 (默认每板块最新)")
    parser.add_argument("--model-dir", default="models/pipeline1")
    parser.add_argument("--list-dir", default="data/lists")
    parser.add_argument("--market-state", default="range", choices=["bull", "bear", "range"])
    parser.add_argument("--refresh", action="store_true", help="忽略缓存强制重拉")
    args = parser.parse_args()

    trade_date = args.trade_date or datetime.now().strftime("%Y%m%d")
    symbols = load_symbols(args)
    bundles = find_bundles(args.model_dir, tag=args.tag)
    logger.info("trade_date=%s, bundles=%s", trade_date, bundles)

    supply = DataSupplyChain()
    end = f"{trade_date[:4]}-{trade_date[4:6]}-{trade_date[6:]}"
    panel = assemble_panel(
        supply, symbols, end=end, years=args.years, refresh=args.refresh
    )
    result = run_prediction(
        panel,
        trade_date,
        bundles,
        list_dir=args.list_dir,
        market_state=args.market_state,
        supply=supply,
    )
    lst = result.get("list")
    if lst is not None and len(lst):
        cols = ["symbol", "board", "pred_ret_1d", "prob_up", "score", "weight"]
        print(lst[[c for c in cols if c in lst.columns]].to_string(index=False))
    else:
        print(f"空清单 (mode={result.get('mode')})")
    return result


if __name__ == "__main__":
    main()
