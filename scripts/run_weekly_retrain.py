"""周频重训入口 (每周第一个交易日 15:30 启动, 与 16:00 清单生成解耦)
=====================================================================
重训频率: 每周一次 (用户 2026-07-22 裁决: 周频, 非月频; 2026-07-26 重申).

用法: python scripts/run_weekly_retrain.py [--symbols-file F] [--tag 2026W30]
流程: 装配双板块 3 年训练面板 (akshare) → train_runner.run_training →
      OOS IC >= 0.03 才切换 current 模型包, 否则保留旧模型 + 告警.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.panel_builder import assemble_panel  # noqa: E402
from app.pipeline1.train_runner import run_training  # noqa: E402
from scripts.train_pipeline1 import load_symbols  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("run_weekly_retrain")

MODEL_DIR = "models/pipeline1"


def main() -> dict:
    parser = argparse.ArgumentParser(description="Pipeline-1 周频重训")
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", default=None)
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    symbols = load_symbols(args)
    tag = args.tag or time.strftime("%GW%V")  # ISO 周标签
    supply = DataSupplyChain()
    panel = assemble_panel(supply, symbols, years=args.years)
    results = run_training(panel, tag, model_dir=MODEL_DIR)
    for board, res in results.items():
        if not res["switched"]:
            logger.warning("[%s] OOS 未达标, 保留旧模型 (path=%s)", board, res["path"])
    return results


if __name__ == "__main__":
    main()
