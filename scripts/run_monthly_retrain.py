# -*- coding: utf-8 -*-
"""
月度重训入口 (每月第一个交易日 15:30 启动, 与 16:00 清单生成解耦)
=====================================================================
用法: python scripts/run_monthly_retrain.py [tag]
流程: 加载双板块 720 日训练面板 → DualTrackTrainer.monthly_retrain →
      OOS IC >= 0.03 才切换 current 模型包, 否则保留旧模型 + 告警.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline1.dual_track_trainer import DualTrackTrainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_monthly_retrain")

MODEL_DIR = "models/pipeline1"


def main(tag: str | None = None) -> dict:
    tag = tag or time.strftime("%Y%m")
    # TODO(生产): 装配 panels = {'main': 主板720日面板, 'dual': 双创720日面板}
    #             + feature_cols_by_board (来自 ICScreener 当期因子清单)
    raise SystemExit(
        "生产重训需先接入全市场历史库 (DataSupplyChain.fetch_history 批量装配). "
        "训练逻辑见 DualTrackTrainer.monthly_retrain, OOS 切换已内建.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
