# -*- coding: utf-8 -*-
"""Pipeline-1 预测编排 (每日清单主链路)
=====================================================
加载当前模型包 → DailySelectionPipeline.run (清洗0→4+5 → 特征 → 推理 →
校准 → Holding Bonus → 空仓触发 → 清单 schema V1.2 → 持久化).

模型包定位: 按 board 在 model_dir 中取最新 {board}_*.pkl (文件名字典序),
或显式指定 tag.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from .daily_pipeline import DailySelectionPipeline
from .data_supply import DataSupplyChain
from .list_generator import MarketEnv

logger = logging.getLogger(__name__)


def find_bundles(
    model_dir: str = "models/pipeline1", tag: str | None = None
) -> dict[str, str]:
    """定位双板块模型包: {'main': path, 'dual': path}.

    tag 给定 → {board}_{tag}.pkl; 否则取该 board 最新 (字典序最大) 的包.
    缺失的板块不出现在返回值中 (调用方决定降级或报错).
    """
    bundles: dict[str, str] = {}
    if not os.path.isdir(model_dir):
        logger.error("模型目录不存在: %s", model_dir)
        return bundles
    for board in ("main", "dual"):
        if tag:
            path = os.path.join(model_dir, f"{board}_{tag}.pkl")
            if os.path.exists(path):
                bundles[board] = path
            else:
                logger.error("模型包缺失: %s", path)
            continue
        candidates = sorted(
            f
            for f in os.listdir(model_dir)
            if f.startswith(f"{board}_") and f.endswith(".pkl")
        )
        if candidates:
            bundles[board] = os.path.join(model_dir, candidates[-1])
        else:
            logger.warning("[%s] 无可用模型包 (%s)", board, model_dir)
    return bundles


def run_prediction(
    panel: pd.DataFrame,
    trade_date: str,
    bundle_paths: dict[str, str],
    list_dir: str = "data/lists",
    env: MarketEnv | None = None,
    market_state: str = "range",
    float_shares_map: dict | None = None,
    supply: DataSupplyChain | None = None,
    cleaner=None,
) -> dict:
    """每日清单主入口: 面板 + 模型包 → 清单 (持久化 parquet).

    Args:
        panel: enrich 后面板 (含当日 bar; panel_builder.assemble_panel 输出)
        trade_date: 'YYYYMMDD'
        bundle_paths: find_bundles 输出; 必须至少含一个板块
        cleaner: 可选自定义 CleaningPipeline (小样本 universe 需放宽流动性安全阀)

    Returns:
        DailySelectionPipeline.run 结果 {'mode', 'list', 'cap_position', ...}
    """
    if not bundle_paths:
        raise RuntimeError("无可用模型包, 请先运行训练 (scripts/train_pipeline1.py)")
    pipe = DailySelectionPipeline(
        supply=supply or DataSupplyChain(),
        bundle_paths=bundle_paths,
        list_dir=list_dir,
        float_shares_map=float_shares_map,
    )
    if cleaner is not None:
        pipe.cleaner = cleaner
    result = pipe.run(trade_date, panel=panel, env=env, market_state=market_state)
    n = 0 if result.get("empty") else len(result.get("list", []))
    logger.info(
        "清单生成完成 (%s): mode=%s, %d 只, cap=%.2f",
        trade_date,
        result.get("mode"),
        n,
        result.get("cap_position", 0.0),
    )
    return result
