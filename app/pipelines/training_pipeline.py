# -*- coding: utf-8 -*-
"""Pipeline 1: 选股预测 (P6, ARCH §3.2, DESIGN_V1 §4).

每日 16:00 后执行, 三步选股:
  第一步 BaseLiquidityFilter: 全市场 → 基础池
  第二步 IwencaiAgent.build_candidate_pool: 基础池 → 候选池
  第三步 模型仅对候选池打分 → {prob_up, pct_up} 按个股上涨概率降序
产出: 推荐股票池 (StockPoolManager) + 自选标记 (WatchlistMarker)。
"""

import logging
from typing import List

from app.core.universe_manager import Universe, UniverseManager

logger = logging.getLogger(__name__)


class SelectionPipeline:
    """Pipeline 1: 三步选股 → 股票池生成 → 标记.

    执行时机: 每日 16:00 后 (APScheduler) 或手动触发。
    """

    def __init__(self, universe: Universe = Universe.ALL) -> None:
        self.universe = universe
        self.universe_mgr = UniverseManager()
        # P6 接线: BaseLiquidityFilter / IwencaiAgent / StockPoolManager /
        # WatchlistMarker / MarketContext / SectorContext

    def run(self, top_n: int = 20) -> List[dict]:
        """完整三步选股流程.

        Returns:
            [{symbol, name, prob_up, pct_up, score, is_watched, ...}]
            按 prob_up 降序 Top-N。
        """
        raise NotImplementedError("P6 待建")

    def predict_batch(self, symbols: List[str]) -> List[dict]:
        """批量预测: 候选池 → 因子 (85 维) → ONNX 推理 → {prob_up, pct_up}."""
        raise NotImplementedError("P6 待建")

    def get_pool_with_watchlist(self, top_n: int = 20):
        """返回股票池 DataFrame, 含 prob_up/pct_up/is_watched 列."""
        raise NotImplementedError("P6 待建")
