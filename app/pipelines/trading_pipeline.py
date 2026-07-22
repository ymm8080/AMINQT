# -*- coding: utf-8 -*-
"""Pipeline 2: 盘中交易 (P7, ARCH §3.3, DESIGN_V1 §5).

盘中 9:15~15:00 每 2 分钟执行 (因子粒度 5 分钟 K 线):
  五分钟因子 (25 维) + 日线因子注入 (85 维) → 110 维综合向量
  → 双层买入/卖出检测 + 扩展规则 + 渐进式信号推进
  → 风控过滤 → 交易信号 → 执行 (miniQMT 手动确认/自动)。
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class TradingPipeline:
    """Pipeline 2: 盘中择时买卖.

    执行时机: 交易日 9:15~15:00, 每 2 分钟一次 (APScheduler cron)。
    监控范围: Pipeline 1 推荐股票池 (含 TICK 标记)。
    """

    def __init__(self, config: dict = None) -> None:
        self.config = config or {}
        # P7 接线: IntradayFactorEngine / DualLayerBuyDetector /
        # DualLayerSellDetector / ExtendedBuyDetector / ExtendedSellDetector /
        # ProgressiveSignal / RiskFilter / RuleEngine / OrderManager

    def run_cycle(self) -> List[dict]:
        """单次 2 分钟循环.

        Returns:
            本周期触发的交易信号列表 [{symbol, action, price, qty, reason}]。
        """
        raise NotImplementedError("P7 待建")

    def evaluate_buy(self, symbol: str) -> dict:
        """买入评估: 双层买入 + 扩展买入 + 渐进式信号 + 风控."""
        raise NotImplementedError("P7 待建")

    def evaluate_sell(self, symbol: str) -> dict:
        """卖出评估: 双层卖出 + 扩展卖出 + 风控 (T+1/跌停)."""
        raise NotImplementedError("P7 待建")
