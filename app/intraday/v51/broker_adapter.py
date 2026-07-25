"""
券商适配层 (V5.1 P20.1/P20.3, 检查清单 #17)
==================================================
券商通道决策 (P20.3): 渤海证券 API 能力未确认 → 不等待;
SimulationAdapter 先行, 半自动模式可长期运行
(攻击档每日仅 0-2 笔交易, 人工下单完全可承受).

接口统一: BrokerAdapter 抽象 → SimulationAdapter (Shadow) /
MiniQMTAdapter (国金, 10万门槛, 阶段四) / 人工 (semi_auto).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 下单执行纪律 (V5.1 §3): ±1 档限价挂单, 15 秒未成交撤单转市价
CANCEL_AFTER_SECONDS = 15


@dataclass(frozen=True)
class Order:
    """统一订单结构."""

    symbol: str
    side: str  # "buy" / "sell"
    qty: int
    limit_price: float
    rule: str = ""  # 触发规则 (B1/B2/S1...) 留痕


@dataclass(frozen=True)
class Fill:
    """统一成交回报."""

    order: Order
    filled_qty: int
    filled_price: float
    slippage: float  # 实际滑点 (filled vs limit)
    auction: bool = False  # S8 集合竞价排队成交


class BrokerAdapter(ABC):
    """券商适配器统一接口 (回测/模拟/实盘同一份决策代码, 铁律 #1)."""

    @abstractmethod
    def place_order(self, order: Order) -> Fill:
        """下单 (±1 档限价; 15 秒未成交撤单转市价)."""

    @abstractmethod
    def place_auction_order(self, order: Order) -> Fill:
        """S8 跌停逃生: 次日 09:25 集合竞价挂跌停价排队 (优先于一切)."""

    @abstractmethod
    def positions(self) -> dict:
        """当前持仓查询."""


class SimulationAdapter(BrokerAdapter):
    """模拟执行器 (Shadow 模式): 按限价立即成交, 记录滑点=0;

    流动性约束: 单笔 ≤ ADV20×1% 超出部分拒单 (与 B6 同口径).
    """

    def __init__(self, adv_20d_map: dict[str, float] | None = None):
        self.adv_20d_map = adv_20d_map or {}
        self._positions: dict[str, int] = {}
        self.fills: list[Fill] = []

    def place_order(self, order: Order) -> Fill:
        adv = self.adv_20d_map.get(order.symbol)
        if adv is not None:
            max_value = adv * 0.01
            if order.qty * order.limit_price > max_value:
                raise ValueError(
                    f"拒单: 单笔金额超 ADV20×1% ({order.symbol}, B6 同口径)")
        fill = Fill(order=order, filled_qty=order.qty,
                    filled_price=order.limit_price, slippage=0.0)
        self._apply(fill)
        self.fills.append(fill)
        logger.info("模拟成交: %s %s %d@%.2f (%s)", order.side, order.symbol,
                    order.qty, order.limit_price, order.rule)
        return fill

    def place_auction_order(self, order: Order) -> Fill:
        fill = Fill(order=order, filled_qty=order.qty,
                    filled_price=order.limit_price, slippage=0.0, auction=True)
        self._apply(fill)
        self.fills.append(fill)
        logger.warning("S8 集合竞价排队成交 (模拟): %s %d@%.2f",
                       order.symbol, order.qty, order.limit_price)
        return fill

    def _apply(self, fill: Fill) -> None:
        sign = 1 if fill.order.side == "buy" else -1
        self._positions[fill.order.symbol] = (
            self._positions.get(fill.order.symbol, 0) + sign * fill.filled_qty)

    def positions(self) -> dict:
        return dict(self._positions)


class MiniQMTAdapter(BrokerAdapter):
    """MiniQMT 适配器 (国金, 10 万门槛, 阶段四全自动).

    xtquant 仅在本适配层 import (决策代码不依赖券商 SDK).
    未连接实盘前以 dry_run 模式运行 (记录意图, 不下单).
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._intents: list[Order] = []
        if not dry_run:
            raise NotImplementedError(
                "MiniQMT 实盘通道待阶段四接入 (xtquant); dry_run=False 暂不可用")

    def place_order(self, order: Order) -> Fill:
        self._intents.append(order)
        logger.warning("MiniQMT dry_run: 记录下单意图 %s %s %d@%.2f",
                       order.side, order.symbol, order.qty, order.limit_price)
        return Fill(order=order, filled_qty=0, filled_price=0.0, slippage=0.0)

    def place_auction_order(self, order: Order) -> Fill:
        self._intents.append(order)
        logger.warning("MiniQMT dry_run: S8 竞价排队意图 %s %d@%.2f",
                       order.symbol, order.qty, order.limit_price)
        return Fill(order=order, filled_qty=0, filled_price=0.0,
                    slippage=0.0, auction=True)

    def positions(self) -> dict:
        return {}
