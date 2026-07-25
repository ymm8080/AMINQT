"""
半自动模式 (V5.1 P20.2, 券商通道决策: 半自动可长期运行)
==============================================================
系统给信号 → 推送 → 人工 APP 下单 → 人工回报成交 → 逐笔核对.
攻击档每日仅 0-2 笔交易, 人工下单完全可承受.

纪律: 信号只读推送 (含规则/限价/数量), 人工动作全部 WORM 留痕 (铁律 #3);
人工偏离信号 (改价/改量/忽略) 必须选择原因代码, 周度复盘.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

from .broker_adapter import Order
from .worm_logger import WormLogger

logger = logging.getLogger(__name__)

# 人工偏离原因代码 (周度复盘分类)
DEVIATE_REASONS = ("price_changed", "qty_changed", "ignored_signal",
                   "manual_override", "executor_error")


@dataclass(frozen=True)
class SignalTicket:
    """推送人工执行的信号单."""

    trade_date: str
    symbol: str
    side: str
    qty: int
    limit_price: float
    rule: str
    note: str = ""


class SemiAutoDesk:
    """半自动工作台: 信号推送 → 人工确认 → 留痕."""

    def __init__(self, worm: WormLogger):
        self.worm = worm
        self._pending: list[SignalTicket] = []

    # ---------------- 信号推送 ----------------
    def push_signal(self, ticket: SignalTicket) -> None:
        """系统信号 → 待人工执行队列 + WORM."""
        self._pending.append(ticket)
        self.worm.log(ticket.trade_date, "signal", asdict(ticket))
        logger.warning("半自动信号待人工执行: %s %s %d@%.2f (%s)",
                       ticket.side, ticket.symbol, ticket.qty,
                       ticket.limit_price, ticket.rule)

    def pending(self) -> list[SignalTicket]:
        return list(self._pending)

    # ---------------- 人工回报 ----------------
    def confirm_fill(
        self, trade_date: str, symbol: str, filled_qty: int,
        filled_price: float, note: str = "",
    ) -> None:
        """人工按信号下单成交 → 回报 + WORM (供逐笔核对)."""
        self._pending = [t for t in self._pending if t.symbol != symbol]
        self.worm.log(trade_date, "order", {
            "symbol": symbol, "filled_qty": filled_qty,
            "filled_price": filled_price, "note": note, "source": "manual"})
        logger.info("人工成交回报: %s %d@%.2f", symbol, filled_qty, filled_price)

    def report_deviation(
        self, trade_date: str, symbol: str, reason: str, detail: str = ""
    ) -> None:
        """人工偏离信号 (改价/改量/忽略) → 原因代码 + WORM (周度复盘)."""
        assert reason in DEVIATE_REASONS, f"原因代码须为 {DEVIATE_REASONS}"
        self._pending = [t for t in self._pending if t.symbol != symbol]
        self.worm.log(trade_date, "manual", {
            "symbol": symbol, "deviation": reason, "detail": detail})
        logger.error("人工偏离信号: %s [%s] %s", symbol, reason, detail)


def ticket_from_order(trade_date: str, order: Order, note: str = "") -> SignalTicket:
    """BrokerAdapter Order → 人工信号单."""
    return SignalTicket(
        trade_date=trade_date, symbol=order.symbol, side=order.side,
        qty=order.qty, limit_price=order.limit_price, rule=order.rule,
        note=note)
