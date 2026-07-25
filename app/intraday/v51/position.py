"""
T+1 持仓状态机 (V5.1 §1, 检查清单 #1/#4, 铁律 #0)
======================================================
T+1 是物理约束: 卖出只作用于 sellable_qty > 0;
当日买入的持仓在当日物理锁死 (满仓档: 任何盘中下跌都只能看着,
这是满仓的代价, 买入决策必须因此更保守).
回测必须同样模拟 T+1 与满仓锁死, 否则结果作废.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    """T+1 持仓状态 (V5.1 §1 攻击档强化版)."""

    symbol: str
    total_qty: int  # 总持仓 (含当日买入)
    sellable_qty: int  # 可卖数量 (昨日及以前买入)
    entry_price: float
    entry_date: str
    hold_days: int = 0
    stop_price: float = 0.0  # 【V5.1】PIPELINE1 下发的当日止损价 (S1)
    max_price_since_entry: float = 0.0  # S2 移动止盈锚点

    def __post_init__(self):
        if not self.max_price_since_entry:
            self.max_price_since_entry = self.entry_price

    def can_sell(self) -> bool:
        """T+1 物理校验: 只有昨日及以前买入的部分可卖."""
        return self.sellable_qty > 0

    def on_buy(self, qty: int, price: float, date: str) -> None:
        """当日买入: total_qty 增加, sellable_qty 不变 (当日锁死)."""
        self.total_qty += qty
        self.entry_price = price
        self.entry_date = date
        self.max_price_since_entry = max(self.max_price_since_entry, price)

    def on_sell(self, qty: int) -> int:
        """卖出: 只能卖 sellable_qty. 返回实际卖出数量."""
        actual = min(qty, self.sellable_qty)
        self.sellable_qty -= actual
        self.total_qty -= actual
        return actual

    def on_bar(self, price: float) -> None:
        """每个 Bar 更新止盈锚点."""
        self.max_price_since_entry = max(self.max_price_since_entry, price)

    def settle_overnight(self) -> None:
        """隔夜结算: 全部转为可卖, 持仓天数 +1."""
        self.sellable_qty = self.total_qty
        self.hold_days += 1
