"""
资金管理 (V5.1 §6, 日内侧本地强制纪律, 检查清单 #16)
============================================================
本系统不再自行决定仓位 (每日 21:30 从 PIPELINE1 清单读取 position_weight /
stop_price / 数量闸门前2). 日内侧保留的资金纪律 (本地强制, 不依赖 PIPELINE1):
  - 单票每日买入次数 ≤ 1 次 (防追涨杀跌)
  - 全局每日新开仓 ≤ 2 笔 (对齐数量闸门)
  - 止损冷却期: 该票当日禁买 (防反复打脸)
  - 日内保险丝: 当日总亏损 ≥ 4% (C档; B档 3%) → 暂停新买入
  - 系统停机线: 总资金自峰值回撤 ≥ 15% → 全系统暂停, 清仓可卖持仓, 人工介入
与 P21.5.1 D.3 三条硬规则联动; E.2 激活后 daily_fuse = position × stop (自动).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .safe_div import safe_divide

logger = logging.getLogger(__name__)

MAX_BUY_PER_SYMBOL_PER_DAY = 1  # 单票每日最多买 1 次, 绝不补仓
MAX_NEW_POSITIONS_PER_DAY = 2  # 全局每日新开仓 ≤ 2 笔 (数量闸门)
DAILY_FUSE_C = 0.04  # 日内保险丝 C档 (100%×-4%, 一笔止损即当日收工)
DAILY_FUSE_B = 0.03  # B档 (75%×4%)
HALT_DRAWDOWN = 0.15  # 停机线 15%


@dataclass
class FundManager:
    """日内资金纪律状态 (每日 reset, 停机线跨日保持)."""

    daily_fuse: float = DAILY_FUSE_C
    _bought_today: set[str] = field(default_factory=set)
    _new_positions_today: int = 0
    _cooldown: set[str] = field(default_factory=set)  # 止损冷却 (当日禁买)
    _daily_pnl: float = 0.0
    _fuse_triggered: bool = False
    peak_nav: float = 0.0
    halted: bool = False  # 停机线: 全系统暂停, 人工介入

    # ---------------- 买入闸门 ----------------
    def can_buy(self, symbol: str, position_weight: float) -> tuple[bool, str]:
        """本地资金纪律链 (任一不过 → 拒绝并给出原因)."""
        if self.halted:
            return False, "停机线触发: 全系统暂停, 人工介入"
        if self._fuse_triggered:
            return False, "日内保险丝: 当日亏损达上限, 暂停新买入"
        if position_weight <= 0:
            return False, "position_weight=0 (清单未入选)"
        if symbol in self._cooldown:
            return False, "止损冷却期: 该票当日禁买"
        if symbol in self._bought_today:
            return False, "当日该票已买过: 只买一次, 绝不补仓"
        if self._new_positions_today >= MAX_NEW_POSITIONS_PER_DAY:
            return False, "全局每日新开仓 ≤ 2 笔 (数量闸门)"
        return True, ""

    def on_buy(self, symbol: str) -> None:
        self._bought_today.add(symbol)
        self._new_positions_today += 1

    def on_stop_loss(self, symbol: str) -> None:
        """止损成交 → 该票当日冷却."""
        self._cooldown.add(symbol)

    # ---------------- 保险丝 / 停机线 ----------------
    def on_daily_pnl(self, pnl_pct: float) -> None:
        self._daily_pnl += pnl_pct
        if self._daily_pnl <= -self.daily_fuse and not self._fuse_triggered:
            self._fuse_triggered = True
            logger.error(
                "日内保险丝: 当日亏损 %.1f%% ≥ %.0f%%, 暂停新买入",
                -self._daily_pnl * 100,
                self.daily_fuse * 100,
            )

    def on_nav(self, nav: float) -> None:
        self.peak_nav = max(self.peak_nav, nav)
        if self.peak_nav > 0 and safe_divide(nav, self.peak_nav) - 1 <= -HALT_DRAWDOWN:
            if not self.halted:
                self.halted = True
                logger.critical(
                    "系统停机线: 回撤 ≥15%%, 全系统暂停, 清仓所有可卖持仓, 人工介入"
                )

    def new_day(self) -> None:
        """每日开盘前 reset (停机线 halted 跨日保持; 保险丝按日重置)."""
        self._bought_today.clear()
        self._new_positions_today = 0
        self._cooldown.clear()
        self._daily_pnl = 0.0
        self._fuse_triggered = False
