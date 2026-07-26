"""
决策链回测引擎 (V5.1 P20.0, 检查清单 #1/#4 验收)
==========================================================
回测必须模拟 (否则结果作废):
  - T+1 满仓锁死 (当日买入当日不可卖, 任何盘中下跌只能看着)
  - 分层滑点 + 冲击成本 (与 PIPELINE1 同源成本模型)
  - 跌停逃生排队 (S8: 次日 09:25 集合竞价挂跌停价, 未成交次日继续挂)
  - 清单失效条件 (跳空/跌停/板块/公告#5)
  - 日内保险丝 + 停机线 (本地强制)

调用方式: 每日每票 5min bars + 当日清单字段 (stop_price/position_weight/
pred_q50/ATR/ADV20) 由调用方装配为 DayContext; 引擎只跑决策链 (纯计算).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .buy_engine import Bar, BuyContext, limit_order_price
from .buy_engine import trigger as buy_trigger
from .cost_model import CostModel, round_trip_cost
from .fund_manager import FundManager
from .position import Position
from .safe_div import safe_divide
from .sell_engine import SellContext
from .sell_engine import trigger as sell_trigger
from .sessions import buy_window_open

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DayContext:
    """单票单日回测输入 (清单字段 + 5min bars + 大盘环境)."""

    symbol: str
    date: str
    bars: tuple[Bar, ...]  # 当日 5min bars (升序)
    pre_close: float
    pred_q50: float
    atr_pct: float
    stop_price: float
    position_weight: float  # 攻击档: 1.0 或 0 (B档 0.75)
    adv_20d: float
    signal_grade: str = "A"
    bear_state: str = "NORMAL"  # 熊市协议接管
    hs300_change: float = 0.0
    sector_drop_count: int = 0
    event_mean: float = 0.0
    limit_up_price: float = 0.0
    limit_down_price: float = 0.0
    turnover_pct: float = 0.0
    invalidation: bool = False  # 清单失效条件
    entry_pos: Position | None = None  # 隔夜持仓 (多日回测由引擎流转)


@dataclass
class DayResult:
    trades: list[dict] = field(default_factory=list)
    end_position: Position | None = None
    veto_log: list[str] = field(default_factory=list)


class IntradayBacktester:
    """单日决策链回放 (满仓档: position_weight × 总资金全买一只)."""

    def __init__(
        self, capital: float, costs: CostModel | None = None, profile: str = "C"
    ):
        self.capital = capital
        self.cash = capital
        self.costs = costs or CostModel()
        self.fm = FundManager(daily_fuse=0.04 if profile == "C" else 0.03)
        self.peak = capital

    # ---------------- 买入 ----------------
    def _try_buy(
        self, dc: DayContext, bar: Bar, pos: Position | None
    ) -> Position | None:
        if pos is not None or dc.invalidation:
            return pos
        if not buy_window_open(bar.t, dc.bear_state, dc.signal_grade, dc.hs300_change):
            return pos
        ok, reason = self.fm.can_buy(dc.symbol, dc.position_weight)
        if not ok:
            return pos
        ctx = BuyContext(
            symbol=dc.symbol,
            t=bar.t,
            price=bar.close,
            pre_close=dc.pre_close,
            pred_q50=dc.pred_q50,
            atr_pct=dc.atr_pct,
            stop_price=dc.stop_price,
            adv_20d=dc.adv_20d,
            order_value=self.cash * dc.position_weight,
            bar_amount=bar.amount,
            sector_drop_count=dc.sector_drop_count,
            event_mean=dc.event_mean,
        )
        r = buy_trigger(ctx, dc.bars[: dc.bars.index(bar) + 1])
        if not r["pass"]:
            return pos
        # 成交: ±1 档限价 + 成本
        px = limit_order_price(bar.close, "buy")
        amount = self.cash * dc.position_weight
        qty = int(safe_divide(amount, px) / 100) * 100
        if qty <= 0:
            return pos
        cost = qty * px * round_trip_cost(dc.adv_20d, qty * px, self.costs) / 2
        self.cash -= qty * px + cost
        self.fm.on_buy(dc.symbol)
        pos = Position(dc.symbol, qty, 0, px, dc.date, stop_price=dc.stop_price)
        logger.info(
            "回测买入: %s %s %d股@%.2f (%s)", dc.date, dc.symbol, qty, px, r["positive"]
        )
        return pos, {
            "date": dc.date,
            "side": "buy",
            "price": px,
            "qty": qty,
            "rule": r["positive"],
        }

    # ---------------- 卖出 ----------------
    def _try_sell(self, dc: DayContext, bar: Bar, pos: Position) -> dict | None:
        ctx = SellContext(
            t=bar.t,
            price=bar.close,
            limit_down_price=dc.limit_down_price,
            limit_up_price=dc.limit_up_price,
            turnover_pct=dc.turnover_pct,
            change_pct=safe_divide(bar.close, dc.pre_close) - 1,
            atr_pct=dc.atr_pct,
            invalidation=dc.invalidation,
        )
        r = sell_trigger(ctx, pos)
        if r["action"] == "HOLD":
            return None
        # S8 跌停: 无法成交 (排队到次日), 其余按 bar 价 - 滑点成交
        if r["action"] == "AUCTION_SELL":
            return {
                "date": dc.date,
                "side": "auction_queue",
                "rule": "S8",
                "qty": r["qty"],
                "price": dc.limit_down_price,
            }
        qty = pos.on_sell(r["qty"])
        proceeds = qty * bar.close
        cost = proceeds * round_trip_cost(dc.adv_20d, proceeds, self.costs) / 2
        self.cash += proceeds - cost
        if r["rule"] == "S1":
            self.fm.on_stop_loss(dc.symbol)
        logger.info(
            "回测卖出: %s %s %d股@%.2f (%s)",
            dc.date,
            dc.symbol,
            qty,
            bar.close,
            r["reason"],
        )
        return {
            "date": dc.date,
            "side": "sell",
            "price": bar.close,
            "qty": qty,
            "rule": r["rule"],
            "reason": r["reason"],
            "pnl": safe_divide(bar.close, pos.entry_price) - 1,
        }

    # ---------------- 单日回放 ----------------
    def run_day(self, dc: DayContext, pos: Position | None = None) -> DayResult:
        """单票单日决策链回放. pos = 隔夜持仓 (T+1 状态由 settle_overnight 流转)."""
        res = DayResult()
        self.fm.new_day()  # 每日纪律重置 (停机线状态跨日保持)
        if pos is not None:
            pos.settle_overnight()  # 隔夜结算: 全部可卖
        start_nav = self.cash + (
            pos.total_qty * dc.bars[0].close if pos and dc.bars else 0
        )
        for bar in dc.bars:
            if pos is not None:
                pos.on_bar(bar.close)
                trade = self._try_sell(dc, bar, pos)
                if trade:
                    res.trades.append(trade)
                    if trade["side"] == "sell" and pos.total_qty == 0:
                        pos = None
                if pos is None:
                    continue
            bought = self._try_buy(dc, bar, pos)
            if isinstance(bought, tuple):
                pos, trade = bought
                res.trades.append(trade)
            elif bought is not None:
                pos = bought
        # 日终净值 → 保险丝/停机线
        nav = self.cash + (pos.total_qty * dc.bars[-1].close if pos else 0)
        if start_nav > 0:
            self.fm.on_daily_pnl(safe_divide(nav, start_nav) - 1)
        self.fm.on_nav(nav)
        res.end_position = pos
        return res
