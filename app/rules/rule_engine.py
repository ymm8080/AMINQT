# -*- coding: utf-8 -*-
"""
规则引擎 v2.0 (合并版 + P11/P12) — 生产组件
================================================
来源: rule_engine_v2.py (DESIGN §15, 实施计划 P15).
L0 合规 / L1 组合闸门 / L2 卖出状态机 P1-P12 / L3 建仓标记 / L4 执行状态机.
确定性裁决层: 不预测、不学习. 指标来源经 IndicatorFeed 注入 (可插拔, 可回测).

v2.8 变更: 并入 v1 退出规则 (用户确认 2026-07-22):
  P11 移动止盈 (持仓期高点回撤 >= trailing_drawdown 且曾盈利), 紧随 P1
  P12 概率衰减 (最新 prob_up < prob_exit), 盘中检查 + 盘后日线标记双保险
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Protocol

from .config import Config, price_limit
from .peak_tracker import PeakTracker

logger = logging.getLogger(__name__)


# ============================================================
# 指标接口: 益盟/主力类指标的可插拔抽象 (生产 = CompositeFeed, 测试 = Mock)
# ============================================================
class IndicatorFeed(Protocol):
    def control_ratio(self, code: str) -> float: ...  # 主力控盘比例 %
    def had_accumulation_peak(self, code: str, lookback: int) -> bool: ...  # 吸筹峰
    def red_above_blue_since_peak(self, code: str) -> bool: ...  # 益盟红蓝线
    def red_blue_distance_min(self, code: str) -> bool: ...  # 红蓝距离N日最小
    def control_weekly_up(self, code: str) -> bool: ...  # 控盘周均线环比
    def bottom_breakout_volume(self, code: str) -> bool: ...  # 底部平台突破放量
    def recent_shadow_lines(self, code: str) -> bool: ...  # 两周内上下影线
    def red_bar_rising_and_majority(self, code: str) -> bool: ...  # 红柱升高且>50%
    def profit_chip_ratio(self, code: str) -> float: ...  # 获利盘筹码 %
    def latest_prob_up(self, code: str) -> float: ...  # P12: 最新 V3.5 清单概率


# ============================================================
# 数据结构
# ============================================================
class Action(str, Enum):
    BUY = "BUY"
    SELL_ALL = "SELL_ALL"
    SELL_HALF = "SELL_HALF"
    HOLD = "HOLD"
    BLOCKED = "BLOCKED"
    WARN = "WARN"


@dataclass
class Tick:
    """2分钟bar"""

    time: str  # "HH:MM"
    price: float
    volume: float
    turnover: float = 0.0  # 当日累计换手率 %
    big_order_net: float = 0.0  # 大单净量


@dataclass
class Candidate:
    code: str
    industry: str
    pred_return: float  # Pipeline1 模型预测涨幅 (schema V1.0 pred_ret_1d)
    pred_prob: float  # 模型预测概率 (schema V1.0 prob_up)
    avg_amount_20d: float = 1e8
    list_days: int = 1000
    is_st: bool = False
    is_suspended: bool = False
    daily_closes: list = field(default_factory=list)
    max_daily_gain_10d: float = 0.0
    turnover_today: float = 0.0
    macd_hist: float = 0.0
    macd_hist_prev: float = 0.0
    flag_watch: bool = False
    flag_daily_buy: bool = False
    flag_daily_sell: bool = False
    flag_intraday_buy: bool = False


class BuyShape(Enum):
    """入场形态状态机"""

    WAIT_OPEN = auto()
    DOWN_WAIT_TROUGH = auto()
    UP_WAIT_PEAK2 = auto()
    UP_WAIT_TROUGH2 = auto()
    READY = auto()
    DONE = auto()


@dataclass
class Position:
    code: str
    industry: str
    cost: float
    weight: float
    buy_day: int
    hold_days: int = 0
    high_since_buy: float = 0.0
    sellable: bool = False  # T+1
    half_sold: bool = False


@dataclass
class Order:
    day: int
    time: str
    code: str
    action: Action
    weight: float = 0.0
    priority: str = ""
    reason: str = ""


@dataclass
class PortfolioState:
    cash: float
    nav: float
    peak_nav: float
    today_pnl_pct: float = 0.0
    positions: dict = field(default_factory=dict)
    cooldown_left: int = 0
    sold_today: set = field(default_factory=set)
    bought_today: set = field(default_factory=set)


# ============================================================
# 规则引擎主体
# ============================================================
class RuleEngine:
    """L0-L4 分层状态机. 依赖注入 IndicatorFeed."""

    def __init__(self, feed: IndicatorFeed, cfg: Config | None = None):
        self.feed = feed
        self.cfg = cfg or Config()
        self.trackers: dict[str, PeakTracker] = {}
        self.buy_shapes: dict[str, BuyShape] = {}
        self._open_prices: dict[str, float] = {}

    # ---------------- L0 合规 ----------------
    def can_buy(
        self, c: Candidate, tick: Optional[Tick], prev_close: float
    ) -> Optional[str]:
        """返回 None 表示可买, 否则返回否决原因."""
        if c.is_st:
            return "ST禁买"
        if c.is_suspended:
            return "停牌"
        if c.list_days < 60:
            return "次新股"
        if tick and prev_close > 0:
            if tick.price >= prev_close * (1 + price_limit(c.code) / 100) * 0.999:
                return "涨停无法买入"
        return None

    def can_sell(self, pos: Position, tick: Tick, prev_close: float) -> Optional[str]:
        if not pos.sellable:
            return "T+1不可卖"
        if tick.price <= prev_close * (1 - price_limit(pos.code) / 100) * 1.001:
            return "跌停无法卖出"
        return None

    # ---------------- L1 组合闸门 ----------------
    def portfolio_gate(self, pf: PortfolioState) -> tuple[float, Optional[str]]:
        """→ (当日允许总仓位上限, 熔断原因). 与 V3.5 D18 空仓触发分层并存."""
        cfg = self.cfg
        if pf.cooldown_left > 0:
            return 0.0, f"冷静期(剩{pf.cooldown_left}日)"
        dd = pf.nav / pf.peak_nav - 1.0
        if dd <= cfg.dd_derisk_level2:
            return 0.0, f"回撤{dd:.1%}触发清仓线"
        if pf.today_pnl_pct <= cfg.daily_loss_breaker:
            return 0.0, f"当日浮亏{pf.today_pnl_pct:.1%}熔断"
        if dd <= cfg.dd_derisk_level1:
            return 0.60, f"回撤{dd:.1%}，仓位上限60%"
        return cfg.total_position_max, None

    # ---------------- 盘后任务 (STEP1~STEP4) ----------------
    def after_close(
        self, day: int, candidates: list[Candidate], pf: PortfolioState
    ) -> dict:
        """盘后 16:00: 选股池 → 自选标记 → 日线买点 → 日线卖出标记."""
        cfg = self.cfg
        pool, watch, daily_buy, rejected = [], [], [], []
        for c in candidates:
            reason = self.can_buy(c, None, 0)
            if reason is None and c.avg_amount_20d < cfg.min_avg_amount_20d:
                reason = "流动性不足"
            if reason:
                rejected.append((c.code, reason))
                continue
            pool.append(c)
            # STEP2 自选标记: 控盘>30% AND 四CASE任一
            if self.feed.control_ratio(c.code) > cfg.control_min:
                case1 = self.feed.had_accumulation_peak(
                    c.code, cfg.peak_lookback
                ) and self.feed.red_above_blue_since_peak(c.code)
                case2 = c.max_daily_gain_10d >= cfg.surge_pct
                case3 = self.feed.control_weekly_up(c.code)
                case4 = self.feed.bottom_breakout_volume(
                    c.code
                ) and self.feed.recent_shadow_lines(c.code)
                if case1 or case2 or case3 or case4:
                    c.flag_watch = True
                    watch.append(c)
            # STEP3 日线买点: 池内 + 红蓝距离最小 + 换手<50%
            if (
                self.feed.red_blue_distance_min(c.code)
                and c.turnover_today <= cfg.turnover_max_entry
            ):
                c.flag_daily_buy = True
                daily_buy.append(c)

        # STEP4 日线卖出标记 (对持仓股, 次日执行)
        sell_marks = []
        for code, pos in pf.positions.items():
            pos.hold_days += 1
            pos.sellable = True  # 隔日
            reasons = []
            hist = next((c for c in candidates if c.code == code), None)
            if hist:
                if len(hist.daily_closes) >= cfg.daily_close_lookback + 1:
                    if (
                        hist.daily_closes[-1]
                        < hist.daily_closes[-1 - cfg.daily_close_lookback]
                    ):
                        reasons.append(f"收盘低于{cfg.daily_close_lookback}日前")
                if self.feed.profit_chip_ratio(code) < cfg.profit_chip_min:
                    reasons.append(
                        f"获利盘{self.feed.profit_chip_ratio(code):.0f}%<40%"
                    )
                if hist.macd_hist_prev > 0 and hist.macd_hist <= 0:
                    reasons.append("MACD由正转负")
            if pos.hold_days >= cfg.max_hold_days:
                reasons.append(f"持仓满{cfg.max_hold_days}日")
            # P12 双保险: 盘后概率衰减也生成日线标记
            if self.feed.latest_prob_up(code) < cfg.prob_exit:
                reasons.append(
                    f"概率衰减{self.feed.latest_prob_up(code):.2f}<{cfg.prob_exit}"
                )
            if reasons:
                sell_marks.append(
                    Order(
                        day,
                        "AFTER_CLOSE",
                        code,
                        Action.SELL_ALL,
                        priority="P9-日线",
                        reason=";".join(reasons),
                    )
                )
        return {
            "pool": pool,
            "watch": watch,
            "daily_buy": daily_buy,
            "daily_sell_marks": sell_marks,
            "rejected": rejected,
        }

    # ---------------- 集合竞价 (9:15-9:25) ----------------
    def on_auction(
        self,
        day: int,
        pf: PortfolioState,
        auction_prices: dict,
        prev_close: dict,
        daily_sell_marks: list[Order],
    ) -> list[Order]:
        """P2: 竞价高开 >= 5% 且日线卖出已标记 → 全仓卖."""
        orders = []
        marked = {o.code for o in daily_sell_marks}
        for code in marked:
            if code not in pf.positions or code not in auction_prices:
                continue
            gap = (auction_prices[code] / prev_close[code] - 1) * 100
            if gap >= self.cfg.auction_gap_sell:
                orders.append(
                    Order(
                        day,
                        "09:25",
                        code,
                        Action.SELL_ALL,
                        priority="P2",
                        reason=f"竞价高开{gap:.1f}%兑现",
                    )
                )
                pf.sold_today.add(code)
        return orders

    # ---------------- 盘中 tick (9:30-15:00, 每2分钟) ----------------
    def on_tick(
        self,
        day: int,
        tick_map: dict[str, Tick],
        candidates: list[Candidate],
        pf: PortfolioState,
        prev_close: dict,
        breadth: float,
        daily_sell_marks: list[Order],
    ) -> list[Order]:
        orders = []
        marked = {o.code for o in daily_sell_marks}
        cand_map = {c.code: c for c in candidates}

        for code, tick in tick_map.items():
            tr = self.trackers.setdefault(code, PeakTracker(self.cfg))
            tr.update(tick.price)
            pc = prev_close.get(code, tick.price)
            chg = (tick.price / pc - 1) * 100

            # ===== 卖出状态机 (P1→P12 显式优先级) =====
            pos = pf.positions.get(code)
            if pos and code not in pf.sold_today:
                pos.high_since_buy = max(pos.high_since_buy, tick.price)
                blocked = self.can_sell(pos, tick, pc)
                if blocked:
                    if blocked == "T+1不可卖":
                        orders.append(
                            Order(
                                day,
                                tick.time,
                                code,
                                Action.HOLD,
                                priority="L0",
                                reason="T+1，风控挂单待明日",
                            )
                        )
                else:
                    o = self._sell_state_machine(day, code, pos, tick, chg, tr, marked)
                    if o:
                        orders.append(o)
                        if o.action == Action.SELL_ALL:
                            pf.sold_today.add(code)
                        continue  # 已出卖出指令, 本轮不再评估买入

            # ===== 买入: 日内买点判定 + 入场形态 =====
            c = cand_map.get(code)
            if (
                c
                and c.flag_daily_buy
                and not c.flag_intraday_buy
                and code not in pf.positions
                and code not in pf.sold_today
                and code not in pf.bought_today
            ):
                self._check_intraday_buy_point(c, tick, chg, breadth)
            if c and c.flag_intraday_buy and code not in pf.bought_today:
                bo = self._entry_shape_machine(day, c, tick, tr)
                if bo:
                    orders.append(bo)
                    pf.bought_today.add(code)
        return orders

    # ---- 卖出状态机: 返回最高优先级的卖出指令 ----
    def _sell_state_machine(self, day, code, pos, tick, chg, tr, marked):
        cfg = self.cfg
        limit = price_limit(code)
        chip = self.feed.profit_chip_ratio(code)

        # P1 硬止损: 日内跌4% (相对昨收) — 无条件, 不依赖日线标记
        if chg <= cfg.hard_stop_intraday:
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P1",
                reason=f"日内{chg:.1f}%硬止损",
            )

        # P11 移动止盈: 持仓期高点回撤 >= trailing_drawdown 且曾盈利
        if pos.high_since_buy > pos.cost and tick.price > 0:
            dd_high = tick.price / pos.high_since_buy - 1.0
            if dd_high <= -cfg.trailing_drawdown:
                return Order(
                    day,
                    tick.time,
                    code,
                    Action.SELL_ALL,
                    priority="P11",
                    reason=f"移动止盈 高点{pos.high_since_buy:.2f}回撤{dd_high:.1%}",
                )

        # P12 概率衰减: 最新模型概率 < prob_exit
        prob = self.feed.latest_prob_up(code)
        if prob < cfg.prob_exit:
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P12",
                reason=f"概率衰减{prob:.2f}<{cfg.prob_exit}",
            )

        # P3 涨15%全卖 (板块可达才生效: 20%/30%板块)
        if chg >= min(cfg.climax_full_exit, limit - 1.0):
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P3",
                reason=f"涨{chg:.1f}%达板块高潮线，全卖",
            )

        # P4 涨7%+无量+高峰回落 (量价背离)
        if (
            chg >= cfg.climax_move
            and tr.confirmed_peaks
            and tick.price < tr.confirmed_peaks[-1] * (1 - cfg.peak_confirm_drop)
            and tick.volume < 1.0
        ):
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P4",
                reason="涨7%+无量背离，高峰回落",
            )

        # P5 涨7%+主力筹码<40%+下午 → 全卖
        if (
            chg >= cfg.climax_move
            and chip < cfg.chip_profit_min
            and tick.time >= cfg.afternoon
        ):
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P5",
                reason=f"午后涨{chg:.1f}%但获利盘{chip:.0f}%<40%，全卖",
            )

        # P6 换手>40% 且涨8% → 全卖
        if tick.turnover > cfg.climax_turnover and chg >= cfg.half_sell_move:
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P6",
                reason=f"换手{tick.turnover:.0f}%+涨{chg:.1f}%，全卖",
            )

        # P7 涨7%+换手>40% → 卖一半 (只执行一次)
        if (
            chg >= cfg.climax_move
            and tick.turnover > cfg.climax_turnover
            and not pos.half_sold
        ):
            pos.half_sold = True
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_HALF,
                priority="P7",
                reason="涨7%+高换手，减半",
            )

        # P8 三峰连续下降 → 第三峰确认后卖
        if tr.three_peaks_descending():
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P8",
                reason="三峰连续下降",
            )

        # P9 日线卖出标记 (盘后生成, 盘中执行)
        if code in marked:
            return Order(
                day,
                tick.time,
                code,
                Action.SELL_ALL,
                priority="P9",
                reason="日线卖出标记执行",
            )

        # P10 浮盈20%+ WARNING (不自动卖, 人工复核)
        gain = (tick.price / pos.cost - 1) * 100
        if gain >= cfg.warn_gain:
            return Order(
                day,
                tick.time,
                code,
                Action.WARN,
                priority="P10",
                reason=f"浮盈{gain:.0f}%≥{cfg.warn_gain:.0f}%，人工复核",
            )
        return None

    # ---- 日内买点: 路径A(跳空) / 路径B(10:40前多维确认) ----
    def _check_intraday_buy_point(
        self, c: Candidate, tick: Tick, chg: float, breadth: float
    ) -> None:
        cfg = self.cfg
        # 路径A: 跳空>4% + 放量
        if chg >= cfg.gap_open_pct and tick.volume >= 1.5:
            c.flag_intraday_buy = True
            self.buy_shapes[c.code] = BuyShape.WAIT_OPEN
            return
        # 路径B: ≤10:40 + 广度 + 大单净量 + 控盘>30% + 红柱升且过半
        if (
            tick.time <= cfg.buy_cutoff
            and breadth >= cfg.breadth_min
            and tick.big_order_net > 0
            and self.feed.control_ratio(c.code) > cfg.control_min
            and self.feed.red_bar_rising_and_majority(c.code)
        ):
            c.flag_intraday_buy = True
            self.buy_shapes[c.code] = BuyShape.WAIT_OPEN

    # ---- 入场形态状态机 ----
    def _entry_shape_machine(self, day, c, tick, tr) -> Optional[Order]:
        state = self.buy_shapes.get(c.code, BuyShape.WAIT_OPEN)

        if state == BuyShape.WAIT_OPEN:
            # 前10分钟 (5根2分钟bar) 定方向
            if len(tr.bars) >= 5:
                first, now = tr.bars[0], tick.price
                self.buy_shapes[c.code] = (
                    BuyShape.DOWN_WAIT_TROUGH if now < first else BuyShape.UP_WAIT_PEAK2
                )
            return None

        if state == BuyShape.DOWN_WAIT_TROUGH:
            # 下行→低峰确认后回升处买入
            if tr.confirmed_troughs and tick.price > tr.confirmed_troughs[-1]:
                self.buy_shapes[c.code] = BuyShape.DONE
                return Order(
                    day,
                    tick.time,
                    c.code,
                    Action.BUY,
                    priority="L4-形态",
                    reason=f"下探后低峰{tr.confirmed_troughs[-1]:.2f}确认回升",
                )

        if state == BuyShape.UP_WAIT_PEAK2:
            p = tr.confirmed_peaks
            if len(p) >= 2 and p[-1] > p[-2]:
                self.buy_shapes[c.code] = BuyShape.UP_WAIT_TROUGH2

        if state == BuyShape.UP_WAIT_TROUGH2:
            t = tr.confirmed_troughs
            if t and tick.price > t[-1]:
                self.buy_shapes[c.code] = BuyShape.DONE
                return Order(
                    day,
                    tick.time,
                    c.code,
                    Action.BUY,
                    priority="L4-形态",
                    reason="双峰走高后回落再升，买入",
                )
        return None
