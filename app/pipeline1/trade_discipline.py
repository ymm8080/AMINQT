"""
附录D.3/D.5 交易纪律状态机 + 交易日志 (IMPLEMENTATION_PLAN_v3.2 P23.5-P23.7)
============================================================================
D.3 三条硬规则:
  1. 止损硬化: ATR 驱动 + board_type 分档 [P23.5].
     时间止损: 个股 20 日 2 日收益中位数 [P23.6].
  2. 日内保险丝: 组合滚动 20 日回撤 2σ, 与固定 daily_loss_limit 双轨并行 [P23.7].
  3. 停机线: 总资金自峰值回撤 >15% → 停机一周复盘.
D.5 样本积累纪律 / D.10 实盘解锁双闸门.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

HARD_STOP = -0.04
TIME_STOP_DAYS = 2
TIME_STOP_MIN_RET = 0.01
DAILY_LOSS_LIMIT = 0.04
DRAWDOWN_LIMIT = 0.15
HALT_DAYS = 7
UNLOCK_MIN_TRADES = 40
UNLOCK_MIN_EXPECTANCY = 0.005
UNLOCK_MAX_CONSEC_LOSS = 5
STAGE3_MIN_TRADES = 20
STAGE3_MIN_EXPECTANCY = 0.003
STAGE3_MAX_CONSEC_LOSS = 4

STATE_ACTIVE = "ACTIVE"
STATE_LOCKED_TODAY = "LOCKED_TODAY"
STATE_HALTED = "HALTED"


# ============================================================
# P23.5 自适应止损: ATR 驱动 + 噪音带校验
# ============================================================
def adaptive_stop_loss(stock, board_type="main", profile=None):
    """D-20: ATR驱动止损, 分主板/双创. 返回止损比例 (负数)."""
    atr_mult = 1.5
    stop_fixed = -0.04
    if profile:
        atr_mult = profile.get("stop_loss_atr_mult", 1.5)
        stop_fixed = profile.get("stop_loss_fixed", -0.04)
    if "atr_pct" in stock:
        atr_pct = float(stock["atr_pct"])
    else:
        atr_14 = stock.get("ATR_14", 0)
        close = stock.get("close", 1)
        atr_pct = float(atr_14) / float(close) if close > 0 else 0.02
    if board_type == "main":
        stop = max(stop_fixed, -atr_mult * atr_pct)
    elif board_type == "chinext":
        stop_fixed_cx = profile.get("stop_loss_fixed", -0.06) if profile else -0.06
        stop = max(stop_fixed_cx, -atr_mult * atr_pct)
    else:
        stop = -atr_mult * atr_pct
    min_buffer = 1.2 * atr_pct
    if abs(stop) < min_buffer:
        logger.warning("止损 %.4f 落入噪音带, 上调至 %.4f", stop, -min_buffer)
        stop = -min_buffer
    return round(stop, 4)


# ============================================================
# P23.6 时间止损: 个股 20 日 2 日收益中位数
# ============================================================
def time_stop(stock, entry_price, current_price, holding_days):
    """D-21: 相对基准时间止损 (替代固定 ret<0.01)."""
    if holding_days < TIME_STOP_DAYS:
        return False
    if entry_price <= 0:
        return False
    ret_2d = current_price / entry_price - 1
    hist_2d_rets = np.asarray(stock.get("rolling_2d_return_20d", []))
    if len(hist_2d_rets) < 5:
        return bool(ret_2d < TIME_STOP_MIN_RET)
    median_2d = float(np.median(hist_2d_rets))
    return bool(ret_2d < median_2d)


# ============================================================
# P23.7 日保险丝: 组合滚动 20 日回撤 2σ
# ============================================================
def daily_fuse(portfolio):
    """D-22: 组合自适应日保险丝 — 2σ阈值 (替代固定 daily_loss_limit=0.04)."""
    daily_pnl = portfolio.get("daily_realized_pnl", 0)
    hist_dd = np.asarray(portfolio.get("daily_drawdown_20d", []))
    if len(hist_dd) < 5:
        limit = portfolio.get("daily_loss_limit_fixed", DAILY_LOSS_LIMIT)
        if daily_pnl < -limit:
            return True, f"DAILY_FUSE_FIXED_{daily_pnl:.3f}"
        return False, "NORMAL"
    mu = float(np.mean(hist_dd))
    sigma = float(np.std(hist_dd))
    threshold = mu - 2 * sigma
    if daily_pnl < threshold:
        logger.error("D.22 日保险丝: 日亏 %.2f%% < u-2o", daily_pnl * 100)
        return True, "DAILY_FUSE_2SIGMA"
    return False, "NORMAL"


# ============================================================
# D.3 硬规则状态机
# ============================================================
class TradingDiscipline:
    """D.3 硬规则状态机. P23.5-P23.7: 止损/时间止损/日保险丝支持 profile 覆盖."""

    def __init__(
        self,
        hard_stop=HARD_STOP,
        daily_loss_limit=DAILY_LOSS_LIMIT,
        drawdown_limit=DRAWDOWN_LIMIT,
        profile=None,
    ):
        self.hard_stop = hard_stop
        self.daily_loss_limit = daily_loss_limit
        self.daily_loss_limit_fixed = daily_loss_limit
        self.drawdown_limit = drawdown_limit
        self.profile = profile
        self.state = STATE_ACTIVE
        self.peak_nav = 0.0
        self.halt_day = -1
        self.attribution = ""
        self._locked_day = -1
        self._worm = []
        self._daily_drawdowns = []
        if profile:
            self.hard_stop = profile.get("stop_loss", hard_stop)
            self.daily_loss_limit = profile.get("daily_loss_limit", daily_loss_limit)
            self.drawdown_limit = profile.get("drawdown_limit", drawdown_limit)

    def check_hard_stop(self, pnl):
        return bool(pnl <= self.hard_stop)

    def check_time_stop(
        self, hold_days, ret, stock=None, entry_price=0.0, current_price=0.0
    ):
        if hold_days < TIME_STOP_DAYS:
            return False
        if stock and entry_price > 0 and current_price > 0:
            return time_stop(stock, entry_price, current_price, hold_days)
        return bool(ret < TIME_STOP_MIN_RET)

    def on_daily_pnl(self, day, daily_pnl_pct, daily_drawdown_20d=None):
        self._daily_drawdowns.append(daily_pnl_pct)
        if len(self._daily_drawdowns) > 60:
            self._daily_drawdowns = self._daily_drawdowns[-60:]
        fused = False
        label = "NORMAL"
        hist_dd_list = (
            list(daily_drawdown_20d)
            if daily_drawdown_20d
            else (
                self._daily_drawdowns[-20:] if len(self._daily_drawdowns) >= 5 else []
            )
        )
        portfolio = {
            "daily_realized_pnl": daily_pnl_pct,
            "daily_drawdown_20d": hist_dd_list,
            "daily_loss_limit_fixed": self.daily_loss_limit_fixed,
        }
        fused_2s, label_2s = daily_fuse(portfolio)
        if fused_2s:
            fused = True
            label = label_2s
        if not fused and daily_pnl_pct <= -self.daily_loss_limit_fixed:
            fused = True
            label = f"DAILY_FUSE_FIXED_{daily_pnl_pct:.3f}"
        if fused and self.state == STATE_ACTIVE:
            self.state = STATE_LOCKED_TODAY
            self._locked_day = day
            self._worm.append(
                {
                    "day": day,
                    "event": "DAILY_FUSE",
                    "pnl": round(daily_pnl_pct, 4),
                    "label": label,
                }
            )
            logger.error(
                "D.3 日保险丝: 日亏 %.1f%% (%s), 当日锁仓", -daily_pnl_pct * 100, label
            )
            return {"state": self.state, "action": "LOCK_TODAY", "label": label}
        if self.state == STATE_LOCKED_TODAY and day > self._locked_day:
            self.state = STATE_ACTIVE
            logger.warning("D.3 日保险丝: 次日恢复交易")
        return {"state": self.state, "action": "HOLD", "label": label}

    def on_nav(self, day, nav):
        self.peak_nav = max(self.peak_nav, nav)
        if self.state == STATE_HALTED:
            return {"state": self.state, "action": "HALTED"}
        if self.peak_nav > 0:
            dd = nav / self.peak_nav - 1
            if dd <= -self.drawdown_limit:
                self.state = STATE_HALTED
                self.halt_day = day
                self._worm.append(
                    {"day": day, "event": "HALT", "drawdown": round(dd, 4)}
                )
                logger.critical(
                    "D.3 停机线: 回撤 %.1f%% 触及 %.0f%%, 停机 %d 天复盘",
                    -dd * 100,
                    self.drawdown_limit * 100,
                    HALT_DAYS,
                )
                return {"state": self.state, "action": "HALT"}
        return {"state": self.state, "action": "HOLD"}

    def can_buy(self, day):
        if self.state == STATE_HALTED:
            return False
        return not (self.state == STATE_LOCKED_TODAY and day <= self._locked_day)

    def resume_after_halt(self, day, attribution):
        assert self.state == STATE_HALTED, "未处于停机状态"
        if day < self.halt_day + HALT_DAYS:
            return {"resumed": False, "reason": f"停机未满 {HALT_DAYS} 天"}
        if not attribution.strip():
            logger.error("D.3 重启拒绝: 缺书面归因")
            return {"resumed": False, "reason": "未归因前不得重启"}
        self.attribution = attribution
        self.state = STATE_ACTIVE
        self._worm.append(
            {"day": day, "event": "RESUME", "attribution": attribution[:200]}
        )
        logger.warning("D.3 停机复盘完成, 重启交易")
        return {"resumed": True}

    def worm_log(self):
        return list(self._worm)


# ============================================================
# D.5 交易日志 + 样本统计
# ============================================================
@dataclass
class TradeRecord:
    symbol: str
    signal_grade: str
    prob_up: float
    rank_score: float
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    slippage: float = 0.0
    hold_days: int = 0
    pnl_pct: float = 0.0


@dataclass
class TradeJournal:
    _trades: list = field(default_factory=list)

    def record(self, rec):
        self._trades.append(rec)

    def close_trade(self, symbol, exit_date, exit_price, hold_days, pnl_pct):
        for rec in reversed(self._trades):
            if rec.symbol == symbol and not rec.exit_date:
                rec.exit_date, rec.exit_price = exit_date, exit_price
                rec.hold_days, rec.pnl_pct = hold_days, pnl_pct
                return
        raise KeyError(f"无未平仓记录: {symbol}")

    @property
    def closed(self):
        return [t for t in self._trades if t.exit_date]

    def sample_stats(self):
        pnls = np.array([t.pnl_pct for t in self.closed])
        n = len(pnls)
        if n == 0:
            return {"n_trades": 0}
        wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
        win_rate = float(len(wins) / n)
        pl_ratio = (
            float(wins.mean() / abs(losses.mean()))
            if len(losses) and len(wins)
            else 0.0
        )
        max_streak = streak = 0
        for p in pnls:
            streak = streak + 1 if p <= 0 else 0
            max_streak = max(max_streak, streak)
        return {
            "n_trades": n,
            "win_rate": round(win_rate, 4),
            "pl_ratio": round(pl_ratio, 4),
            "expectancy": round(float(pnls.mean()), 5),
            "max_consec_loss": max_streak,
        }

    def unlock_check(self):
        s = self.sample_stats()
        n = s.get("n_trades", 0)
        if n < UNLOCK_MIN_TRADES:
            return {"unlock": False, "reason": f"样本不足 {n}/{UNLOCK_MIN_TRADES}", **s}
        ok = (
            s["expectancy"] > UNLOCK_MIN_EXPECTANCY
            and s["max_consec_loss"] <= UNLOCK_MAX_CONSEC_LOSS
        )
        if not ok:
            logger.warning(
                "D.10 解锁否决: 期望 %.3f%% / 最大连亏 %d",
                s["expectancy"] * 100,
                s["max_consec_loss"],
            )
        return {"unlock": ok, **s}

    def stage3_gate(self):
        s = self.sample_stats()
        n = s.get("n_trades", 0)
        if n < STAGE3_MIN_TRADES:
            return {"pass": False, "reason": f"样本不足 {n}/{STAGE3_MIN_TRADES}", **s}
        ok = (
            s["expectancy"] > STAGE3_MIN_EXPECTANCY
            and s["max_consec_loss"] <= STAGE3_MAX_CONSEC_LOSS
        )
        if not ok:
            logger.warning(
                "P19.2 阶段三裁决否决: 期望 %.3f%% / 最大连亏 %d",
                s["expectancy"] * 100,
                s["max_consec_loss"],
            )
        return {"pass": ok, **s}
