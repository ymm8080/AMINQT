"""
附录D.3/D.5 交易纪律状态机 + 交易日志 (PIPELINE1_V3.8 附录D, 检查清单 D-2~D-5)
================================================================================
D.3 三条硬规则 (不可违背, 抽掉就是爆仓不是暴利):
  1. 止损硬化: 单笔 -4% 无条件砍 (优先级高于一切其他卖出规则);
     时间止损: 买入 2 日涨幅 <1% → 撤
  2. 日内保险丝: 单日亏损 > 日限 → 当日锁仓, 次日恢复
     (阻断"止损+新仓再亏"的连锁失血)
  3. 停机线: 总资金自峰值回撤 >15% → 停机一周复盘, 复盘产出书面归因
     (模式失效/执行变形/市场环境), 未归因前不得重启
D.5 样本积累纪律: 前 2-3 个月的目标是买数据不是买收益 (40-60 笔);
  每笔记录 信号等级/prob_up/rank_score/入场/出场/滑点/持仓时长/盈亏;
  满 40 笔计算真实胜率与盈亏比回喂参数; 禁止样本不足时凭感觉调参 (安全网#15).
D.10 实盘解锁双闸门: 40 笔满足 期望>+0.5%/笔 且 最大连亏≤5 笔, 方可切 C 档.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# D.3 默认值 (与 profiles.aggressive 对齐; stable 档经 profile 传入覆盖)
HARD_STOP = -0.04  # 单笔硬止损
TIME_STOP_DAYS = 2  # 时间止损: 买入 N 日
TIME_STOP_MIN_RET = 0.01  # 涨幅 <1% → 撤
DAILY_LOSS_LIMIT = 0.04  # 日保险丝
DRAWDOWN_LIMIT = 0.15  # 停机线
HALT_DAYS = 7  # 停机一周
# D.10 解锁闸门
UNLOCK_MIN_TRADES = 40
UNLOCK_MIN_EXPECTANCY = 0.005  # 期望 > +0.5%/笔
UNLOCK_MAX_CONSEC_LOSS = 5

STATE_ACTIVE = "ACTIVE"
STATE_LOCKED_TODAY = "LOCKED_TODAY"  # 日保险丝触发
STATE_HALTED = "HALTED"  # 停机线触发


class TradingDiscipline:
    """D.3 硬规则状态机 (组合级, 首日生效无豁免; 熊市协议优先级更高)."""

    def __init__(
        self,
        hard_stop: float = HARD_STOP,
        daily_loss_limit: float = DAILY_LOSS_LIMIT,
        drawdown_limit: float = DRAWDOWN_LIMIT,
    ):
        self.hard_stop = hard_stop
        self.daily_loss_limit = daily_loss_limit
        self.drawdown_limit = drawdown_limit
        self.state = STATE_ACTIVE
        self.peak_nav = 0.0
        self.halt_day = -1  # 停机触发日 (交易日序号)
        self.attribution = ""  # 书面归因 (未归因不得重启)
        self._locked_day = -1
        self._worm: list[dict] = []  # WORM 事件日志

    # ---------------- 硬规则 1: 止损硬化 + 时间止损 ----------------
    def check_hard_stop(self, pnl: float) -> bool:
        """单笔浮亏 ≤ -4% → 无条件砍 (True=立即卖出, 优先级高于一切卖出规则)."""
        return bool(pnl <= self.hard_stop)

    @staticmethod
    def check_time_stop(hold_days: int, ret: float) -> bool:
        """时间止损: 买入 2 日涨幅 <1% → 撤 (亏得干脆, 赚得耐心)."""
        return bool(hold_days >= TIME_STOP_DAYS and ret < TIME_STOP_MIN_RET)

    # ---------------- 硬规则 2: 日内保险丝 ----------------
    def on_daily_pnl(self, day: int, daily_pnl_pct: float) -> dict:
        """每日收盘上报当日盈亏. 亏损 > 日限 → 当日锁仓, 次日自动恢复."""
        if daily_pnl_pct <= -self.daily_loss_limit and self.state == STATE_ACTIVE:
            self.state = STATE_LOCKED_TODAY
            self._locked_day = day
            self._worm.append(
                {"day": day, "event": "DAILY_FUSE", "pnl": round(daily_pnl_pct, 4)}
            )
            logger.error(
                "D.3 日保险丝: 日亏 %.1f%% 触及 %.1f%%, 当日锁仓",
                -daily_pnl_pct * 100,
                self.daily_loss_limit * 100,
            )
            return {"state": self.state, "action": "LOCK_TODAY"}
        if self.state == STATE_LOCKED_TODAY and day > self._locked_day:
            self.state = STATE_ACTIVE  # 次日恢复
            logger.warning("D.3 日保险丝: 次日恢复交易")
        return {"state": self.state, "action": "HOLD"}

    # ---------------- 硬规则 3: 停机线 ----------------
    def on_nav(self, day: int, nav: float) -> dict:
        """每日上报净值. 自峰值回撤 >15% → 停机一周 + 书面归因后方可重启."""
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

    def can_buy(self, day: int) -> bool:
        """当日是否允许开新仓 (停机/锁仓日禁止; 满仓买入后当日物理锁死由执行层保证)."""
        if self.state == STATE_HALTED:
            return False
        return not (self.state == STATE_LOCKED_TODAY and day <= self._locked_day)

    def resume_after_halt(self, day: int, attribution: str) -> dict:
        """停机复盘后重启: 满 7 天 且 有书面归因 (未归因前不得重启, D.3)."""
        assert self.state == STATE_HALTED, "未处于停机状态"
        if day < self.halt_day + HALT_DAYS:
            return {"resumed": False, "reason": f"停机未满 {HALT_DAYS} 天"}
        if not attribution.strip():
            logger.error("D.3 重启拒绝: 缺书面归因 (模式失效/执行变形/市场环境)")
            return {"resumed": False, "reason": "未归因前不得重启"}
        self.attribution = attribution
        self.state = STATE_ACTIVE
        self._worm.append(
            {"day": day, "event": "RESUME", "attribution": attribution[:200]}
        )
        logger.warning("D.3 停机复盘完成, 重启交易")
        return {"resumed": True}

    def worm_log(self) -> list[dict]:
        """WORM 事件日志 (只读)."""
        return list(self._worm)


# ============================================================
# D.5 交易日志 + 样本统计
# ============================================================
@dataclass
class TradeRecord:
    """D.5 交易日志 schema (每笔留痕)."""

    symbol: str
    signal_grade: str  # 信号等级 (A/B)
    prob_up: float
    rank_score: float
    entry_date: str
    entry_price: float
    exit_date: str = ""
    exit_price: float = 0.0
    slippage: float = 0.0  # 实际滑点 (E5 分层对照)
    hold_days: int = 0
    pnl_pct: float = 0.0  # 净盈亏 (扣费后)


@dataclass
class TradeJournal:
    """D.5 交易日志: 前 2-3 个月目标是买数据 (40-60 笔), 不是买收益."""

    _trades: list[TradeRecord] = field(default_factory=list)

    def record(self, rec: TradeRecord) -> None:
        self._trades.append(rec)

    def close_trade(
        self,
        symbol: str,
        exit_date: str,
        exit_price: float,
        hold_days: int,
        pnl_pct: float,
    ) -> None:
        """平仓回填 (找最近一笔未平仓的该标的)."""
        for rec in reversed(self._trades):
            if rec.symbol == symbol and not rec.exit_date:
                rec.exit_date, rec.exit_price = exit_date, exit_price
                rec.hold_days, rec.pnl_pct = hold_days, pnl_pct
                return
        raise KeyError(f"无未平仓记录: {symbol}")

    @property
    def closed(self) -> list[TradeRecord]:
        return [t for t in self._trades if t.exit_date]

    def sample_stats(self) -> dict:
        """满 40 笔后的真实统计: 胜率 / 盈亏比 / 期望 / 最大连亏 (D.5)."""
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
        # 最大连亏
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

    def unlock_check(self) -> dict:
        """D.10 实盘解锁闸门: ≥40 笔 且 期望>+0.5%/笔 且 最大连亏≤5 → 可切 C 档.

        回测不覆盖执行风险 (成交率/真实滑点/跳空频率), 解锁必须用实盘数据二次确认.
        """
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
                "D.10 解锁否决: 期望 %.3f%% / 最大连亏 %d (门槛 >0.5%% / ≤5)",
                s["expectancy"] * 100,
                s["max_consec_loss"],
            )
        return {"unlock": ok, **s}
