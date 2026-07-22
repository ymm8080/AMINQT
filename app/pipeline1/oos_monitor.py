# -*- coding: utf-8 -*-
"""
OOS 监控 + Kill Switch (DESIGN §14.6, 安全网 #10)
=====================================================
每日计算 Top 15 组合 Rank IC (滚动 5 日):
  IC > 0.03           → 正常
  0.01 < IC < 0.03    → 黄色预警, 人工复核
  IC < 0.01 连续 3 日 → 红色警报, 自动降级为模拟盘
  IC < 0 连续 5 日    → 立即停机 (熔断)
Kill Switch: 连续 2 个月滚动 20 日 IC 均值 < 0.01 → 模型退役
  退役流程: 停实盘 → 排查(数据源/特征/市场结构) → 重训 → OOS 复验后才可重上线
没有 kill switch 的量化模型, 亏损期你分不清是运气还是失效.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

IC_NORMAL = 0.03
IC_WARN = 0.01
RED_DAYS = 3
HALT_DAYS = 5
KILL_MONTHS = 2
KILL_WINDOW = 20

STATE_NORMAL = "NORMAL"
STATE_YELLOW = "YELLOW_REVIEW"
STATE_RED_SIM = "RED_SIMULATE"     # 降级模拟盘
STATE_HALT = "HALT"                # 熔断停机
STATE_RETIRED = "RETIRED"          # Kill Switch 退役


@dataclass
class OOSMonitor:
    """OOS 监控状态机. 每日调用 daily_check(当日 Top15 预测 vs 实际收益)."""

    ic_history: list[float] = field(default_factory=list)   # 每日 Rank IC
    state: str = STATE_NORMAL
    _red_streak: int = 0
    _neg_streak: int = 0

    # ---------------- 当日 IC ----------------
    @staticmethod
    def daily_rank_ic(pred_scores: pd.Series, actual_returns: pd.Series) -> float:
        """Top 15 清单预测得分 vs 次日实际收益的 Spearman IC."""
        df = pd.DataFrame({"s": pred_scores, "r": actual_returns}).dropna()
        if len(df) < 5:
            return 0.0
        return float(spearmanr(df["s"], df["r"]).statistic)

    # ---------------- 每日检查 ----------------
    def daily_check(self, ic_today: float) -> dict:
        """输入当日 Rank IC, 返回 {'state', 'action', 'rolling_ic_5d'}."""
        self.ic_history.append(ic_today)
        rolling = float(np.mean(self.ic_history[-5:]))

        if rolling >= IC_NORMAL:
            self._red_streak = self._neg_streak = 0
            self.state = STATE_NORMAL
            action = "正常运行"
        elif rolling >= IC_WARN:
            self._red_streak = self._neg_streak = 0
            self.state = STATE_YELLOW
            action = "黄色预警: 人工复核"
            logger.warning("OOS 黄色预警: 滚动5日 IC=%.4f", rolling)
        else:
            if rolling < IC_WARN:
                self._red_streak += 1
            if rolling < 0:
                self._neg_streak += 1
            else:
                self._neg_streak = 0
            if self._neg_streak >= HALT_DAYS:
                self.state = STATE_HALT
                action = "熔断: IC<0 连续5日, 立即停机"
                logger.critical(action)
            elif self._red_streak >= RED_DAYS:
                self.state = STATE_RED_SIM
                action = "红色警报: IC<0.01 连续3日, 自动降级为模拟盘"
                logger.error(action)
            else:
                self.state = STATE_YELLOW
                action = "黄色预警: 人工复核"
        return {"state": self.state, "action": action, "rolling_ic_5d": round(rolling, 4)}

    # ---------------- Kill Switch ----------------
    def kill_switch_check(self) -> dict:
        """连续 2 个月滚动 20 日 IC 均值 < 0.01 → 模型退役."""
        if len(self.ic_history) < KILL_WINDOW * KILL_MONTHS:
            return {"retire": False, "reason": "样本不足"}
        recent_2m = self.ic_history[-KILL_WINDOW * KILL_MONTHS:]
        m1 = float(np.mean(recent_2m[:KILL_WINDOW]))
        m2 = float(np.mean(recent_2m[KILL_WINDOW:]))
        if m1 < IC_WARN and m2 < IC_WARN:
            self.state = STATE_RETIRED
            logger.critical("KILL SWITCH: 连续2月滚动20日 IC 均值 %.4f/%.4f < 0.01, 模型退役", m1, m2)
            return {"retire": True, "month_ic": [round(m1, 4), round(m2, 4)],
                    "procedure": ["停止实盘交易", "排查原因 (数据源/特征/市场结构变化)",
                                  "重新训练或调整特征", "通过 OOS 验收后才可重新上线"]}
        return {"retire": False, "month_ic": [round(m1, 4), round(m2, 4)]}
