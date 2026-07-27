# -*- coding: utf-8 -*-
"""
OOS 监控 + Kill Switch + BIAS 红灯规则 (IMPLEMENTATION_PLAN_v3.2 P25.3)
========================================================================
[E4] IC 日度三色灯: 每日推理后记录当日 OOS IC, 滚动 20 日 u/o:
  🟢 IC > u-1o    : 正常
  🟡 u-2o < IC < u-1o : 警告, 连续3日🟡 → L1
  🔴 IC < u-2o    : 模型失效嫌疑, 触发 L1 (立即降级模拟盘)
[P25.3] BIAS 红灯规则: bias_big_down > +0.02 → 触发 E4-L1 模型降级.
Kill Switch: 连续 2 个月滚动 20 日 IC 均值 < 0.01 → 模型退役.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.utils.daily_rank_ic import cross_sectional_rank_ic

logger = logging.getLogger(__name__)

IC_NORMAL = 0.03
IC_WARN = 0.01
RED_DAYS = 3
HALT_DAYS = 5
KILL_MONTHS = 2
KILL_WINDOW = 20
LIGHT_WINDOW = 20
YELLOW_STREAK_L1 = 3
LIGHT_MIN_SAMPLES = 5

STATE_NORMAL = "NORMAL"
STATE_YELLOW = "YELLOW_REVIEW"
STATE_RED_SIM = "RED_SIMULATE"
STATE_HALT = "HALT"
STATE_RETIRED = "RETIRED"


# ---- P25.3 BIAS 红灯规则 (D-25) ----
def bias_traffic_light(quality):
    """D-25: BIAS红灯规则.

    Args:
        quality: compute_quality_metrics + compute_bias_buckets 输出 dict

    Returns:
        'GREEN' | 'YELLOW' | 'RED'
    """
    bias_big_down = quality.get("bias_big_down", 0)
    if bias_big_down and not (
        isinstance(bias_big_down, float) and np.isnan(bias_big_down)
    ):
        if bias_big_down > 0.02:
            logger.critical(
                "BIAS 红灯: bias_big_down=%.4f > 0.02, 大跌日模型高估!", bias_big_down
            )
            trigger_e4_l1("BIAS_BIG_DOWN_RED", bias_big_down)
            return "RED"

    bias_1d = quality.get("bias_1d", 0)
    if bias_1d and not (isinstance(bias_1d, float) and np.isnan(bias_1d)):
        if abs(bias_1d) > 0.01:
            return "RED"
        if abs(bias_1d) > 0.005:
            return "YELLOW"

    dir_acc = quality.get("direction_accuracy", 0)
    if dir_acc and not (isinstance(dir_acc, float) and np.isnan(dir_acc)):
        if dir_acc < 0.50:
            return "RED"

    return "GREEN"


def trigger_e4_l1(reason, value):
    """触发 E4-L1 模型降级 (记录 + 告警, 实际降级由上层调用)."""
    logger.critical("E4-L1 触发: %s (value=%.4f), 模型降级为模拟盘", reason, value)


@dataclass
class OOSMonitor:
    """OOS 监控状态机. P25.3: daily_check 增加 BIAS 输入."""

    ic_history: list[float] = field(default_factory=list)
    state: str = STATE_NORMAL
    _red_streak: int = 0
    _neg_streak: int = 0
    _yellow_streak: int = 0

    @staticmethod
    def daily_rank_ic(pred_scores, actual_returns):
        df = pd.DataFrame({"s": pred_scores, "r": actual_returns}).dropna()
        try:
            ic = cross_sectional_rank_ic(
                df["s"], df["r"], min_x_unique=2, min_y_unique=1
            )
        except Exception:
            logger.warning("cross_sectional_rank_ic 计算异常, 返回 0.0", exc_info=True)
            return 0.0
        return 0.0 if np.isnan(ic) else float(ic)

    def ic_traffic_light(self, ic_today):
        hist = self.ic_history[-LIGHT_WINDOW:]
        if len(hist) < LIGHT_MIN_SAMPLES:
            return "GREEN"
        mu, sigma = float(np.mean(hist)), float(np.std(hist))
        if ic_today < mu - 2 * sigma:
            return "RED"
        if ic_today < mu - 1 * sigma:
            return "YELLOW"
        return "GREEN"

    def daily_check(self, ic_today, quality=None):
        """输入当日 Rank IC + (可选) BIAS 质量指标.

        Args:
            ic_today: 当日 Rank IC
            quality: P25.1/P25.2 输出 dict (含 mae_1d/bias_1d/direction_accuracy/bias_big_down)

        Returns:
            {'state', 'action', 'rolling_ic_5d', 'light', 'bias_light'}
        """
        light = self.ic_traffic_light(ic_today)
        self.ic_history.append(ic_today)
        rolling = float(np.mean(self.ic_history[-5:]))

        # P25.3 BIAS 红灯 (与 IC 三色灯并列)
        bias_light = "NONE"
        if quality:
            bias_light = bias_traffic_light(quality)
            if bias_light == "RED":
                self._yellow_streak = 0
                self.state = STATE_RED_SIM
                action = "🔴 BIAS_RED: bias_big_down>0.02, E4-L1 模型降级"
                logger.error(action)
                return {
                    "state": self.state,
                    "action": action,
                    "rolling_ic_5d": round(rolling, 4),
                    "light": light,
                    "bias_light": bias_light,
                }

        # E4 三色灯裁决
        if light == "RED":
            self._yellow_streak = 0
            self.state = STATE_RED_SIM
            action = "🔴 L1: IC < u-2o, 模型失效嫌疑, 立即降级模拟盘"
            logger.error(action)
            return {
                "state": self.state,
                "action": action,
                "rolling_ic_5d": round(rolling, 4),
                "light": light,
                "bias_light": bias_light,
            }
        if light == "YELLOW":
            self._yellow_streak += 1
            if self._yellow_streak >= YELLOW_STREAK_L1:
                self.state = STATE_RED_SIM
                action = f"🟡x{self._yellow_streak} → L1: 连续黄灯, 降级模拟盘"
                logger.error(action)
                return {
                    "state": self.state,
                    "action": action,
                    "rolling_ic_5d": round(rolling, 4),
                    "light": light,
                    "bias_light": bias_light,
                }
        else:
            self._yellow_streak = 0

        # 绝对阈值档位 (L2/L3 后备)
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
        return {
            "state": self.state,
            "action": action,
            "rolling_ic_5d": round(rolling, 4),
            "light": light,
            "bias_light": bias_light,
        }

    def kill_switch_check(self):
        if len(self.ic_history) < KILL_WINDOW * KILL_MONTHS:
            return {"retire": False, "reason": "样本不足"}
        recent_2m = self.ic_history[-KILL_WINDOW * KILL_MONTHS :]
        m1 = float(np.mean(recent_2m[:KILL_WINDOW]))
        m2 = float(np.mean(recent_2m[KILL_WINDOW:]))
        if m1 < IC_WARN and m2 < IC_WARN:
            self.state = STATE_RETIRED
            logger.critical(
                "KILL SWITCH: 连续2月滚动20日 IC 均值 %.4f/%.4f < 0.01, 模型退役",
                m1,
                m2,
            )
            return {
                "retire": True,
                "month_ic": [round(m1, 4), round(m2, 4)],
                "procedure": [
                    "停止实盘交易",
                    "排查原因",
                    "重新训练或调整特征",
                    "通过 OOS 验收后才可重新上线",
                ],
            }
        return {"retire": False, "month_ic": [round(m1, 4), round(m2, 4)]}
