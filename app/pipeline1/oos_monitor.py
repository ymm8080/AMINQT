"""
OOS 监控 + Kill Switch (DESIGN §14.6, 安全网 #10, PIPELINE1_V3.8 §2.7 E4)
==========================================================================
[E4] IC 日度三色灯 (替代月度监控): 每日推理后记录当日 OOS IC,
  滚动 20 日均值 μ 与标准差 σ:
  🟢 IC > μ-1σ        : 正常
  🟡 μ-2σ < IC < μ-1σ : 警告, 进入复检队列 (连续3日🟡 → 按 L1 处理)
  🔴 IC < μ-2σ        : 模型失效嫌疑, 触发 L1 切换 (立即降级模拟盘)
历史绝对阈值档位 (IC>0.03 正常 / IC<0.01 连续3日降级 / IC<0 连续5日熔断) 保留作
L2/L3 后备; Kill Switch: 连续 2 个月滚动 20 日 IC 均值 < 0.01 → 模型退役.
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
# E4 三色灯
LIGHT_WINDOW = 20  # 滚动 μ/σ 窗口
YELLOW_STREAK_L1 = 3  # 连续 3 日黄灯 → 按 L1 处理
LIGHT_MIN_SAMPLES = 5  # 历史不足时不亮灯 (回退绝对阈值)

STATE_NORMAL = "NORMAL"
STATE_YELLOW = "YELLOW_REVIEW"
STATE_RED_SIM = "RED_SIMULATE"  # L1: 降级模拟盘
STATE_HALT = "HALT"  # L3: 熔断停机
STATE_RETIRED = "RETIRED"  # Kill Switch 退役


@dataclass
class OOSMonitor:
    """OOS 监控状态机. 每日调用 daily_check(当日 Top15 预测 vs 实际收益)."""

    ic_history: list[float] = field(default_factory=list)  # 每日 Rank IC
    state: str = STATE_NORMAL
    _red_streak: int = 0
    _neg_streak: int = 0
    _yellow_streak: int = 0  # E4 连续黄灯计数

    # ---------------- 当日 IC ----------------
    @staticmethod
    def daily_rank_ic(pred_scores: pd.Series, actual_returns: pd.Series) -> float:
        """Top 15 清单预测得分 vs 次日实际收益的 Spearman IC."""
        df = pd.DataFrame({"s": pred_scores, "r": actual_returns}).dropna()
        if len(df) < 5:
            return 0.0
        return float(spearmanr(df["s"], df["r"]).statistic)

    # ---------------- E4: IC 日度三色灯 ----------------
    def ic_traffic_light(self, ic_today: float) -> str:
        """[E4] 基于滚动 20 日 μ/σ 的三色灯 (替代月度监控).

        🟢 IC > μ-1σ / 🟡 μ-2σ < IC < μ-1σ / 🔴 IC < μ-2σ (模型失效嫌疑 → L1).
        历史 < 5 日返回 "GREEN" (样本不足不亮灯, 由绝对阈值档位接管).
        """
        hist = self.ic_history[-LIGHT_WINDOW:]
        if len(hist) < LIGHT_MIN_SAMPLES:
            return "GREEN"
        mu, sigma = float(np.mean(hist)), float(np.std(hist))
        if ic_today < mu - 2 * sigma:
            return "RED"
        if ic_today < mu - 1 * sigma:
            return "YELLOW"
        return "GREEN"

    # ---------------- 每日检查 ----------------
    def daily_check(self, ic_today: float) -> dict:
        """输入当日 Rank IC, 返回 {'state', 'action', 'rolling_ic_5d', 'light'}.

        [E4] 三色灯优先: 🔴 → 立即 L1 降级; 🟡 连续 3 日 → 按 L1 处理.
        """
        light = self.ic_traffic_light(ic_today)
        self.ic_history.append(ic_today)
        rolling = float(np.mean(self.ic_history[-5:]))

        # E4 三色灯裁决 (优先于绝对阈值)
        if light == "RED":
            self._yellow_streak = 0
            self.state = STATE_RED_SIM
            action = "🔴 L1: IC < μ-2σ, 模型失效嫌疑, 立即降级模拟盘"
            logger.error(action)
            return {"state": self.state, "action": action,
                    "rolling_ic_5d": round(rolling, 4), "light": light}
        if light == "YELLOW":
            self._yellow_streak += 1
            if self._yellow_streak >= YELLOW_STREAK_L1:
                self.state = STATE_RED_SIM
                action = f"🟡×{self._yellow_streak} → L1: 连续黄灯, 降级模拟盘"
                logger.error(action)
                return {"state": self.state, "action": action,
                        "rolling_ic_5d": round(rolling, 4), "light": light}
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
        }

    # ---------------- Kill Switch ----------------
    def kill_switch_check(self) -> dict:
        """连续 2 个月滚动 20 日 IC 均值 < 0.01 → 模型退役."""
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
                    "排查原因 (数据源/特征/市场结构变化)",
                    "重新训练或调整特征",
                    "通过 OOS 验收后才可重新上线",
                ],
            }
        return {"retire": False, "month_ic": [round(m1, 4), round(m2, 4)]}
