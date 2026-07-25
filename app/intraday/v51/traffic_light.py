"""
红黄绿灯监控 (V5.1 §8, 检查清单 #14; 与 PIPELINE1 E4 三色灯联动)
======================================================================
绿灯: 20 日滚动夏普在历史均值 ±1σ 内 → 正常
黄灯: 低于均值 1σ 连续 3 日 → 复检队列 (联动 PIPELINE1 L1 复核)
红灯: 跌破 2σ → 规则下线, 触发紧急寻优
平原漂移检查: 参数高原中心漂移 > 20% → 系统级复盘
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

SHARPE_WINDOW = 20  # 滚动夏普窗口
YELLOW_DAYS = 3  # 连续黄灯天数 → 复检
MIN_HISTORY = 20  # 均值/σ 最少样本
PLATEAU_DRIFT_MAX = 0.20  # 平原中心漂移 >20% → 系统级复盘

LIGHT_GREEN, LIGHT_YELLOW, LIGHT_RED = "GREEN", "YELLOW", "RED"


@dataclass
class IntradayTrafficLight:
    """日内规则红黄绿灯状态机 (每日收盘后调用 daily_check)."""

    sharpe_history: list[float] = field(default_factory=list)  # 日度收益计算的夏普
    _yellow_streak: int = 0

    def daily_check(self, sharpe_20d: float) -> dict:
        """输入当日 20 日滚动夏普, 返回 {'light', 'action'}."""
        hist = self.sharpe_history[-SHARPE_WINDOW:]
        self.sharpe_history.append(sharpe_20d)
        if len(hist) < MIN_HISTORY:
            return {"light": LIGHT_GREEN, "action": "样本不足, 正常"}
        mu, sigma = float(np.mean(hist)), float(np.std(hist))
        if sharpe_20d < mu - 2 * sigma:
            self._yellow_streak = 0
            logger.critical(
                "红灯: 滚动夏普 %.3f < μ-2σ (%.3f), 规则下线, 紧急寻优",
                sharpe_20d,
                mu - 2 * sigma,
            )
            return {"light": LIGHT_RED, "action": "规则下线 + 紧急寻优"}
        if sharpe_20d < mu - 1 * sigma:
            self._yellow_streak += 1
            if self._yellow_streak >= YELLOW_DAYS:
                logger.error(
                    "黄灯×%d: 进入复检队列 (联动 PIPELINE1 L1)", self._yellow_streak
                )
                return {"light": LIGHT_YELLOW, "action": "复检队列 (L1 联动)"}
            return {"light": LIGHT_YELLOW, "action": "黄灯观察"}
        self._yellow_streak = 0
        return {"light": LIGHT_GREEN, "action": "正常"}


def plateau_drift(old_center: dict, new_center: dict) -> dict:
    """平原漂移检查: 参数高原中心漂移 > 20% → 系统级复盘.

    Args:
        old_center/new_center: {参数名: 平原中心值} (季度寻优产出)
    """
    drifted = {}
    for k, old in old_center.items():
        new = new_center.get(k)
        if new is None or old == 0:
            continue
        drift = abs(new - old) / abs(old)
        if drift > PLATEAU_DRIFT_MAX:
            drifted[k] = {"old": old, "new": new, "drift": round(drift, 4)}
    if drifted:
        logger.error("平原漂移: %s 漂移超 20%%, 系统级复盘", list(drifted))
    return {"drifted": drifted, "need_review": len(drifted) > 0}
