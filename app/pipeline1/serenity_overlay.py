# -*- coding: utf-8 -*-
"""
Serenity 质化叠加层 (源自 serenity-skill, 瓶颈评分卡)
=====================================================
8 因子瓶颈评分 + 罚分体系, 用于清单后置质化过滤.
非量化特征, 不入 LightGBM; 作为 ListGenerator 的后置风险闸门.

用法:
    overlay = SerenityOverlay()
    adjusted = overlay.apply(daily_list, scorecard_data)
    # scorecard_data: {symbol: {demand_inflection: 0-5, ...}}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

WEIGHTS = {
    "demand_inflection": 15,
    "architecture_coupling": 10,
    "chokepoint_severity": 15,
    "supplier_concentration": 12,
    "expansion_difficulty": 12,
    "evidence_quality": 15,
    "valuation_disconnect": 11,
    "catalyst_timing": 10,
}

PENALTY_MULTIPLIER = 2.0
MIN_SERENITY_SCORE = 40  # < 40 的票从清单移除


@dataclass
class SerenityOverlay:
    """瓶颈评分卡叠加层 — 清单后置质化过滤."""

    min_score: float = MIN_SERENITY_SCORE

    def compute_score(self, factors: dict, penalties: dict) -> float:
        """计算综合得分: 加权因子 - 罚分×2."""
        raw = sum(factors.get(k, 0) * w for k, w in WEIGHTS.items())
        penalty = sum(penalties.values()) * PENALTY_MULTIPLIER
        return raw - penalty

    def apply(
        self,
        daily_list: pd.DataFrame,
        scorecard_data: dict[str, dict] | None = None,
    ) -> pd.DataFrame:
        """对清单施加质化过滤: < min_score 的票移除.

        Args:
            daily_list: ListGenerator.emit() 返回的清单 DataFrame
            scorecard_data: {symbol: {factors: {...}, penalties: {...}}}
                           None 时原样返回 (无质化数据则不过滤)
        Returns:
            过滤后的清单 (可能 < 15 只)
        """
        if scorecard_data is None or len(scorecard_data) == 0:
            return daily_list

        scores = {}
        for sym in daily_list["symbol"]:
            data = scorecard_data.get(sym)
            if data is None:
                scores[sym] = 100.0  # 无质化数据 → 放行
                continue
            scores[sym] = self.compute_score(
                data.get("factors", {}), data.get("penalties", {})
            )

        daily_list = daily_list.copy()
        daily_list["serenity_score"] = daily_list["symbol"].map(scores)
        before = len(daily_list)
        filtered = daily_list[daily_list["serenity_score"] >= self.min_score]
        removed = before - len(filtered)
        if removed > 0:
            logger.warning(
                "Serenity 叠加层: 移除 %d 只 (瓶颈评分 < %.0f)", removed, self.min_score
            )
        return filtered.reset_index(drop=True)
