# -*- coding: utf-8 -*-
"""按个股自适应权重器 (P10.8b, ARCH §5.13.7, DESIGN_V1 §9 #1).

每只股票独立计算 8 组指标权重 (非全局统一), 内部委托 RightSideFilter
做右侧交易预筛选 — 非上行股票权重=0, 不参与指标评分。
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from app.core.indicator_weighter import IndicatorWeighter, _resolve
from app.core.right_side_filter import RightSideFilter

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """sigmoid 归一化到 0~1 (数值截断防溢出)."""
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


class AdaptiveWeighter:
    """按个股自适应权重器.

    工作流程:
        1. pre_filter_uptrend(): 委托 RightSideFilter, 仅对上行股票继续
        2. compute_adaptive_weights(): 按个股特征计算 8 组权重 (sigmoid 增强)
        3. compute_indicator_score(): 自适应权重计算指标得分
        4. compute_final_score(): 融合模型得分 + 自适应指标得分

    非上行股票: 权重=0, final_score = model_score * 0.8, 标记"非上行"。
    """

    # 自适应增强目标组与最大增强倍数 (ARCH §5.13.7.C / §5.13.7.E)
    # factor -> [(group, max_boost, sigmoid_input_scale)]
    BOOST_RULES: Dict[str, List[tuple]] = {
        "trend_strength": [
            ("G3_bull_finder", 0.50, 1.0),
            ("G4_trend_top_bottom", 0.30, 1.0),
        ],
        "ctrl_ratio": [
            ("G2_chip_control", 0.40, 3.0),
            ("E3_ctrl_enhance", 0.30, 3.0),
        ],
        "flow_strength": [
            ("E2_fund_flow", 0.50, 2.0),
        ],
        "vol_price_corr": [
            ("E1_vol_price", 0.30, 1.0),
        ],
        "chip_concentration": [
            ("E4_chip_dist", 0.30, 1.0),
            ("G2_chip_control", 0.20, 1.0),
        ],
        "volatility": [
            ("G4_trend_top_bottom", 0.20, 10.0),
        ],
        "main_force_signal": [
            ("G1_main_force_chip", 0.30, 1.0),
        ],
    }

    # boost 因子名 → factors 字典中的提取键 (缺省回退)
    FACTOR_KEYS: Dict[str, List[str]] = {
        "trend_strength": ["trend_strength"],
        "ctrl_ratio": ["tech_ths_ctrl_ratio"],
        "flow_strength": ["tech_ths_flow_strength"],
        "vol_price_corr": ["tech_ths_vol_price_corr"],
        "chip_concentration": ["tech_ths_chip_concentration_20"],
        "volatility": ["volatility"],
        "main_force_signal": ["main_force_signal"],
    }

    def __init__(self, config: dict = None) -> None:
        """加载预筛选阈值与自适应调整参数, 内部创建 RightSideFilter.

        Args:
            config: 配置字典。支持键:
                - right_side_filter / pre_filter: MA 周期/最低成交额
                  (支持 {initial: x} 结构)
                - adjustment: {factor_boost: 倍数} 覆盖 BOOST_RULES 最大增强
                - scoring_mix.model_weight: 模型得分权重 (默认 0.6)
                - groups: 组权重覆盖 (透传 IndicatorWeighter)
        """
        self.config = config or {}

        # 右侧预筛选器 (参数可被配置覆盖)
        rs_cfg = (self.config.get("right_side_filter")
                  or self.config.get("pre_filter") or {})
        self.right_side_filter = RightSideFilter(
            ma_short=int(_resolve(rs_cfg.get("ma_short"), 5)),
            ma_mid=int(_resolve(rs_cfg.get("ma_mid"), 10)),
            ma_long=int(_resolve(rs_cfg.get("ma_long"), 20)),
            min_amount=_resolve(rs_cfg.get("min_amount"), 50_000_000.0),
            require_market_above_ma20=bool(
                rs_cfg.get("market_above_ma20", False)
            ),
        )

        # 指标加权器 (复用组定义/归一化/组内打分)
        self.indicator_weighter = IndicatorWeighter(self.config)
        self.base_weights: Dict[str, float] = dict(
            self.indicator_weighter.group_weights
        )

        # 自适应增强倍数覆盖
        adj_cfg = self.config.get("adjustment", {}) or {}
        self.boost_rules: Dict[str, List[tuple]] = {
            factor: [
                (group, float(adj_cfg.get(f"{factor}_boost", max_boost)), scale)
                for group, max_boost, scale in rules
            ]
            for factor, rules in self.BOOST_RULES.items()
        }

        scoring_mix = self.config.get("scoring_mix", {}) or {}
        self.model_weight = _resolve(scoring_mix.get("model_weight"), 0.6)
        # 非上行股票得分折损系数 (ARCH §5.13.7.D)
        self.non_uptrend_discount = float(
            self.config.get("non_uptrend_discount", 0.8)
        )
        logger.info("AdaptiveWeighter 初始化: model_weight=%.2f", self.model_weight)

    # ── 右侧交易预筛选 ──────────────────────────────────────────────

    def pre_filter_uptrend(self, stock_df: pd.DataFrame,
                           market_above_ma20: bool = True) -> bool:
        """右侧交易预筛选 (委托 RightSideFilter.is_uptrend)."""
        return self.right_side_filter.is_uptrend(stock_df, market_above_ma20)

    def batch_pre_filter(self, all_stocks: Dict[str, pd.DataFrame],
                         market_above_ma20: bool = True) -> Dict[str, bool]:
        """批量预筛选, 返回 {symbol: is_uptrend}."""
        return self.right_side_filter.batch_filter(all_stocks, market_above_ma20)

    # ── 按个股自适应权重计算 ────────────────────────────────────────

    def compute_adaptive_weights(self, factors: dict) -> Dict[str, float]:
        """对单只上行股票计算 8 组自适应权重.

        Args:
            factors: 个股因子值字典 (85 维)。

        Returns:
            {G1_main_force_chip: w1, ..., E4_chip_dist: w8}, 总和=1.0。
            增强逻辑: 趋势强度→G3/G4, 控盘→G2/E3, 资金流→E2,
            量价→E1, 筹码→E4/G2, 波动率→G4, 主力信号→G1 (ARCH §5.13.7.C)。
        """
        weights = dict(self.base_weights)
        for factor_name, rules in self.boost_rules.items():
            value = self._extract_factor(factors, factor_name)
            for group, max_boost, scale in rules:
                boost = _sigmoid(value * scale)
                weights[group] = weights.get(group, 0.0) * (1.0 + max_boost * boost)

        total = sum(weights.values())
        if total <= 0:
            return {k: 0.0 for k in self.base_weights}
        return {k: v / total for k, v in weights.items()}

    def compute_indicator_score(self, factors: dict,
                                adaptive_weights: Dict[str, float] = None
                                ) -> float:
        """使用自适应权重计算指标加权得分 (0~1).

        Args:
            factors: 个股因子值字典。
            adaptive_weights: 自适应组权重; None 时用基础权重。
        """
        weights = adaptive_weights or self.base_weights
        return self.indicator_weighter._score_with_weights(factors, weights)

    def compute_final_score(self, model_score: float, factors: dict,
                            is_uptrend: bool,
                            adaptive_weights: Dict[str, float]) -> dict:
        """融合模型得分 + 自适应指标得分.

        上行股票: final = model_weight×model + (1-model_weight)×indicator。
        非上行股票: final = model_score × 0.8 (不纳入指标加权)。

        Returns:
            {final_score, model_score, indicator_score,
             is_uptrend, adaptive_weights, weight_adjustments, tag}
        """
        model_score = float(np.clip(np.nan_to_num(model_score), 0.0, 1.0))

        if not is_uptrend:
            return {
                "final_score": float(model_score * self.non_uptrend_discount),
                "model_score": model_score,
                "indicator_score": 0.0,
                "is_uptrend": False,
                "adaptive_weights": {k: 0.0 for k in self.base_weights},
                "weight_adjustments": [],
                "tag": "非上行-仅模型得分",
            }

        weights = adaptive_weights or self.compute_adaptive_weights(factors)
        indicator_score = self.compute_indicator_score(factors, weights)
        final = (self.model_weight * model_score
                 + (1.0 - self.model_weight) * indicator_score)
        return {
            "final_score": float(np.clip(final, 0.0, 1.0)),
            "model_score": model_score,
            "indicator_score": float(indicator_score),
            "is_uptrend": True,
            "adaptive_weights": weights,
            "weight_adjustments": self._collect_adjustments(factors),
            "tag": "上行",
        }

    # ── 权重可解释性 ────────────────────────────────────────────────

    def explain_weights(self, factors: dict) -> dict:
        """返回权重调整可解释性报告 (基础权重/自适应权重/调整原因)."""
        adaptive = self.compute_adaptive_weights(factors)
        return {
            "base_weights": dict(self.base_weights),
            "adaptive_weights": adaptive,
            "adjustments": self._collect_adjustments(factors),
        }

    # ── 内部工具 ────────────────────────────────────────────────────

    def _extract_factor(self, factors: dict, factor_name: str) -> float:
        """从 factors 提取 boost 因子值 (多键回退, NaN→0)."""
        for key in self.FACTOR_KEYS.get(factor_name, [factor_name]):
            if key in factors:
                return float(np.nan_to_num(factors[key]))
        # main_force_signal 缺省时由 G1 衰减信号合成
        if factor_name == "main_force_signal":
            entry = float(np.nan_to_num(
                factors.get("tech_ths_entry_flag_decay10", 0.0)))
            pullup = float(np.nan_to_num(
                factors.get("tech_ths_pullup_flag_decay10", 0.0)))
            golden = float(np.nan_to_num(
                factors.get("tech_ths_golden_cross_decay10", 0.0)))
            return 0.4 * entry + 0.3 * pullup + 0.3 * golden
        return 0.0

    def _collect_adjustments(self, factors: dict) -> List[dict]:
        """收集各 boost 因子的调整明细 (可解释性)."""
        reason_map = {
            "trend_strength": "趋势强度高→{group}权重增强",
            "ctrl_ratio": "控盘比例高→{group}权重增强",
            "flow_strength": "资金持续流入→{group}权重增强",
            "vol_price_corr": "量价配合好→{group}权重增强",
            "chip_concentration": "筹码集中→{group}权重增强",
            "volatility": "波动率大→{group}权重增强",
            "main_force_signal": "主力信号明确→{group}权重增强",
        }
        adjustments: List[dict] = []
        for factor_name, rules in self.boost_rules.items():
            value = self._extract_factor(factors, factor_name)
            for group, max_boost, scale in rules:
                boost = _sigmoid(value * scale)
                adjustments.append({
                    "group": group,
                    "factor": factor_name,
                    "value": value,
                    "reason": reason_map[factor_name].format(group=group),
                    "boost": float(max_boost * boost),
                })
        return adjustments
