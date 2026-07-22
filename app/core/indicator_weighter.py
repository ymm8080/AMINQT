# -*- coding: utf-8 -*-
"""自定义指标加权器 (P10.8, ARCH §5.13).

为用户 REFERENCE/INDICATOR/ 4 个同花顺公式指标建立三层显式加权:
Layer 1 选股评分 / Layer 2 模型训练特征加权 / Layer 3 交易信号增强。

组→列映射与 ths_indicators.py 的 45 列 tech_ths_* 因子一一对齐
(ARCH §5.13.1 / §5.13.3)。
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from app.core.ths_indicators import THS_FACTOR_COLUMNS

logger = logging.getLogger(__name__)


def _resolve(value, default: float) -> float:
    """解析配置值: 支持标量或 {initial: x, bounds: [...]} 结构."""
    if isinstance(value, dict):
        return float(value.get("initial", default))
    if value is None:
        return float(default)
    return float(value)


class IndicatorWeighter:
    """自定义指标加权器 — 三层加权.

    8 组指标 (基础权重, 可被 AdaptiveWeighter 按个股覆盖):
        G1 主力筹码指标 (0.30) — 主力筹码指标.docx
        G2 筹码控盘程度 (0.20) — 主力筹码控盘程度N.docx
        G3 发现牛股     (0.15) — 发现牛股.docx
        G4 趋势顶底     (0.15) — 益盟趋势顶底新公式.docx
        E1 量价派生 / E2 资金流向 / E3 控盘增强 / E4 筹码分布 (各 0.05)
    """

    # ── 指标分组定义 (与 ths_indicators.py 45 列对齐, ARCH §5.13.3) ──
    INDICATOR_GROUPS: Dict[str, dict] = {
        "G1_main_force_chip": {
            "name": "主力筹码指标",
            "source": "主力筹码指标.docx",
            "weight": 0.30,
            "factors": [
                "tech_ths_trajectory",
                "tech_ths_mazl",
                "tech_ths_entry",
                "tech_ths_washout",
                "tech_ths_pullup",
                "tech_ths_ship",
                "tech_ths_entry_flag_decay10",
                "tech_ths_pullup_flag_decay10",
                "tech_ths_ship_flag_decay10",
                "tech_ths_golden_cross_decay10",
            ],
            "signal_factors": [
                "tech_ths_entry_flag_decay10",
                "tech_ths_pullup_flag_decay10",
                "tech_ths_ship_flag_decay10",
                "tech_ths_golden_cross_decay10",
            ],
        },
        "G2_chip_control": {
            "name": "筹码控盘程度",
            "source": "主力筹码控盘程度N.docx",
            "weight": 0.20,
            "factors": [
                "tech_ths_ctrl_low",
                "tech_ths_ctrl_mid",
                "tech_ths_ctrl_high",
                "tech_ths_ctrl_flag_decay10",
            ],
            "signal_factors": ["tech_ths_ctrl_flag_decay10"],
        },
        "G3_bull_finder": {
            "name": "发现牛股",
            "source": "发现牛股.docx",
            "weight": 0.15,
            "factors": [
                "tech_ths_ema3_dev_ema20",
                "tech_ths_ema7_dev_ema20",
                "tech_ths_ema12_dev_ema50",
                "tech_ths_bull_ss_decay10",
            ],
            "signal_factors": ["tech_ths_bull_ss_decay10"],
        },
        "G4_trend_top_bottom": {
            "name": "趋势顶底",
            "source": "同花顺益盟趋势顶底新公式.docx",
            "weight": 0.15,
            "factors": [
                "tech_ths_trend_short",
                "tech_ths_trend_mid",
                "tech_ths_trend_long",
                "tech_ths_trend_top_decay10",
                "tech_ths_trend_bottom_decay10",
                "tech_ths_trend_golden_decay10",
                "tech_ths_vol_price_divergence",
            ],
            "signal_factors": [
                "tech_ths_trend_top_decay10",
                "tech_ths_trend_bottom_decay10",
                "tech_ths_trend_golden_decay10",
            ],
        },
        "E1_vol_price": {
            "name": "量价派生",
            "source": "G1-G4通用派生",
            "weight": 0.05,
            "factors": [
                "tech_ths_vol_ratio",
                "tech_ths_vwap_dev",
                "tech_ths_obv_slope",
                "tech_ths_vol_price_corr",
                "tech_ths_vol_weighted_mtm",
            ],
            "signal_factors": [],
        },
        "E2_fund_flow": {
            "name": "主力资金流向",
            "source": "G1扩展",
            "weight": 0.05,
            "factors": [
                "tech_ths_flow_net",
                "tech_ths_flow_net_ma5",
                "tech_ths_flow_net_ma20",
                "tech_ths_flow_ratio",
                "tech_ths_flow_accum",
                "tech_ths_flow_divergence",
                "tech_ths_flow_strength",
                "tech_ths_flow_trend",
            ],
            "signal_factors": ["tech_ths_flow_divergence"],
        },
        "E3_ctrl_enhance": {
            "name": "控盘增强",
            "source": "G2扩展",
            "weight": 0.05,
            "factors": [
                "tech_ths_ctrl_ratio",
                "tech_ths_ctrl_concentration",
                "tech_ths_ctrl_change",
            ],
            "signal_factors": [],
        },
        "E4_chip_dist": {
            "name": "筹码分布增强",
            "source": "G2扩展",
            "weight": 0.05,
            "factors": [
                "tech_ths_chip_profit_ratio",
                "tech_ths_chip_concentration_20",
                "tech_ths_chip_cost_skew",
                "tech_ths_chip_low_high_ratio",
            ],
            "signal_factors": [],
        },
    }

    def __init__(self, config: dict = None) -> None:
        """加载权重配置; config 为 None 时使用 ARCH §5.13.1 默认权重.

        Args:
            config: 配置字典。支持键:
                - groups: {group_key: {"weight": w} 或 w} 组权重覆盖
                - groups.<key>.factor_weights_override: 组内因子权重覆盖
                - scoring_mix.model_weight: Layer 1 模型得分权重 (默认 0.6)
                - trading_mix.model_weight: Layer 3 模型信号权重 (默认 0.6)
                - factor_influence.ths_boost: Layer 2 THS 倍数 (默认 3.6)
                - signal_threshold: 信号触发阈值 (默认 0.3)
        """
        self.config = config or {}

        # 组权重 (默认 → config 覆盖)
        self.group_weights: Dict[str, float] = {
            key: grp["weight"] for key, grp in self.INDICATOR_GROUPS.items()
        }
        cfg_groups = self.config.get("groups", {}) or {}
        for key, val in cfg_groups.items():
            if key in self.group_weights:
                if isinstance(val, dict):
                    if "weight" in val:
                        self.group_weights[key] = float(val["weight"])
                else:
                    self.group_weights[key] = float(val)

        # 组内因子权重覆盖 (config → 默认均匀)
        self.factor_weight_overrides: Dict[str, Dict[str, float]] = {}
        for key, val in cfg_groups.items():
            if isinstance(val, dict) and "factor_weights_override" in val:
                self.factor_weight_overrides[key] = dict(val["factor_weights_override"])

        scoring_mix = self.config.get("scoring_mix", {}) or {}
        self.model_weight = _resolve(scoring_mix.get("model_weight"), 0.6)
        trading_mix = self.config.get("trading_mix", {}) or {}
        self.trading_model_weight = _resolve(trading_mix.get("model_weight"), 0.6)
        factor_influence = self.config.get("factor_influence", {}) or {}
        self.ths_boost = _resolve(factor_influence.get("ths_boost"), 3.6)
        self.signal_threshold = float(self.config.get("signal_threshold", 0.3))

        if not self.validate_weights():
            total = sum(self.group_weights.values())
            logger.warning("组权重总和 %.6f != 1.0, 自动归一化", total)
            if total > 0:
                self.group_weights = {
                    k: v / total for k, v in self.group_weights.items()
                }
        logger.info(
            "IndicatorWeighter 初始化: model_weight=%.2f ths_boost=%.2f",
            self.model_weight,
            self.ths_boost,
        )

    # ── Layer 1: 选股评分加权 ───────────────────────────────────────

    def compute_indicator_score(self, factors: dict) -> float:
        """Layer 1: 指标加权得分 (0~1).

        indicator_score = Σ(group_weight × group_score), 对缺失组重归一化。
        group_score = Σ(factor_weight × normalize(factor_value))。
        """
        return self._score_with_weights(factors, self.group_weights)

    def compute_final_score(self, model_score: float, factors: dict) -> float:
        """Layer 1: 融合模型得分 + 指标加权得分 → final_score.

        final_score = model_weight × model_score
                    + (1 - model_weight) × indicator_score。
        """
        indicator_score = self.compute_indicator_score(factors)
        final = (
            self.model_weight * float(model_score)
            + (1.0 - self.model_weight) * indicator_score
        )
        return float(np.clip(np.nan_to_num(final), 0.0, 1.0))

    # ── Layer 2: 模型训练特征加权 ───────────────────────────────────

    def get_feature_weights(self, all_factor_names: List[str]) -> np.ndarray:
        """Layer 2: 训练特征列权重向量 (THS 因子影响力 ≈ 3.6× 非THS).

        THS 列 raw weight = ths_boost, 非 THS 列 = 1.0; 归一化到均值 1 后
        开平方 (ARCH §5.13.2: X[:, i] × sqrt(weight), 避免方差过大)。

        Returns:
            np.ndarray, shape = (len(all_factor_names),)。
        """
        ths_set = set(THS_FACTOR_COLUMNS)
        raw = np.array(
            [self.ths_boost if name in ths_set else 1.0 for name in all_factor_names],
            dtype=float,
        )
        mean = raw.mean() if raw.size else 1.0
        weights = np.sqrt(raw / (mean if mean > 0 else 1.0))
        return np.nan_to_num(weights, nan=1.0)

    def weight_features(self, X: np.ndarray, factor_names: List[str]) -> np.ndarray:
        """Layer 2: 特征矩阵加权 (支持 (N, F) 与 (N, T, F)).

        Args:
            X: 特征矩阵, 最后一维 = len(factor_names)。
            factor_names: 特征列名。

        Returns:
            加权后的 X, 同 shape。
        """
        weights = self.get_feature_weights(factor_names)
        X = np.asarray(X, dtype=float)
        if X.shape[-1] != len(weights):
            raise ValueError(
                f"特征维度 {X.shape[-1]} 与 factor_names 长度 {len(weights)} 不一致"
            )
        return np.nan_to_num(X * weights, nan=0.0)

    # ── Layer 3: 交易信号加权 ───────────────────────────────────────

    def compute_indicator_signal(self, factors: dict) -> dict:
        """Layer 3: 指标信号 (主力进场/拉高/出货/金叉等).

        信号规则 (ARCH §5.13.3):
        - G1: entry/pullup/golden_cross > 阈值 → buy; ship > 阈值 → sell
        - G2: ctrl_flag > 阈值 → buy (高控盘)
        - G3: bull_ss > 阈值 → buy
        - G4: bottom/golden > 阈值 → buy; top > 阈值 → sell

        Returns:
            {signal, strength, boost, flags, triggered_groups, details}。
        """
        thr = self.signal_threshold

        def _v(name: str) -> float:
            return float(np.nan_to_num(factors.get(name, 0.0)))

        entry = _v("tech_ths_entry_flag_decay10")
        pullup = _v("tech_ths_pullup_flag_decay10")
        ship = _v("tech_ths_ship_flag_decay10")
        golden_cross = _v("tech_ths_golden_cross_decay10")
        ctrl_flag = _v("tech_ths_ctrl_flag_decay10")
        bull_ss = _v("tech_ths_bull_ss_decay10")
        trend_top = _v("tech_ths_trend_top_decay10")
        trend_bottom = _v("tech_ths_trend_bottom_decay10")
        trend_golden = _v("tech_ths_trend_golden_decay10")

        flags = {
            "main_entry": entry > thr or golden_cross > thr,
            "pullup": pullup > thr,
            "ship": ship > thr,
            "ctrl": ctrl_flag > thr,
            "bull_ss": bull_ss > thr,
            "bottom": trend_bottom > thr or trend_golden > thr,
            "top": trend_top > thr,
        }
        buy_flags = ["main_entry", "pullup", "ctrl", "bull_ss", "bottom"]
        sell_flags = ["ship", "top"]
        buy_hit = any(flags[f] for f in buy_flags)
        sell_hit = any(flags[f] for f in sell_flags)

        if buy_hit and not sell_hit:
            signal = "buy"
        elif sell_hit and not buy_hit:
            signal = "sell"
        else:
            signal = "hold"  # 含买卖同时触发 (冲突, 保守处理)

        # 强度 = 触发侧衰减信号最大值
        if signal == "buy":
            strength = max(
                entry,
                pullup,
                golden_cross,
                ctrl_flag,
                bull_ss,
                trend_bottom,
                trend_golden,
            )
        elif signal == "sell":
            strength = max(ship, trend_top)
        else:
            strength = 0.0
        strength = float(np.clip(strength, 0.0, 1.0))

        group_of_flag = {
            "main_entry": "G1_main_force_chip",
            "pullup": "G1_main_force_chip",
            "ship": "G1_main_force_chip",
            "ctrl": "G2_chip_control",
            "bull_ss": "G3_bull_finder",
            "bottom": "G4_trend_top_bottom",
            "top": "G4_trend_top_bottom",
        }
        triggered = sorted({group_of_flag[f] for f, on in flags.items() if on})

        return {
            "signal": signal,
            "strength": strength,
            "boost": strength * 0.4,  # 指标信号对交易信号的增强贡献上限
            "flags": flags,
            "triggered_groups": triggered,
            "details": {
                "G1_main_force_chip": {
                    "signal": "sell"
                    if flags["ship"]
                    else ("buy" if flags["main_entry"] or flags["pullup"] else "hold"),
                    "factors": {
                        "entry": entry,
                        "pullup": pullup,
                        "ship": ship,
                        "golden_cross": golden_cross,
                    },
                },
                "G4_trend_top_bottom": {
                    "signal": "sell"
                    if flags["top"]
                    else ("buy" if flags["bottom"] else "hold"),
                    "factors": {
                        "top": trend_top,
                        "bottom": trend_bottom,
                        "golden": trend_golden,
                    },
                },
            },
        }

    def boost_trading_signal(
        self, model_signal: int, model_strength: float, factors: dict
    ) -> dict:
        """Layer 3: 核心信号触发时增强交易信号.

        融合规则 (ARCH §5.13.3):
        - 同向 → 该方向, strength = model×0.6 + indicator×0.4, source='both'
        - 冲突 → hold, source='conflict'
        - 一方 hold → 另一方主导, strength ×0.7

        Args:
            model_signal: 模型信号 (1=buy, -1=sell, 0=hold)。
            model_strength: 模型信号强度 (0~1)。
            factors: 个股因子值字典。

        Returns:
            {signal, signal_name, strength, source,
             model_component, indicator_component}。
        """
        ind = self.compute_indicator_signal(factors)
        ind_signal = {"buy": 1, "sell": -1, "hold": 0}[ind["signal"]]
        ind_strength = ind["strength"]
        m_str = float(np.clip(np.nan_to_num(model_strength), 0.0, 1.0))
        w_m = self.trading_model_weight
        w_i = 1.0 - w_m

        model_component = w_m * m_str * int(model_signal)
        indicator_component = w_i * ind_strength * ind_signal

        if model_signal != 0 and ind_signal == model_signal:
            final_signal = int(model_signal)
            strength = w_m * m_str + w_i * ind_strength
            source = "both"
        elif model_signal != 0 and ind_signal != 0:
            final_signal = 0
            strength = 0.0
            source = "conflict"
        elif ind_signal == 0:
            final_signal = int(model_signal)
            strength = m_str * 0.7
            source = "model"
        else:
            final_signal = ind_signal
            strength = ind_strength * 0.7
            source = "indicator"

        return {
            "signal": final_signal,
            "signal_name": {1: "buy", -1: "sell", 0: "hold"}[final_signal],
            "strength": float(np.clip(strength, 0.0, 1.0)),
            "source": source,
            "model_component": model_component,
            "indicator_component": indicator_component,
        }

    # ── 控盘比例 5 日线上升显式加权 ─────────────────────────────────

    def ctrl_ma5_rising_weight(self, ctrl_ratio_series: pd.Series) -> float:
        """控盘比例 5 日线上升显式加权 (ARCH §5.13.8.F, DESIGN_V1 §7).

        Args:
            ctrl_ratio_series: tech_ths_ctrl_ratio 时间序列。

        Returns:
            权重加成系数: ctrl > MA5(ctrl) 时 min((ctrl-MA5)/MA5, 0.20), 否则 0。
            作用于 G2/E3 组权重, 归一化前叠加。
        """
        max_boost = _resolve(
            (self.config.get("ctrl_ratio", {}) or {}).get("ma5_rising_max_boost"), 0.20
        )
        if ctrl_ratio_series is None or len(ctrl_ratio_series) < 5:
            return 0.0
        series = pd.Series(
            np.nan_to_num(ctrl_ratio_series.astype(float).values),
            index=ctrl_ratio_series.index,
        )
        ctrl = float(series.iloc[-1])
        ma5 = float(series.rolling(5, min_periods=5).mean().iloc[-1])
        if np.isnan(ma5) or ma5 <= 0 or ctrl <= ma5:
            return 0.0
        return float(min((ctrl - ma5) / ma5, max_boost))

    # ── 工具方法 ────────────────────────────────────────────────────

    def get_group_weights_summary(self) -> pd.DataFrame:
        """返回 8 组权重汇总表 (看板展示用)."""
        rows = []
        for key, grp in self.INDICATOR_GROUPS.items():
            weight = self.group_weights[key]
            n = len(grp["factors"])
            rows.append(
                {
                    "group": key,
                    "name": grp["name"],
                    "source": grp["source"],
                    "weight": weight,
                    "num_factors": n,
                    "factor_weight": weight / n if n else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def validate_weights(self) -> bool:
        """校验 8 组基础权重总和 = 1.0."""
        return abs(sum(self.group_weights.values()) - 1.0) < 1e-6

    @staticmethod
    def normalize_factor(value: float, factor_name: str) -> float:
        """因子值归一化到 0~1 (按因子类型选择 min-max / sigmoid / clip).

        - 信号类 (decay10): 原值 0~1, clip
        - 趋势线类 (0~200): /200
        - 百分比类 (0~100): /100
        - 比率类 (0~1): clip
        - 轨迹类 (-100~100): (v+100)/200
        - 其他无界类: sigmoid
        """
        v = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        name = factor_name

        if "decay10" in name:
            return float(np.clip(v, 0.0, 1.0))
        if name in (
            "tech_ths_trend_short",
            "tech_ths_trend_mid",
            "tech_ths_trend_long",
        ):
            return float(np.clip(v / 200.0, 0.0, 1.0))
        if name in ("tech_ths_ctrl_low", "tech_ths_ctrl_mid", "tech_ths_ctrl_high"):
            return float(np.clip(v / 100.0, 0.0, 1.0))
        if name in (
            "tech_ths_ctrl_ratio",
            "tech_ths_ctrl_concentration",
            "tech_ths_chip_profit_ratio",
            "tech_ths_chip_concentration_20",
            "tech_ths_chip_cost_skew",
        ):
            return float(np.clip(v, 0.0, 1.0))
        if name in ("tech_ths_trajectory", "tech_ths_mazl"):
            return float(np.clip((v + 100.0) / 200.0, 0.0, 1.0))
        # 无界因子 (dev/corr/slope/flow/mtm/ratio>1 等) → sigmoid
        return float(1.0 / (1.0 + np.exp(-np.clip(v, -50.0, 50.0))))

    # ── 内部: 分组打分 ──────────────────────────────────────────────

    def _score_with_weights(
        self, factors: dict, group_weights: Dict[str, float]
    ) -> float:
        """按给定组权重计算指标加权得分 (0~1), 缺失组重归一化."""
        total_score = 0.0
        total_weight = 0.0
        for key, grp in self.INDICATOR_GROUPS.items():
            weight = float(group_weights.get(key, 0.0))
            if weight <= 0:
                continue
            present = [f for f in grp["factors"] if f in factors]
            if not present:
                continue
            overrides = self.factor_weight_overrides.get(key, {})
            if overrides:
                fw = np.array([float(overrides.get(f, 0.0)) for f in present])
                if fw.sum() <= 0:  # 覆盖权重全缺失 → 退回均匀
                    fw = np.ones(len(present))
            else:
                fw = np.ones(len(present))
            fw = fw / fw.sum()
            norm_vals = np.array(
                [self.normalize_factor(factors[f], f) for f in present]
            )
            group_score = float(np.dot(fw, norm_vals))
            total_score += weight * group_score
            total_weight += weight
        if total_weight <= 0:
            return 0.0
        return float(np.clip(np.nan_to_num(total_score / total_weight), 0.0, 1.0))
