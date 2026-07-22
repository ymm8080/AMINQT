# -*- coding: utf-8 -*-
"""AdaptiveWeighter 单元测试 (P10.8b, ARCH §5.13.7)."""

import numpy as np
import pandas as pd
import pytest

from app.core.adaptive_weighter import AdaptiveWeighter
from app.core.indicator_weighter import IndicatorWeighter
from app.core.ths_indicators import THS_FACTOR_COLUMNS


def _make_trend_df(n: int = 60, slope: float = 0.1,
                   amount: float = 1e8) -> pd.DataFrame:
    """构造已知趋势的合成日线."""
    rng = np.random.default_rng(7)
    t = np.arange(n)
    close = 10.0 + slope * t + rng.normal(0.0, 0.005, n)
    close = np.maximum(close, 0.5)
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": amount / close,
        "amount": np.full(n, amount),
    })


def _neutral_factors() -> dict:
    return {col: 0.0 for col in THS_FACTOR_COLUMNS}


@pytest.fixture()
def weighter() -> AdaptiveWeighter:
    return AdaptiveWeighter()


class TestPreFilter:
    def test_delegates_to_right_side_filter(self, weighter):
        assert weighter.pre_filter_uptrend(_make_trend_df(slope=0.1)) is True
        assert weighter.pre_filter_uptrend(_make_trend_df(slope=-0.1)) is False

    def test_batch_pre_filter(self, weighter):
        pool = {
            "UP": _make_trend_df(slope=0.1),
            "DOWN": _make_trend_df(slope=-0.1),
        }
        result = weighter.batch_pre_filter(pool)
        assert result == {"UP": True, "DOWN": False}

    def test_pre_filter_params_from_config(self):
        w = AdaptiveWeighter({
            "right_side_filter": {
                "ma_long": {"initial": 15},
                "min_amount": {"initial": 20000000},
            }
        })
        assert w.right_side_filter.ma_long == 15
        assert w.right_side_filter.min_amount == pytest.approx(2e7)


class TestAdaptiveWeights:
    def test_weights_sum_to_one(self, weighter):
        weights = weighter.compute_adaptive_weights(_neutral_factors())
        assert sum(weights.values()) == pytest.approx(1.0)
        assert len(weights) == 8

    def test_all_groups_present(self, weighter):
        weights = weighter.compute_adaptive_weights(_neutral_factors())
        assert set(weights) == set(IndicatorWeighter.INDICATOR_GROUPS)

    def test_strong_trend_boosts_g3(self, weighter):
        strong = _neutral_factors()
        strong["trend_strength"] = 3.0
        weak = _neutral_factors()
        weak["trend_strength"] = -3.0
        w_strong = weighter.compute_adaptive_weights(strong)
        w_weak = weighter.compute_adaptive_weights(weak)
        assert w_strong["G3_bull_finder"] > w_weak["G3_bull_finder"]
        assert w_strong["G4_trend_top_bottom"] > w_weak["G4_trend_top_bottom"]

    def test_high_ctrl_boosts_g2_e3(self, weighter):
        high_ctrl = _neutral_factors()
        high_ctrl["tech_ths_ctrl_ratio"] = 0.9
        low_ctrl = _neutral_factors()
        low_ctrl["tech_ths_ctrl_ratio"] = 0.05
        w_high = weighter.compute_adaptive_weights(high_ctrl)
        w_low = weighter.compute_adaptive_weights(low_ctrl)
        assert w_high["G2_chip_control"] > w_low["G2_chip_control"]
        assert w_high["E3_ctrl_enhance"] > w_low["E3_ctrl_enhance"]

    def test_main_force_signal_boosts_g1(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_entry_flag_decay10"] = 1.0
        factors["tech_ths_pullup_flag_decay10"] = 1.0
        w_mf = weighter.compute_adaptive_weights(factors)
        w_none = weighter.compute_adaptive_weights(_neutral_factors())
        assert w_mf["G1_main_force_chip"] > w_none["G1_main_force_chip"]

    def test_nan_factors_safe(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_ctrl_ratio"] = np.nan
        weights = weighter.compute_adaptive_weights(factors)
        assert sum(weights.values()) == pytest.approx(1.0)


class TestFinalScore:
    def test_non_uptrend_fallback(self, weighter):
        """非上行: final_score = model_score * 0.8, 指标得分=0."""
        result = weighter.compute_final_score(
            model_score=0.7, factors=_neutral_factors(),
            is_uptrend=False, adaptive_weights=None)
        assert result["final_score"] == pytest.approx(0.7 * 0.8)
        assert result["indicator_score"] == 0.0
        assert result["is_uptrend"] is False
        assert all(v == 0.0 for v in result["adaptive_weights"].values())
        assert result["tag"] == "非上行-仅模型得分"

    def test_uptrend_mix(self, weighter):
        factors = _neutral_factors()
        result = weighter.compute_final_score(
            model_score=0.7, factors=factors,
            is_uptrend=True, adaptive_weights=None)
        expected = 0.6 * 0.7 + 0.4 * result["indicator_score"]
        assert result["final_score"] == pytest.approx(expected)
        assert result["is_uptrend"] is True
        assert sum(result["adaptive_weights"].values()) == pytest.approx(1.0)

    def test_uptrend_with_explicit_weights(self, weighter):
        factors = _neutral_factors()
        weights = weighter.compute_adaptive_weights(factors)
        result = weighter.compute_final_score(
            0.7, factors, is_uptrend=True, adaptive_weights=weights)
        assert result["adaptive_weights"] == weights

    def test_final_score_bounds(self, weighter):
        result = weighter.compute_final_score(
            1.5, _neutral_factors(), is_uptrend=True, adaptive_weights=None)
        assert 0.0 <= result["final_score"] <= 1.0


class TestExplainWeights:
    def test_explain_structure(self, weighter):
        report = weighter.explain_weights(_neutral_factors())
        assert "base_weights" in report
        assert "adaptive_weights" in report
        assert "adjustments" in report
        assert report["base_weights"]["G1_main_force_chip"] == pytest.approx(0.30)
        assert sum(report["adaptive_weights"].values()) == pytest.approx(1.0)
        assert len(report["adjustments"]) > 0

    def test_explain_reasons_reference_groups(self, weighter):
        factors = _neutral_factors()
        factors["trend_strength"] = 3.0
        report = weighter.explain_weights(factors)
        g3_adj = [a for a in report["adjustments"]
                  if a["group"] == "G3_bull_finder"
                  and a["factor"] == "trend_strength"]
        assert len(g3_adj) == 1
        assert g3_adj[0]["boost"] > 0.2  # sigmoid(3)*0.5 ≈ 0.476

    def test_adjustment_override_from_config(self):
        w = AdaptiveWeighter({"adjustment": {"trend_strength_boost": 0.9}})
        factors = _neutral_factors()
        factors["trend_strength"] = 3.0
        report = w.explain_weights(factors)
        g3_adj = [a for a in report["adjustments"]
                  if a["group"] == "G3_bull_finder"
                  and a["factor"] == "trend_strength"]
        assert g3_adj[0]["boost"] == pytest.approx(0.9 * (1 / (1 + np.exp(-3))))
