# -*- coding: utf-8 -*-
"""IndicatorWeighter 单元测试 (P10.8, ARCH §5.13)."""

import numpy as np
import pandas as pd
import pytest

from app.core.indicator_weighter import IndicatorWeighter
from app.core.ths_indicators import THS_FACTOR_COLUMNS


@pytest.fixture()
def weighter() -> IndicatorWeighter:
    return IndicatorWeighter()


def _neutral_factors() -> dict:
    """全部 THS 因子置 0 的因子字典."""
    return {col: 0.0 for col in THS_FACTOR_COLUMNS}


class TestGroupDefinition:
    def test_eight_groups(self, weighter):
        assert len(IndicatorWeighter.INDICATOR_GROUPS) == 8

    def test_all_45_columns_mapped(self, weighter):
        mapped = [
            f for grp in IndicatorWeighter.INDICATOR_GROUPS.values()
            for f in grp["factors"]
        ]
        assert sorted(mapped) == sorted(THS_FACTOR_COLUMNS)

    def test_validate_weights(self, weighter):
        assert weighter.validate_weights() is True

    def test_default_group_weights(self, weighter):
        w = weighter.group_weights
        assert w["G1_main_force_chip"] == pytest.approx(0.30)
        assert w["G2_chip_control"] == pytest.approx(0.20)
        assert w["G3_bull_finder"] == pytest.approx(0.15)
        assert w["G4_trend_top_bottom"] == pytest.approx(0.15)
        for key in ("E1_vol_price", "E2_fund_flow",
                    "E3_ctrl_enhance", "E4_chip_dist"):
            assert w[key] == pytest.approx(0.05)

    def test_group_weights_summary(self, weighter):
        summary = weighter.get_group_weights_summary()
        assert len(summary) == 8
        assert summary["weight"].sum() == pytest.approx(1.0)
        assert set(summary.columns) >= {
            "group", "name", "weight", "num_factors", "factor_weight"}


class TestLayer1Scoring:
    def test_score_bounds(self, weighter):
        score = weighter.compute_indicator_score(_neutral_factors())
        assert 0.0 <= score <= 1.0

    def test_strong_signals_higher_score(self, weighter):
        weak = _neutral_factors()
        strong = _neutral_factors()
        strong.update({
            "tech_ths_entry_flag_decay10": 0.9,
            "tech_ths_pullup_flag_decay10": 0.8,
            "tech_ths_golden_cross_decay10": 0.8,
            "tech_ths_bull_ss_decay10": 0.9,
            "tech_ths_trend_bottom_decay10": 0.9,
            "tech_ths_ctrl_flag_decay10": 0.8,
        })
        assert (weighter.compute_indicator_score(strong)
                > weighter.compute_indicator_score(weak))

    def test_missing_factors_renormalized(self, weighter):
        partial = {"tech_ths_entry_flag_decay10": 1.0}
        score = weighter.compute_indicator_score(partial)
        assert score == pytest.approx(1.0)  # 唯一因子=1 → 重归一化后=1

    def test_final_score_mix(self, weighter):
        factors = _neutral_factors()
        ind = weighter.compute_indicator_score(factors)
        final = weighter.compute_final_score(0.8, factors)
        assert final == pytest.approx(0.6 * 0.8 + 0.4 * ind)

    def test_final_score_model_weight_from_config(self):
        w = IndicatorWeighter({"scoring_mix": {"model_weight": 0.5}})
        factors = _neutral_factors()
        ind = w.compute_indicator_score(factors)
        assert w.compute_final_score(0.8, factors) == pytest.approx(
            0.5 * 0.8 + 0.5 * ind)


class TestLayer2FeatureWeighting:
    def test_ths_weight_gt_non_ths(self, weighter):
        names = ["tech_ths_ctrl_ratio", "rsi_14", "tech_ths_entry", "close"]
        weights = weighter.get_feature_weights(names)
        assert weights[0] == weights[2]  # 同为 THS
        assert weights[1] == weights[3]  # 同为非 THS
        assert weights[0] > weights[1]

    def test_ths_boost_from_config(self):
        w = IndicatorWeighter({"factor_influence": {"ths_boost": 6.0}})
        names = ["tech_ths_ctrl_ratio", "rsi_14"]
        weights = w.get_feature_weights(names)
        ratio = weights[0] / weights[1]
        # sqrt 缩放后比例 = sqrt(6 / 归一化比) > 默认 3.6 的比例
        default_ratio = (IndicatorWeighter().get_feature_weights(names)[0]
                         / IndicatorWeighter().get_feature_weights(names)[1])
        assert ratio > default_ratio

    def test_weight_features_2d(self, weighter):
        names = ["tech_ths_ctrl_ratio", "rsi_14"]
        X = np.ones((5, 2))
        Xw = weighter.weight_features(X, names)
        assert Xw.shape == (5, 2)
        assert Xw[0, 0] > Xw[0, 1]

    def test_weight_features_3d(self, weighter):
        names = ["tech_ths_ctrl_ratio", "rsi_14", "close"]
        X = np.ones((4, 20, 3))
        Xw = weighter.weight_features(X, names)
        assert Xw.shape == (4, 20, 3)
        assert Xw[0, 0, 0] > Xw[0, 0, 1]

    def test_weight_features_dim_mismatch(self, weighter):
        with pytest.raises(ValueError):
            weighter.weight_features(np.ones((3, 2)), ["a", "b", "c"])


class TestLayer3Signals:
    def test_main_entry_flag_buy(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_entry_flag_decay10"] = 0.5
        sig = weighter.compute_indicator_signal(factors)
        assert sig["flags"]["main_entry"] is True
        assert sig["signal"] == "buy"
        assert "G1_main_force_chip" in sig["triggered_groups"]

    def test_pullup_flag_buy(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_pullup_flag_decay10"] = 0.5
        sig = weighter.compute_indicator_signal(factors)
        assert sig["flags"]["pullup"] is True
        assert sig["signal"] == "buy"

    def test_ship_flag_sell(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_ship_flag_decay10"] = 0.5
        sig = weighter.compute_indicator_signal(factors)
        assert sig["flags"]["ship"] is True
        assert sig["signal"] == "sell"

    def test_conflict_hold(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_entry_flag_decay10"] = 0.5
        factors["tech_ths_ship_flag_decay10"] = 0.5
        assert weighter.compute_indicator_signal(factors)["signal"] == "hold"

    def test_no_signal_hold(self, weighter):
        sig = weighter.compute_indicator_signal(_neutral_factors())
        assert sig["signal"] == "hold"
        assert sig["strength"] == 0.0

    def test_boost_both_buy(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_entry_flag_decay10"] = 0.8
        result = weighter.boost_trading_signal(1, 0.6, factors)
        assert result["signal"] == 1
        assert result["source"] == "both"
        assert result["strength"] == pytest.approx(0.6 * 0.6 + 0.4 * 0.8)

    def test_boost_conflict_hold(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_ship_flag_decay10"] = 0.8
        result = weighter.boost_trading_signal(1, 0.6, factors)
        assert result["signal"] == 0
        assert result["source"] == "conflict"

    def test_boost_model_only(self, weighter):
        result = weighter.boost_trading_signal(1, 0.6, _neutral_factors())
        assert result["signal"] == 1
        assert result["source"] == "model"
        assert result["strength"] == pytest.approx(0.6 * 0.7)

    def test_boost_both_sell(self, weighter):
        factors = _neutral_factors()
        factors["tech_ths_ship_flag_decay10"] = 0.8
        result = weighter.boost_trading_signal(-1, 0.6, factors)
        assert result["signal"] == -1
        assert result["source"] == "both"


class TestCtrlMa5RisingWeight:
    def test_rising_gives_positive_boost(self, weighter):
        series = pd.Series([0.30, 0.30, 0.30, 0.30, 0.33])
        boost = weighter.ctrl_ma5_rising_weight(series)
        assert boost > 0.0

    def test_boost_capped_at_020(self, weighter):
        series = pd.Series([0.10, 0.10, 0.10, 0.10, 0.50])
        boost = weighter.ctrl_ma5_rising_weight(series)
        assert boost == pytest.approx(0.20)

    def test_falling_gives_zero(self, weighter):
        series = pd.Series([0.40, 0.38, 0.36, 0.34, 0.32])
        assert weighter.ctrl_ma5_rising_weight(series) == 0.0

    def test_flat_gives_zero(self, weighter):
        series = pd.Series([0.30] * 10)
        assert weighter.ctrl_ma5_rising_weight(series) == 0.0

    def test_short_series_zero(self, weighter):
        assert weighter.ctrl_ma5_rising_weight(pd.Series([0.3, 0.4])) == 0.0


class TestNormalizeFactor:
    def test_decay_clipped(self):
        assert IndicatorWeighter.normalize_factor(
            1.5, "tech_ths_entry_flag_decay10") == pytest.approx(1.0)
        assert IndicatorWeighter.normalize_factor(
            -0.2, "tech_ths_entry_flag_decay10") == pytest.approx(0.0)

    def test_trend_line_scaled(self):
        assert IndicatorWeighter.normalize_factor(
            100.0, "tech_ths_trend_mid") == pytest.approx(0.5)

    def test_percentage_scaled(self):
        assert IndicatorWeighter.normalize_factor(
            50.0, "tech_ths_ctrl_low") == pytest.approx(0.5)

    def test_sigmoid_for_unbounded(self):
        assert IndicatorWeighter.normalize_factor(
            0.0, "tech_ths_vwap_dev") == pytest.approx(0.5)
        assert IndicatorWeighter.normalize_factor(
            10.0, "tech_ths_vwap_dev") > 0.99

    def test_nan_safe(self):
        assert IndicatorWeighter.normalize_factor(
            np.nan, "tech_ths_ctrl_ratio") == pytest.approx(0.0)
