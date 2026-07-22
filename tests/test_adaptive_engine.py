# -*- coding: utf-8 -*-
"""AdaptiveEngine 测试 (P10.12, ARCH §5.17)."""

import json
import os

import pytest
import yaml

from app.core.adaptive_engine import AdaptiveEngine

CONFIG_SPEC = {
    "scoring_mix": {
        "model_weight": {"initial": 0.6, "bounds": [0.3, 0.9]},
    },
    "trading_mix": {
        "model_weight": {"initial": 0.6, "bounds": [0.3, 0.9]},
    },
    "factor_influence": {
        "ths_boost": {"initial": 3.6, "bounds": [1.5, 6.0]},
    },
    "risk": {
        "max_drawdown": {"initial": 0.03, "bounds": [0.02, 0.08]},
        "ctrl_ratio_threshold": {"initial": 0.30, "bounds": [0.20, 0.50]},
        "buy_deadline": {"initial": "10:40", "candidates": ["10:30", "10:40", "11:00"]},
    },
}


@pytest.fixture
def paths(tmp_path):
    config_path = str(tmp_path / "adaptive_config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(CONFIG_SPEC, f, allow_unicode=True)
    state_path = str(tmp_path / "adaptive_state.json")
    return config_path, state_path


@pytest.fixture
def engine(paths):
    return AdaptiveEngine(config_path=paths[0], state_path=paths[1])


class TestInit:
    def test_initial_values_loaded(self, engine):
        cfg = engine.get_adaptive_config()
        assert cfg["scoring_mix"]["model_weight"] == pytest.approx(0.6)
        assert cfg["risk"]["max_drawdown"] == pytest.approx(0.03)
        assert cfg["risk"]["buy_deadline"] == "10:40"

    def test_state_persisted_and_reloaded(self, engine, paths):
        engine.compute_selection_mix({"selection_ic_gap": 1.0})
        assert os.path.exists(paths[1])
        engine2 = AdaptiveEngine(config_path=paths[0], state_path=paths[1])
        v1 = engine.get_adaptive_config()["scoring_mix"]["model_weight"]
        v2 = engine2.get_adaptive_config()["scoring_mix"]["model_weight"]
        assert v1 == pytest.approx(v2)
        assert v2 != pytest.approx(0.6)  # 已偏离 initial


class TestAdjust:
    def test_positive_gap_increases(self, engine):
        new = engine.compute_selection_mix({"selection_ic_gap": 0.5})
        # step = 10% * (0.9-0.3) = 0.06 → 0.66
        assert new == pytest.approx(0.66)

    def test_negative_gap_decreases(self, engine):
        new = engine.compute_selection_mix({"selection_ic_gap": -0.5})
        assert new == pytest.approx(0.54)

    def test_zero_gap_no_change(self, engine):
        new = engine.compute_selection_mix({"selection_ic_gap": 0.0})
        assert new == pytest.approx(0.6)

    def test_bounds_clamp_upper(self, engine):
        for _ in range(10):
            new = engine.compute_selection_mix({"selection_ic_gap": 1.0})
        assert new == pytest.approx(0.9)  # clamp 到上界

    def test_bounds_clamp_lower(self, engine):
        for _ in range(10):
            new = engine.compute_trading_mix({"trading_win_rate_gap": -1.0})
        assert new == pytest.approx(0.3)  # clamp 到下界

    def test_factor_influence_step(self, engine):
        new = engine.compute_factor_influence({"factor_ic_gap": 1.0})
        # step = 10% * (6.0-1.5) = 0.45 → 3.6+0.45 = 4.05
        assert new == pytest.approx(4.05)

    def test_risk_thresholds_group(self, engine):
        result = engine.compute_risk_thresholds({"risk_drawdown_gap": -1.0})
        # max_drawdown step = 10% * 0.06 = 0.006 → 0.024
        assert result["max_drawdown"] == pytest.approx(0.024)
        # ctrl_ratio step = 10% * 0.30 = 0.03 → 0.27
        assert result["ctrl_ratio_threshold"] == pytest.approx(0.27)
        # candidates 型参数跳过, 不出现在结果中且值不变
        assert "buy_deadline" not in result
        assert engine.get_adaptive_config()["risk"]["buy_deadline"] == "10:40"

    def test_risk_per_param_gap(self, engine):
        result = engine.compute_risk_thresholds({"risk_max_drawdown_gap": 1.0})
        assert result["max_drawdown"] == pytest.approx(0.036)
        assert result["ctrl_ratio_threshold"] == pytest.approx(0.30)  # 默认 gap=0


class TestRollback:
    def test_rollback_restores_previous(self, engine):
        engine.compute_selection_mix({"selection_ic_gap": 1.0})  # 0.6 → 0.66
        engine.compute_selection_mix({"selection_ic_gap": 1.0})  # 0.66 → 0.72
        assert engine.get_adaptive_config()["scoring_mix"][
            "model_weight"
        ] == pytest.approx(0.72)
        engine.rollback()
        assert engine.get_adaptive_config()["scoring_mix"][
            "model_weight"
        ] == pytest.approx(0.66)
        engine.rollback()
        assert engine.get_adaptive_config()["scoring_mix"][
            "model_weight"
        ] == pytest.approx(0.6)

    def test_rollback_empty_history_noop(self, engine):
        engine.rollback()  # 不应抛异常
        assert engine.get_adaptive_config()["scoring_mix"][
            "model_weight"
        ] == pytest.approx(0.6)

    def test_rollback_persisted(self, engine, paths):
        engine.compute_selection_mix({"selection_ic_gap": 1.0})
        engine.rollback()
        with open(paths[1], "r", encoding="utf-8") as f:
            state = json.load(f)
        assert state["current"]["scoring_mix"]["model_weight"] == pytest.approx(0.6)
        assert state["history"] == []
