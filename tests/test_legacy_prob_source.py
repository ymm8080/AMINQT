# -*- coding: utf-8 -*-
"""LEGACY_PROB_SOURCE 板级概率源/闸头 (0912 dual 切 cls) — 交付规则单测.

铁律对应: 交易规则改动必须同时更新单元测试。覆盖三面:
  1. V35Predictor: dual prob_up_kd = cls predict_proba (校准直通);
     main 保持 reg 残差派生 (pred NaN 行回退 cls); 旋钮可回退。
  2. DualTrackTrainer.validate_oos: 闸头随板切换 (dual=cls 经 predict_proba,
     main=reg); cls hard-label 不进 Rank IC。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.pipeline1.dual_track_trainer as dtt
from app.pipeline1.predictor import V35Predictor


class _FakeReg:
    def __init__(self, out):
        self.out = np.asarray(out, dtype=float)

    def predict(self, X):
        return self.out[: len(X)]


class _FakeCls:
    def __init__(self, out):
        self.out = np.asarray(out, dtype=float)

    def predict_proba(self, X):
        p = np.clip(self.out[: len(X)], 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.out[: len(X)] > 0.5).astype(int)


def _make_features(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"{600000 + i:06d}" for i in range(n)],
            "date": pd.Timestamp("2026-09-12"),
            "board": "dual",
            "industry": "T",
            "f1": np.linspace(-1.0, 1.0, n),
        }
    )


def _make_bundle(n: int, reg_out: np.ndarray, cls_out: np.ndarray) -> dict:
    models = {}
    for k in (3, 5, 10):
        models[f"{k}d_reg"] = (_FakeReg(reg_out), f"label_{k}d")
        models[f"{k}d_cls"] = (_FakeCls(cls_out), f"label_{k}d_cls")
    return {
        "feature_cols": ["f1"],
        "models": models,
        "calibrators": {},
        # 残差数组: 与 pred 同序单调 → p_reg 与 pred_ret 同向, 与 p_cls 可区分
        "reg_resid_3d": np.linspace(-0.05, 0.05, 200),
        "reg_resid_5d": np.linspace(-0.05, 0.05, 200),
        "reg_resid_10d": np.linspace(-0.05, 0.05, 200),
    }


def _predict_with(bundle: dict, board: str, features: pd.DataFrame) -> pd.DataFrame:
    p = V35Predictor.__new__(V35Predictor)
    p.bundles = {board: bundle}
    return p.predict(features, board)


class TestPredictorProbSource:
    def test_dual_uses_cls_proba(self):
        n = 6
        reg_out = np.linspace(0.01, 0.06, n)
        cls_out = np.array([0.20, 0.35, 0.50, 0.60, 0.75, 0.90])
        out = _predict_with(_make_bundle(n, reg_out, cls_out), "dual", _make_features(n))
        assert np.allclose(out["prob_up_10d"], cls_out)
        assert np.allclose(out["prob_up_3d"], cls_out)  # prob_up 别名同源
        # dual 排序 ≡ cls 序, 与 (反向的) pred_ret 序脱钩
        assert list(out.nlargest(3, "prob_up_10d")["symbol"]) == [
            f"{600000 + i:06d}" for i in (5, 4, 3)
        ]

    def test_main_keeps_reg_resid_with_cls_nan_fallback(self):
        n = 6
        reg_out = np.array([0.06, 0.05, 0.04, 0.03, 0.02, np.nan])
        cls_out = np.full(n, 0.40)
        out = _predict_with(_make_bundle(n, reg_out, cls_out), "main", _make_features(n))
        p_reg = out["prob_up_10d"].to_numpy()
        assert np.isfinite(p_reg).all()
        assert p_reg[0] > p_reg[1] > p_reg[4]  # pred 越大残差概率越高 (同向)
        assert p_reg[5] == pytest.approx(0.40)  # pred NaN 行回退 cls

    def test_rollback_knob_restores_reg_for_dual(self, monkeypatch):
        monkeypatch.setitem(
            __import__("config.settings", fromlist=["LEGACY_PROB_SOURCE"]).LEGACY_PROB_SOURCE,
            "dual",
            "reg",
        )
        n = 4
        reg_out = np.linspace(0.01, 0.05, n)
        cls_out = np.full(n, 0.30)
        out = _predict_with(_make_bundle(n, reg_out, cls_out), "dual", _make_features(n))
        # 回退后 = main 同公式: 与 main 同输入同输出
        out_main = _predict_with(_make_bundle(n, reg_out, cls_out), "main", _make_features(n))
        assert np.allclose(out["prob_up_10d"], out_main["prob_up_10d"])


def _make_trained(board: str) -> dict:
    """假模型帧: cls 概率与标签正相关 (IC>0), reg 与标签反相关 (IC<0)."""
    rng = np.random.default_rng(7)
    n_days, n_sym = 12, 12
    dates = pd.date_range("2026-08-01", periods=n_days, freq="D")
    df = pd.DataFrame(
        {
            "symbol": [f"{300000 + i:06d}" for i in range(n_sym)] * n_days,
            "date": np.repeat(dates.values, n_sym),
            "f1": rng.normal(size=n_days * n_sym),
        }
    )
    latent = df["f1"].to_numpy()
    df["label_3d"] = latent + rng.normal(scale=0.05, size=len(df))
    df["label_5d"] = latent + rng.normal(scale=0.05, size=len(df))
    df["label_10d"] = latent + rng.normal(scale=0.05, size=len(df))
    for k in (3, 5, 10):
        df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0).astype(float)
    models = {}
    for k in (3, 5, 10):
        models[f"{k}d_reg"] = (_FakeReg(-latent), f"label_{k}d")  # 反相关 → IC 负
        models[f"{k}d_cls"] = (_FakeCls((latent + 5) / 10), f"label_{k}d_cls")  # 正相关
    return {
        "board": board,
        "segs": {"test": df},
        "models": models,
        "feature_cols": ["f1"],
    }


class TestValidateOosGateHead:
    def test_dual_gate_uses_cls_heads(self):
        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("dual"))
        assert oos["gate_head"] == "cls"
        assert oos["pass"] is True  # cls 头正 IC, reg 头全负也不拦
        assert oos["ics"]["10d_cls"] > 0.3
        assert oos["ics"]["10d_reg"] < -0.3  # 残差头反相关被如实记录

    def test_main_gate_keeps_reg_heads(self):
        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("main"))
        assert oos["gate_head"] == "reg"
        assert oos["pass"] is False  # main 闸仍由 (反相关) reg 头判死

    def test_gate_formula_matches_head_ics(self):
        from app.pipeline1.label_engine import LABEL_WEIGHTS

        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("dual"))
        expected = (
            sum(LABEL_WEIGHTS[k] * oos["ics"].get(f"{k}d_cls", 0.0) for k in LABEL_WEIGHTS)
            / sum(LABEL_WEIGHTS.values())
        )
        assert oos["weighted_ic"] == pytest.approx(expected, abs=1e-9)
