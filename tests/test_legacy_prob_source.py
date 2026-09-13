# -*- coding: utf-8 -*-
"""LEGACY_PROB_SOURCE 概率源/闸头 (0912 夜: 每训动态自选) — 交付规则单测.

铁律对应: 交易规则改动必须同时更新单元测试。覆盖四面:
  1. V35Predictor: 显式 reg/cls 强制 > bundle["prob_source"] (重训 argmax
     落盘) > 旧 bundle 回退 FALLBACK (main=reg 残差 / dual=cls Platt);
     pred NaN 行回退 cls 不变。
  2. DualTrackTrainer.validate_oos: "auto" 默认 = argmax(weighted_ic_reg,
     weighted_ic_cls) 自选闸头 (用户令: cls vs reg 不固定, 每训复判);
     显式值强制。cls hard-label 不进 Rank IC。
  3. bundle 持久化: save() 落 prob_source, 旧包无键。
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
        out = _predict_with(
            _make_bundle(n, reg_out, cls_out), "dual", _make_features(n)
        )
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
        out = _predict_with(
            _make_bundle(n, reg_out, cls_out), "main", _make_features(n)
        )
        p_reg = out["prob_up_10d"].to_numpy()
        assert np.isfinite(p_reg).all()
        assert p_reg[0] > p_reg[1] > p_reg[4]  # pred 越大残差概率越高 (同向)
        assert p_reg[5] == pytest.approx(0.40)  # pred NaN 行回退 cls

    def test_rollback_knob_restores_reg_for_dual(self, monkeypatch):
        monkeypatch.setitem(
            __import__(
                "config.settings", fromlist=["LEGACY_PROB_SOURCE"]
            ).LEGACY_PROB_SOURCE,
            "dual",
            "reg",
        )
        n = 4
        reg_out = np.linspace(0.01, 0.05, n)
        cls_out = np.full(n, 0.30)
        out = _predict_with(
            _make_bundle(n, reg_out, cls_out), "dual", _make_features(n)
        )
        # 回退后 = main 同公式: 与 main 同输入同输出
        out_main = _predict_with(
            _make_bundle(n, reg_out, cls_out), "main", _make_features(n)
        )
        assert np.allclose(out["prob_up_10d"], out_main["prob_up_10d"])

    def test_auto_honors_bundle_recorded_choice(self):
        """auto 旋钮: bundle 记录的重训自选头生效 (压过板级 FALLBACK)."""
        n = 6
        reg_out = np.linspace(0.01, 0.06, n)
        cls_out = np.array([0.20, 0.35, 0.50, 0.60, 0.75, 0.90])
        b = _make_bundle(n, reg_out, cls_out)
        b["prob_source"] = "cls"
        out = _predict_with(b, "main", _make_features(n))
        assert np.allclose(
            out["prob_up_10d"], cls_out
        )  # main FALLBACK=reg 被包记录压过
        b2 = _make_bundle(n, reg_out, cls_out)
        b2["prob_source"] = "reg"
        out2 = _predict_with(b2, "dual", _make_features(n))
        out_main = _predict_with(
            _make_bundle(n, reg_out, cls_out), "main", _make_features(n)
        )
        assert np.allclose(
            out2["prob_up_10d"], out_main["prob_up_10d"]
        )  # dual FALLBACK=cls 被压过

    def test_forced_knob_overrides_bundle(self, monkeypatch):
        """显式强制旋钮最高优先: 压过 bundle 记录 (回滚语义)."""
        monkeypatch.setitem(
            __import__(
                "config.settings", fromlist=["LEGACY_PROB_SOURCE"]
            ).LEGACY_PROB_SOURCE,
            "main",
            "cls",
        )
        n = 4
        reg_out = np.linspace(0.01, 0.05, n)
        cls_out = np.array([0.20, 0.40, 0.60, 0.80])
        b = _make_bundle(n, reg_out, cls_out)
        b["prob_source"] = "reg"
        out = _predict_with(b, "main", _make_features(n))
        assert np.allclose(out["prob_up_10d"], cls_out)


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

    def test_auto_picks_winning_head_for_main(self):
        """[0912 夜] auto 默认: main 不再钉死 reg — argmax 自选 (cls 正 → cls)."""
        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("main"))
        assert oos["gate_head"] == "cls"
        assert oos["pass"] is True  # cls 头正 IC, reg 头全负也不拦

    def test_forced_knob_pins_reg_gate_for_main(self, monkeypatch):
        """显式强制 = 回滚旋钮: 钉死 reg 头闸判 (压过 argmax)."""
        monkeypatch.setitem(
            __import__(
                "config.settings", fromlist=["LEGACY_PROB_SOURCE"]
            ).LEGACY_PROB_SOURCE,
            "main",
            "reg",
        )
        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("main"))
        assert oos["gate_head"] == "reg"
        assert oos["pass"] is False  # (反相关) reg 头判死

    def test_gate_formula_matches_head_ics(self):
        from app.pipeline1.label_engine import LABEL_WEIGHTS

        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        oos = trainer.validate_oos(_make_trained("dual"))
        expected = sum(
            LABEL_WEIGHTS[k] * oos["ics"].get(f"{k}d_cls", 0.0) for k in LABEL_WEIGHTS
        ) / sum(LABEL_WEIGHTS.values())
        assert oos["weighted_ic"] == pytest.approx(expected, abs=1e-9)

    def test_both_head_family_aggregates_always_stored(self):
        """[09-12] cls/reg 双头族聚合恒存: 重训曾只落闸头聚合 → 另一头无档可查."""
        from app.pipeline1.label_engine import LABEL_WEIGHTS

        trainer = dtt.DualTrackTrainer.__new__(dtt.DualTrackTrainer)
        for board in ("main", "dual"):
            oos = trainer.validate_oos(_make_trained(board))
            for suffix in ("reg", "cls"):
                expected = sum(
                    LABEL_WEIGHTS[k] * oos["ics"].get(f"{k}d_{suffix}", 0.0)
                    for k in LABEL_WEIGHTS
                ) / sum(LABEL_WEIGHTS.values())
                assert oos[f"weighted_ic_{suffix}"] == pytest.approx(expected, abs=1e-9)
            assert oos["weighted_ic"] == pytest.approx(
                oos[f"weighted_ic_{oos['gate_head']}"], abs=1e-12
            )
            assert "threshold" in oos  # FAIL 行占位 '?' 修复
        # main=reg 闸帧: cls 聚合为正、reg 聚合为负 → 同一报告双头齐备
        oos_main = trainer.validate_oos(_make_trained("main"))
        assert oos_main["weighted_ic_cls"] > 0
        assert oos_main["weighted_ic_reg"] < 0


class TestProbSourcePersistence:
    def test_save_persists_prob_source(self, tmp_path):
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = _make_trained("dual")
        trained["calibrators"] = {}
        trained["calibrator"] = {}
        trained["prob_source"] = "cls"
        path = trainer.save(trained, "t1")
        b = dtt.DualTrackTrainer.load(path)
        assert b["prob_source"] == "cls"

    def test_save_without_prob_source_omits_key(self, tmp_path):
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = _make_trained("main")
        trained["calibrators"] = {}
        trained["calibrator"] = {}
        path = trainer.save(trained, "t2")
        b = dtt.DualTrackTrainer.load(path)
        assert "prob_source" not in b  # 旧包语义: 推理端走 FALLBACK
