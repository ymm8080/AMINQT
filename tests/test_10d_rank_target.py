# -*- coding: utf-8 -*-
"""Tests for 10d 头秩目标变换 (LEGACY_10D_RANK_TARGET, 2026-09-10, 688228 案).

背景: 纯幅度 Huber 目标下 10d 头顶部承诺虚高 ~−10pp (匹配视界审计), 排名键
恰好取到最虚高的承诺 → 四连阴股入 TOP10。修复 = 训练目标改 per-date 截面
百分位 + 训练段桶中位映射回收益语义 (bundle["10d_rank_map"])。

覆盖: 门控开关 / _train_one 秩目标分支 (真实训练, 断言目标 ∈ [0,1] 且
per-date 归一) / fit_rank_map (单调 + 空桶插值 + 顶桶中位) / rank_map_apply
(空映射透传 + 插值 + 裁剪) / fit_calibrator 10d 残差在收益语义空间 /
V35Predictor 百分位→收益映射 (旧 bundle no-op)。
"""

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.dual_track_trainer import (
    DualTrackTrainer,
    fit_rank_map,
    rank_map_apply,
    rank_target_enabled,
)
from app.pipeline1.predictor import V35Predictor


class _StubReg:
    """predict 返回预定数组的 stub 回归头 (输出=百分位 0..1)."""

    def __init__(self, out):
        self.out = np.asarray(out, dtype=float)

    def predict(self, X):
        return self.out[: len(X)]


class _StubCls:
    """predict_proba 返回常数列的 stub 分类头 (calibrators 空时原样透传)."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


def _patch_cfg(monkeypatch, **kw):
    cfg = {"enable": False, "bins": 20, "boards": ["main", "dual"]}
    cfg.update(kw)
    monkeypatch.setattr("config.settings.LEGACY_10D_RANK_TARGET", cfg)
    return cfg


# ── 门控 ──────────────────────────────────────────────────────
def test_gate_off_by_default(monkeypatch):
    _patch_cfg(monkeypatch, enable=False)
    assert not rank_target_enabled("main")
    assert not rank_target_enabled("dual")


def test_gate_board_filter(monkeypatch):
    _patch_cfg(monkeypatch, enable=True, boards=["main"])
    assert rank_target_enabled("main")
    assert not rank_target_enabled("dual")


# ── _train_one 秩目标分支 (真实 LightGBM, 迷你数据) ────────────
def _mk_segs():
    rows = []
    labels = [-0.05, 0.0, 0.03, 0.10]
    for d in ("2026-01-05", "2026-01-06", "2026-01-07"):
        for i in range(4):
            rows.append(
                {
                    "date": d,
                    "symbol": f"{600000 + i}.SH",
                    "f1": float(i) + 0.1 * labels.index(labels[i]),
                    "label_10d_net": labels[i],
                }
            )
    tr = pd.DataFrame(rows)
    return {"train": tr.copy(), "es": tr.copy()}


def test_train_one_rank_target_output_in_unit_interval(monkeypatch, tmp_path):
    _patch_cfg(monkeypatch, enable=True, bins=20)
    trainer = DualTrackTrainer(model_dir=str(tmp_path))
    model, label = trainer._train_one("10d_reg", _mk_segs(), ["f1"], "main")
    seg = _mk_segs()["train"].dropna(subset=[label])
    p = np.asarray(model.predict(seg[["f1"]].to_numpy(dtype=float)))
    assert np.nanmin(p) >= 0.0 - 1e-6
    assert np.nanmax(p) <= 1.0 + 1e-6


def test_train_one_off_returns_raw_scale(monkeypatch, tmp_path):
    _patch_cfg(monkeypatch, enable=False)
    trainer = DualTrackTrainer(model_dir=str(tmp_path))
    model, label = trainer._train_one("10d_reg", _mk_segs(), ["f1"], "main")
    seg = _mk_segs()["train"].dropna(subset=[label])
    p = np.asarray(model.predict(seg[["f1"]].to_numpy(dtype=float)))
    # 幅度目标: 输出量级 = 标签量级 (百分位量级为 0..1, 幅度可超 ±0.1)
    assert np.nanmax(np.abs(p)) > 0.01


# ── fit_rank_map / rank_map_apply ─────────────────────────────
def test_fit_rank_map_monotone_and_top_bucket_median():
    n = 400
    rng = np.random.RandomState(7)
    p = np.sort(rng.uniform(0, 1, n))
    labels = -0.05 + 0.2 * p + rng.normal(0, 0.001, n)  # 单调生成
    seg = pd.DataFrame({"f1": p, "label_10d_net": labels})
    model = _StubReg(p)
    rmap = fit_rank_map(model, seg, "label_10d_net", ["f1"], bins=20)
    gr = np.asarray(rmap["grid_ret"])
    assert len(gr) == 20
    assert np.all(np.diff(gr) >= -1e-9)  # 单调非降
    # 顶桶 = p≥0.95 的 5% 样本, 映射顶值 ≈ 该桶 label 中位数
    top = labels[p >= 0.95]
    assert gr[-1] == pytest.approx(float(np.median(top)), abs=0.01)


def test_fit_rank_map_empty_bucket_interpolated():
    p = np.array([0.05, 0.5, 0.55, 0.95] * 10)  # 大量空桶
    labels = np.linspace(-0.02, 0.08, len(p))
    seg = pd.DataFrame({"f1": p, "label_10d_net": labels})
    rmap = fit_rank_map(_StubReg(p), seg, "label_10d_net", ["f1"], bins=20)
    gr = np.asarray(rmap["grid_ret"])
    assert np.all(np.isfinite(gr))
    assert np.all(np.diff(gr) >= -1e-9)


def test_rank_map_apply_passthrough_when_none():
    p = np.array([0.1, 0.9])
    out = rank_map_apply(None, p)
    assert np.allclose(out, p)


def test_rank_map_apply_interp_and_clip():
    rmap = {
        "grid_pct": [0.05, 0.5, 0.95],
        "grid_ret": [-0.02, 0.03, 0.08],
        "bins": 3,
    }
    out = rank_map_apply(rmap, np.array([0.05, 0.275, 0.95, 1.2, -0.5]))
    assert out[0] == pytest.approx(-0.02)
    assert out[1] == pytest.approx(0.005)  # 线性插值中点
    assert out[2] == pytest.approx(0.08)
    assert out[3] == pytest.approx(0.08)  # >1 裁剪到顶
    assert out[4] == pytest.approx(-0.02)  # <0 裁剪到底


# ── fit_calibrator: 10d 残差在收益语义空间 ─────────────────────
def test_fit_calibrator_resid_in_return_space(monkeypatch):
    monkeypatch.setattr("config.settings.LEGACY_10D_RANK_TARGET", {"enable": True, "bins": 20, "boards": ["main"]})
    p = np.linspace(0.05, 0.95, 60)
    labels = np.linspace(-0.02, 0.10, 60)
    calib = pd.DataFrame({"f1": p, "label_10d_net": labels})
    rmap = {
        "grid_pct": [0.05, 0.5, 0.95],
        "grid_ret": [-0.02, 0.04, 0.10],
        "bins": 3,
    }
    trained = {
        "board": "main",
        "feature_cols": ["f1"],
        "models": {"10d_reg": (_StubReg(p), "label_10d_net")},
        "segs": {"calib": calib},
        "10d_rank_map": rmap,
    }
    DualTrackTrainer.fit_calibrator(trained)
    resid = np.asarray(trained["reg_resid_10d"])
    mapped = np.interp(p, rmap["grid_pct"], rmap["grid_ret"])
    expect = np.sort(labels - mapped)
    assert np.allclose(resid, expect, atol=1e-9)


# ── V35Predictor: 推理端映射 ──────────────────────────────────
def _mk_bundle(rmap):
    return {
        "feature_cols": ["f1"],
        "models": {
            "3d_reg": (_StubReg([0.01, 0.02]), "label_3d_net"),
            "3d_cls": (_StubCls(), "label_3d_cls"),
            "5d_reg": (_StubReg([0.02, 0.03]), "label_5d_net"),
            "5d_cls": (_StubCls(), "label_5d_cls"),
            "10d_reg": (_StubReg([0.95, 0.50]), "label_10d_net"),
            "10d_cls": (_StubCls(), "label_10d_cls"),
        },
        **({"10d_rank_map": rmap} if rmap is not None else {}),
    }


def _mk_features():
    return pd.DataFrame(
        {
            "symbol": ["600001.SH", "600002.SH"],
            "date": [pd.Timestamp("2026-01-06")] * 2,
            "board": ["main"] * 2,
            "industry": ["银行"] * 2,
            "f1": [1.0, 2.0],
        }
    )


def _mk_predictor(rmap):
    pred = object.__new__(V35Predictor)
    pred.bundles = {"main": _mk_bundle(rmap)}
    return pred


def test_predictor_maps_rank_to_return_space():
    rmap = {"grid_pct": [0.05, 0.5, 0.95], "grid_ret": [-0.02, 0.03, 0.08], "bins": 3}
    out = _mk_predictor(rmap).predict(_mk_features(), "main")
    v = out.sort_values("symbol")["pred_ret_10d"].to_numpy(dtype=float)
    assert v[0] == pytest.approx(0.08)  # pct 0.95 → 顶桶收益
    assert v[1] == pytest.approx(0.03)  # pct 0.50 → 中桶收益


def test_predictor_legacy_bundle_without_map_unchanged():
    out = _mk_predictor(None).predict(_mk_features(), "main")
    v = out.sort_values("symbol")["pred_ret_10d"].to_numpy(dtype=float)
    assert v[0] == pytest.approx(0.95)
    assert v[1] == pytest.approx(0.50)
