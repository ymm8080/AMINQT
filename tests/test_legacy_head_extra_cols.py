# -*- coding: utf-8 -*-
"""Tests for per-head 特征集 (LEGACY_HEAD_EXTRA_COLS, 2026-09-12).

reg/cls 头分开训练 → 各头族挂独有 extras。铁律: 模型实际训练列数必须与
bundle 声明一致 (缺失 extra 剔除+告警), 否则 serving 0 填充后特征数错位崩溃。

覆盖: kind_feature_cols (append/去重/缺失剔除+告警/available=None) /
cols_for (per-kind + 旧 bundle 共享回退) / add_quality_factor (per-date
截面 rank 数学 + NaN skipna + 无基列不加列) / 端到端 (_train_one 按头族
列数训练 → save 落盘 by_kind → V35Predictor 推理端合成 quality_factor
+ per-head X 不崩溃)。
"""

import logging

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.dual_track_trainer import (
    DualTrackTrainer,
    cols_for,
    kind_feature_cols,
)
from app.pipeline1.feature_selector import QUALITY_BASE_COLS, add_quality_factor
from app.pipeline1.predictor import V35Predictor


# ── kind_feature_cols ─────────────────────────────────────────
def test_kind_feature_cols_appends_per_family():
    by = kind_feature_cols("main", ["f1", "f2"])
    assert by["10d_reg"] == ["f1", "f2", "SL斜率20"]
    assert by["3d_reg"] == ["f1", "f2", "SL斜率20"]
    assert by["10d_cls"] == ["f1", "f2", "quality_factor"]
    assert by["3d_cls"] == ["f1", "f2", "quality_factor"]


def test_kind_feature_cols_shared_prefix_no_dup():
    by = kind_feature_cols("main", ["f1", "SL斜率20"])
    assert by["10d_reg"] == ["f1", "SL斜率20"]  # 已在共享集 → 不重复 append


def test_kind_feature_cols_board_without_extras():
    by = kind_feature_cols("dual", ["f1"])
    # dual 无 reg extras (SL斜率20 已在共享 force_include)
    assert by["10d_reg"] == ["f1"]
    assert by["10d_cls"] == ["f1", "quality_factor", "ps_ttm"]


def test_kind_feature_cols_drops_missing_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        by = kind_feature_cols("main", ["f1"], available={"f1"})
    assert by["10d_reg"] == ["f1"]  # SL斜率20 不在帧内 → 剔除
    assert by["10d_cls"] == ["f1"]  # quality_factor 不在帧内 → 剔除
    assert any("per-head 特征缺失" in r.message for r in caplog.records)


def test_kind_feature_cols_available_none_keeps_all():
    by = kind_feature_cols("main", ["f1"])
    assert "SL斜率20" in by["10d_reg"]
    assert "quality_factor" in by["10d_cls"]


# ── cols_for ──────────────────────────────────────────────────
def test_cols_for_per_kind_and_fallback():
    b = {"feature_cols": ["f1"], "feature_cols_by_kind": {"10d_reg": ["f1", "e"]}}
    assert cols_for(b, "10d_reg") == ["f1", "e"]
    assert cols_for(b, "3d_reg") == ["f1"]  # by_kind 缺该 kind → 共享回退


def test_cols_for_old_bundle():
    assert cols_for({"feature_cols": ["f1"]}, "10d_reg") == ["f1"]
    assert cols_for({}, "10d_reg") == []


# ── add_quality_factor ────────────────────────────────────────
def test_add_quality_factor_per_date_rank_mean():
    df = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d2", "d2"],
            "roe": [1.0, 2.0, 3.0, 5.0, 4.0],
            "eps": [10.0, 20.0, 30.0, 1.0, 2.0],
        }
    )
    assert add_quality_factor(df) is True
    got = df["quality_factor"].to_numpy(dtype=float)
    # d1: roe/eps 同序 → rank pct [1/3, 2/3, 1.0]
    assert got[0] == pytest.approx(1 / 3)
    assert got[1] == pytest.approx(2 / 3)
    assert got[2] == pytest.approx(1.0)
    # d2: roe [5,4] → pct [1.0, 0.5]; eps [1,2] → pct [0.5, 1.0]; mean 均 0.75
    assert got[3] == pytest.approx(0.75)
    assert got[4] == pytest.approx(0.75)


def test_add_quality_factor_nan_skipna():
    df = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1"],
            "roe": [1.0, np.nan, 3.0],
            "eps": [1.0, 2.0, 3.0],
        }
    )
    add_quality_factor(df)
    v = df["quality_factor"].to_numpy(dtype=float)
    # roe [1,nan,3] 非空 2 个 → pct [0.5, nan, 1.0]; eps pct [1/3, 2/3, 1.0]
    assert v[1] == pytest.approx(2 / 3)  # roe NaN → eps 独撑 (mean skipna)
    assert v[0] == pytest.approx((0.5 + 1 / 3) / 2)
    assert v[2] == pytest.approx(1.0)


def test_add_quality_factor_no_base_cols():
    df = pd.DataFrame({"date": ["d1"], "close": [1.0]})
    assert add_quality_factor(df) is False
    assert "quality_factor" not in df.columns


def test_quality_base_cols_not_in_panel_vocab():
    # 基列必须真实存在于基本面面板词表 (拼写漂移会让 quality_factor 永远 NaN)
    assert len(QUALITY_BASE_COLS) == 16
    assert "roe" in QUALITY_BASE_COLS and "ps_ttm" not in QUALITY_BASE_COLS


# ── 端到端: _train_one per-head 列数 → save → predictor ───────
def _mk_frame() -> pd.DataFrame:
    rows = []
    for di, d in enumerate(("2026-01-05", "2026-01-06", "2026-01-07")):
        for i in range(4):
            rows.append(
                {
                    "date": d,
                    "symbol": f"{600000 + i}.SH",
                    "board": "main",
                    "industry": "银行",
                    "f1": float(i) + 0.1 * di,
                    "SL斜率20": float(4 - i) * 0.01,
                    "roe": float(i) + 0.5,
                    "eps": float(i) * 2.0,
                    "label_pm_3d": 0.01 * (i - 2),
                    "label_pm_5d": 0.02 * (i - 2),
                    "label_pm_10d": 0.03 * (i - 2),
                    "label_pm_3d_cls": float(i % 2),
                    "label_pm_5d_cls": float((i + 1) % 2),
                    "label_pm_10d_cls": float(i % 2),
                    "label_pm_3d_net": 0.01 * (i - 2) + 0.001 * di,
                    "label_pm_5d_net": 0.02 * (i - 2),
                    "label_pm_10d_net": 0.03 * (i - 2),
                    "label_pm_3d_cls_net": float(i % 2),
                    "label_pm_5d_cls_net": float((i + 1) % 2),
                    "label_pm_10d_cls_net": float(i % 2),
                }
            )
    return pd.DataFrame(rows)


def test_end_to_end_train_save_predict_per_head(tmp_path):
    df = _mk_frame()
    assert add_quality_factor(df) is True
    by = kind_feature_cols("main", ["f1"], available=set(df.columns))
    assert by["10d_reg"] == ["f1", "SL斜率20"]
    assert by["10d_cls"] == ["f1", "quality_factor"]

    trainer = DualTrackTrainer(model_dir=str(tmp_path))
    segs = {"train": df.copy(), "es": df.copy()}
    models = {}
    for kind in ("3d_reg", "5d_reg", "10d_reg", "3d_cls", "5d_cls", "10d_cls"):
        model, label = trainer._train_one(kind, segs, by[kind], "main")
        models[kind] = (model, label)
        assert model.n_features_in_ == len(by[kind])  # 模型列数 = 头族清单

    trained = {
        "board": "main",
        "feature_cols": ["f1"],
        "feature_cols_by_kind": by,
        "models": models,
        "segs": {"calib": df.copy()},
    }
    path = trainer.save(trained, "headextra")
    bundle = DualTrackTrainer.load(path)
    assert bundle["feature_cols_by_kind"] == by
    assert bundle["models"]["10d_reg"][0].n_features_in_ == 2
    assert bundle["models"]["10d_cls"][0].n_features_in_ == 2

    # 推理端: 帧无 quality_factor (由 predictor 合成) 且 SL斜率20 透传
    pred = object.__new__(V35Predictor)
    pred.bundles = {"main": bundle}
    feats = _mk_frame()
    out = pred.predict(feats, "main")
    assert np.isfinite(out["pred_ret_3d"].to_numpy(dtype=float)).all()
    assert np.isfinite(out["prob_up_3d"].to_numpy(dtype=float)).all()
    # 合成就地生效 (per-date 截面 rank 在最新截面截断前完成)
    assert "quality_factor" in feats.columns
