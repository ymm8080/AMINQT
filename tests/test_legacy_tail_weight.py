# -*- coding: utf-8 -*-
"""LEGACY_TAIL_WEIGHT 尾部样本加权 (爆发猎杀) 单测.

覆盖:
1. 10d_reg: top_q 分位以上样本权重 = 基础时间衰减权重 × weight, 其余不变;
2. 3d_cls 头不吃尾部加权 (kinds 只含 reg);
3. enable=False 回退 (w 与基础权重逐元素一致).
"""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

import config.settings as settings
from app.pipeline1.dual_track_trainer import DualTrackTrainer


class _StubModel:
    captured = None

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, sample_weight=None, **kw):
        _StubModel.captured = (
            None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        )
        return self


def _make_segs() -> dict:
    """train: 2 日 × 50 股 = 100 行, label_pm_10d_net = 0..99 (top10%=90..99);
    es: 1 日 5 行 (es 日数 < MIN_ES_DATES → 不走早停分支)."""
    d1, d2 = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-02")
    rows = []
    for di, d in enumerate((d1, d2)):
        for j in range(50):
            rows.append(
                {
                    "symbol": f"{j:06d}",
                    "date": d,
                    "f1": float(di * 10 + j),
                    "f2": float(j) / 10.0,
                    # 解析链 label_10d → label_pm_10d → label_pm_10d_net: 中间列须在场
                    "label_pm_10d": float(di * 50 + j),
                    "label_pm_10d_net": float(di * 50 + j),
                }
            )
    train = pd.DataFrame(rows)
    es = train.head(5).copy()
    es["date"] = pd.Timestamp("2026-08-03")
    return {"train": train, "es": es}


def _base_reg_w(train: pd.DataFrame) -> np.ndarray:
    from app.pipeline_parallel.prob_head import (
        HALF_LIFE_DAYS,
        decay_sample_weights,
    )

    if HALF_LIFE_DAYS is not None:
        return decay_sample_weights(train["date"], HALF_LIFE_DAYS)
    return DualTrackTrainer.time_weights(train)


class TestTailWeight(unittest.TestCase):
    def setUp(self):
        _StubModel.captured = None

    @mock.patch("lightgbm.LGBMRegressor", _StubModel)
    def test_reg_head_top_decile_gets_4x(self):
        segs = _make_segs()
        trainer = DualTrackTrainer()
        # 默认 enable=False (0912 A/B 判 FAIL 关闭); ON 分支数学显式 mock 验证
        on = dict(settings.LEGACY_TAIL_WEIGHT, enable=True)
        with mock.patch.object(settings, "LEGACY_TAIL_WEIGHT", on):
            model, label = trainer._train_one("10d_reg", segs, ["f1", "f2"], "main")
        self.assertEqual(label, "label_pm_10d_net")
        y = segs["train"][label].to_numpy()
        thr = np.quantile(y, settings.LEGACY_TAIL_WEIGHT["top_q"])
        w = settings.LEGACY_TAIL_WEIGHT["weight"]
        expected = _base_reg_w(segs["train"]) * np.where(y >= thr, w, 1.0)
        self.assertIsNotNone(_StubModel.captured)
        np.testing.assert_allclose(_StubModel.captured, expected, rtol=1e-12)
        # 覆盖率 ~10%
        self.assertAlmostEqual((y >= thr).mean(), 0.10, places=6)

    @mock.patch("lightgbm.LGBMClassifier", _StubModel)
    def test_cls_head_not_weighted(self):
        segs = _make_segs()
        segs["train"]["label_pm_3d_cls"] = (
            segs["train"]["label_pm_10d_net"] > 50
        ).astype(int)
        segs["es"]["label_pm_3d_cls"] = 0
        DualTrackTrainer()._train_one("3d_cls", segs, ["f1", "f2"], "main")
        # cls 头权重必须逐元素等于 time_weights (无尾部 ×4 因子)
        expected = DualTrackTrainer.time_weights(segs["train"])
        np.testing.assert_allclose(_StubModel.captured, expected, rtol=1e-12)

    @mock.patch("lightgbm.LGBMRegressor", _StubModel)
    def test_disable_reverts(self):
        segs = _make_segs()
        off = dict(settings.LEGACY_TAIL_WEIGHT, enable=False)
        with mock.patch.object(settings, "LEGACY_TAIL_WEIGHT", off):
            DualTrackTrainer()._train_one("10d_reg", segs, ["f1", "f2"], "main")
        np.testing.assert_allclose(
            _StubModel.captured, _base_reg_w(segs["train"]), rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
