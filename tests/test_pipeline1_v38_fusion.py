"""V3.8 第六批: 双模型动态融合 + 周三重训排程."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.pipeline1.model_fusion import DualModelFusion, retrain_schedule


class TestFusionWeight:
    def test_softmax_favors_better_ic(self):
        f = DualModelFusion()
        w = f.update_weight(ic_long_20d=0.05, ic_short_20d=0.01)
        # long IC 明显优 → w_long > 0.5 (首次: 0.7×0.5 + 0.3×0.8)
        assert w > 0.5
        f2 = DualModelFusion()
        w2 = f2.update_weight(ic_long_20d=0.01, ic_short_20d=0.05)
        assert w2 < 0.5

    def test_negative_ic_protection_and_bounds(self):
        f = DualModelFusion()
        # 双负 IC → clip 0 → softmax 等权 0.5 → 平滑后仍 ~0.5, 且在边界内
        w = f.update_weight(ic_long_20d=-0.05, ic_short_20d=-0.03)
        assert 0.2 <= w <= 0.8
        # long 极强 short 为负 → w 上限 0.8 附近 (平滑收敛)
        f2 = DualModelFusion()
        for _ in range(20):
            w2 = f2.update_weight(ic_long_20d=0.10, ic_short_20d=-0.01)
        assert w2 <= 0.8 + 1e-9 and w2 > 0.7

    def test_inertia_smoothing(self):
        f = DualModelFusion()
        w1 = f.update_weight(0.05, 0.01)  # 0.7×0.5+0.3×0.8 = 0.59
        assert w1 == pytest.approx(0.59, abs=0.01)
        # 第二次 IC 反转: 惯性使权重不会一步到位
        w2 = f.update_weight(0.01, 0.05)
        assert abs(w2 - 0.5) < abs(0.5 - w1) + 0.2  # 平滑过渡

    def test_fuse_predictions(self):
        f = DualModelFusion()
        f.w_long = 0.6
        out = f.fuse(np.array([0.02, 0.03]), np.array([0.01, 0.01]))
        assert out[0] == pytest.approx(0.6 * 0.02 + 0.4 * 0.01)

    def test_worm_log(self, tmp_path):
        f = DualModelFusion(str(tmp_path))
        f.update_weight(0.03, 0.02)
        with open(tmp_path / "fusion_weights.jsonl", encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        assert {"ic_long_20d", "ic_short_20d", "w_softmax", "w_long"} <= set(rec)


class TestRetrainSchedule:
    def test_saturday_both(self):
        assert retrain_schedule(5) == ["long", "short"]

    def test_wednesday_conditional(self):
        assert retrain_schedule(2, short_ic_alert=True) == ["short"]
        assert retrain_schedule(2, short_ic_alert=False) == []

    def test_weekday_none(self):
        for d in (0, 1, 3, 4, 6):
            assert retrain_schedule(d, short_ic_alert=True) == []
