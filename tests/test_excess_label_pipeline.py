"""超额标签管线 (2026-08-29): 训练端板内按日去均值 + bundle 市场均值常数 + 推理端加回.

2026-08-30 dual 启用 (行情重审翻案). 语义: 训练目标改"跑赢同板市场", 推理端
pred_ret_{k}d 加回 bundle 常数复原绝对口径 — 下游全部不变, 只有日内排名变化.
常数必须在去均值前的标签上计算 (去均值后日均按构造≈0), 经 attrs 带出.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.predictor import V35Predictor
from app.pipeline1.train_runner import (
    compute_mkt_expected,
    demean_excess_labels,
    prepare_board_frame,
)


def _label_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-08-27", "2026-08-27", "2026-08-28", "2026-08-28"])
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600001", "600002"],
            "date": dates,
            "label_pm_3d_net": [0.02, 0.04, -0.01, 0.03],
            "label_pm_5d_net": [0.03, 0.05, 0.00, 0.02],
            "label_pm_10d_net": [0.05, 0.07, 0.01, 0.03],
            "label_pm_3d": [0.02, 0.04, -0.01, 0.03],  # 非 _net 不动
            "label_ret_3d": [0.03, 0.06, 0.00, 0.05],  # 非目标列不动
        }
    )


class TestDemean:
    def test_net_labels_demeaned_per_date(self):
        df = demean_excess_labels(_label_frame())
        for c in ("label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net"):
            per_day = df.groupby("date")[c].mean()
            assert per_day.map(abs).lt(1e-12).all()
        # 板内按日: 同日两票差不变 (排名不变), 跨日水平中性
        assert df.loc[0, "label_pm_3d_net"] == pytest.approx(-0.01)
        assert df.loc[2, "label_pm_3d_net"] == pytest.approx(-0.02)

    def test_non_net_and_missing_cols_untouched(self):
        df = demean_excess_labels(_label_frame())
        assert df["label_pm_3d"].tolist() == [0.02, 0.04, -0.01, 0.03]
        assert df["label_ret_3d"].tolist() == [0.03, 0.06, 0.00, 0.05]
        thin = demean_excess_labels(_label_frame().drop(columns=["label_pm_10d_net"]))
        assert "label_pm_10d_net" not in thin.columns


class TestMktExpected:
    def test_window_mean_of_realized_daily_means(self):
        df = _label_frame()
        df.loc[df["date"] == "2026-08-28", "label_pm_3d_net"] = np.nan  # 近端未实现
        out = compute_mkt_expected(df, window=2)
        # 已实现日 08-27 均值 0.03; 08-28 全 NaN 剔除 → 只剩 0.03
        assert out["mkt_expected_3d"] == pytest.approx(0.03)
        assert set(out) == {"mkt_expected_3d", "mkt_expected_5d", "mkt_expected_10d"}

    def test_no_realized_labels_returns_zero(self):
        df = _label_frame()
        for c in ("label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net"):
            df[c] = np.nan
        out = compute_mkt_expected(df, window=2)
        assert out["mkt_expected_5d"] == 0.0


class _IdentityFeatures:
    def build(self, board_df, float_shares_map=None, **kw):  # noqa: ARG002
        return board_df.copy()


class TestPrepareBoardFrameWiring:
    def test_flag_routes_demean(self, monkeypatch):
        import app.pipeline1.train_runner as tr

        monkeypatch.setattr(
            tr.LabelEngine, "build_path_labels", staticmethod(lambda df: df)
        )
        monkeypatch.setattr(tr.LabelEngine, "build_labels", staticmethod(lambda df: df))
        monkeypatch.setattr(
            tr.LabelEngine, "mask_suspension", staticmethod(lambda df: df)
        )
        monkeypatch.setattr(
            tr.LabelEngine, "mask_recent_days", staticmethod(lambda df, days: df)
        )
        raw = _label_frame()

        off = prepare_board_frame(raw.copy(), _IdentityFeatures(), label_excess=False)
        assert off["label_pm_3d_net"].tolist() == raw["label_pm_3d_net"].tolist()
        assert "mkt_expected" not in off.attrs

        on = prepare_board_frame(raw.copy(), _IdentityFeatures(), label_excess=True)
        assert on.groupby("date")["label_pm_3d_net"].mean().map(abs).lt(1e-12).all()

    def test_mkt_expected_from_pre_demean_labels(self, monkeypatch):
        """常数取自去均值前标签 — 去均值后日均按构造≈0, 必须非零真市场预期."""
        import app.pipeline1.train_runner as tr

        for m in ("build_path_labels", "build_labels", "mask_suspension"):
            monkeypatch.setattr(tr.LabelEngine, m, staticmethod(lambda df: df))
        monkeypatch.setattr(
            tr.LabelEngine,
            "mask_recent_days",
            staticmethod(lambda df, days: df),
        )
        on = prepare_board_frame(
            _label_frame(), _IdentityFeatures(), label_excess=True
        )
        mkt = on.attrs["mkt_expected"]
        # 3d: 已实现日均值 08-27=0.03, 08-28=0.01 → 常数 0.02 (而非 ≈0)
        assert mkt["mkt_expected_3d"] == pytest.approx(0.02)
        # 10d: 0.06 与 0.02 → 0.04
        assert mkt["mkt_expected_10d"] == pytest.approx(0.04)
        assert set(mkt) == {"mkt_expected_3d", "mkt_expected_5d", "mkt_expected_10d"}


class _ConstModel:
    def __init__(self, val: float):
        self.val = val

    def predict(self, X):  # noqa: N802
        return np.full(len(X), self.val)


class _ConstClsModel(_ConstModel):
    def predict_proba(self, X):  # noqa: N802
        p = np.full(len(X), self.val)
        return np.column_stack([1 - p, p])


def _features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600003"],
            "date": pd.to_datetime(["2026-08-28"] * 3),
            "board": ["main"] * 3,
            "industry": ["银行"] * 3,
            "f1": [1.0, 2.0, 3.0],
        }
    )


class TestPredictorRebase:
    def _predictor(self, bundle: dict) -> V35Predictor:
        pred = V35Predictor.__new__(V35Predictor)
        pred.bundles = {"main": bundle}
        return pred

    def _bundle(self, excess: bool) -> dict:
        return {
            "board": "main",
            "feature_cols": ["f1"],
            "models": {
                "3d_reg": (_ConstModel(0.010), "x"),
                "5d_reg": (_ConstModel(0.020), "x"),
                "10d_reg": (_ConstModel(0.030), "x"),
                "3d_cls": (_ConstClsModel(0.5), "x"),
                "5d_cls": (_ConstClsModel(0.5), "x"),
            },
            **(
                {
                    "label_excess": True,
                    "mkt_expected_3d": 0.003,
                    "mkt_expected_5d": -0.001,
                    "mkt_expected_10d": 0.005,
                }
                if excess
                else {}
            ),
        }

    def test_excess_bundle_preds_shifted_by_constants(self):
        out = self._predictor(self._bundle(excess=True)).predict(_features_df(), "main")
        assert np.allclose(out["pred_ret_3d"], 0.013)
        assert np.allclose(out["pred_ret_5d"], 0.019)
        assert np.allclose(out["pred_ret_10d"], 0.035)

    def test_plain_bundle_preds_unchanged(self):
        out = self._predictor(self._bundle(excess=False)).predict(_features_df(), "main")
        assert np.allclose(out["pred_ret_3d"], 0.010)
        assert np.allclose(out["pred_ret_10d"], 0.030)

    def test_partial_constants_tolerated(self):
        b = self._bundle(excess=True)
        del b["mkt_expected_5d"]  # 缺常数 → 该视界不加回
        out = self._predictor(b).predict(_features_df(), "main")
        assert np.allclose(out["pred_ret_3d"], 0.013)
        assert np.allclose(out["pred_ret_5d"], 0.020)


class TestSaveWhitelist:
    def test_extras_persisted_through_save(self, tmp_path):
        tr = DualTrackTrainer.__new__(DualTrackTrainer)
        tr.model_dir = str(tmp_path)
        trained = {
            "board": "main",
            "feature_cols": ["f1"],
            "models": {},
            "calibrator": object(),
            "label_excess": True,
            "mkt_expected_3d": 0.012,
            "mkt_expected_5d": 0.008,
            "mkt_expected_10d": 0.015,
        }
        path = tr.save(trained, "t_excess")
        with open(path, "rb") as fh:
            persisted = pickle.load(fh)
        assert persisted["label_excess"] is True
        assert persisted["mkt_expected_3d"] == pytest.approx(0.012)
        assert persisted["mkt_expected_10d"] == pytest.approx(0.015)
