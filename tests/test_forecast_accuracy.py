# -*- coding: utf-8 -*-
"""预测准确度 (WMAPE/bias) 测试."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.forecast_accuracy import (
    bias,
    score_forecast,
    score_matured_forecasts,
    wmape,
)
from app.pipeline1.label_engine import LabelEngine


class TestMetrics:
    def test_wmape(self):
        actual = pd.Series([0.02, -0.01, 0.03])
        pred = pd.Series([0.01, 0.0, 0.05])
        # |err| = .01+.01+.02=.04; Σ|actual|=.06 → 2/3
        assert wmape(actual, pred) == pytest.approx(0.04 / 0.06)

    def test_wmape_zero_denominator(self):
        assert np.isnan(wmape(pd.Series([0.0, 0.0]), pd.Series([0.01, 0.02])))

    def test_bias_sign(self):
        actual = pd.Series([0.01, 0.02])
        assert bias(actual, pd.Series([0.03, 0.04])) == pytest.approx(0.02)  # 高估
        assert bias(actual, pd.Series([0.0, 0.0])) == pytest.approx(-0.015)  # 低估


def _labeled_panel(symbols=("AAA", "BBB"), days=40, seed=3):
    """合成面板 + PM 净口径标签."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-06-01", periods=days)
    frames = []
    for sym in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "close_hfq": close,
                    "volume": 1e6,
                    "amount": 1e8,
                }
            )
        )
    return LabelEngine.build_labels(pd.concat(frames, ignore_index=True))


class TestScoreForecast:
    def test_perfect_forecast(self):
        labeled = _labeled_panel()
        fdate = labeled["date"].iloc[10].strftime("%Y%m%d")
        row = labeled[labeled["date"] == pd.to_datetime(fdate)]
        forecast = pd.DataFrame(
            {
                "symbol": row["symbol"],
                "pred_ret_1d": row["label_pm_1d_net"],
                "pred_ret_3d": row["label_pm_3d_net"],
                "pred_ret_5d": row["label_pm_5d_net"],
            }
        )
        out = score_forecast(forecast, labeled, fdate)
        assert out["mature"]
        for k in (1, 3, 5):
            assert out["horizons"][k]["mae_1d"] == pytest.approx(0.0, abs=1e-4)
            assert out["horizons"][k]["bias_1d"] == pytest.approx(0.0, abs=1e-4)
            assert out["horizons"][k]["n_samples"] == 2

    def test_constant_overestimate(self):
        labeled = _labeled_panel()
        fdate = labeled["date"].iloc[10].strftime("%Y%m%d")
        row = labeled[labeled["date"] == pd.to_datetime(fdate)]
        forecast = pd.DataFrame(
            {
                "symbol": row["symbol"],
                "pred_ret_1d": row["label_pm_1d_net"] + 0.01,  # 系统性高估 1%
                "pred_ret_3d": row["label_pm_3d_net"] + 0.01,
                "pred_ret_5d": row["label_pm_5d_net"] + 0.01,
            }
        )
        out = score_forecast(forecast, labeled, fdate)
        assert out["horizons"][1]["bias_1d"] == pytest.approx(0.01)

    def test_immature_when_no_5d_actuals(self):
        labeled = _labeled_panel()
        fdate = labeled["date"].iloc[-1].strftime("%Y%m%d")  # 最后一天无未来数据
        forecast = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "pred_ret_1d": [0.01, 0.02],
                "pred_ret_3d": [0.01, 0.02],
                "pred_ret_5d": [0.01, 0.02],
            }
        )
        out = score_forecast(forecast, labeled, fdate)
        assert not out["mature"]


class TestScoreMatured:
    def test_writes_and_skips(self, tmp_path):
        labeled = _labeled_panel()
        panel = labeled.drop(
            columns=[c for c in labeled.columns if c.startswith("label_")]
        )
        list_dir = tmp_path / "lists"
        out_dir = tmp_path / "acc"
        list_dir.mkdir()
        fdate = labeled["date"].iloc[10].strftime("%Y%m%d")
        row = labeled[labeled["date"] == pd.to_datetime(fdate)]
        pd.DataFrame(
            {
                "symbol": row["symbol"],
                "pred_ret_1d": row["label_pm_1d_net"] + 0.005,
                "pred_ret_3d": row["label_pm_3d_net"] + 0.005,
                "pred_ret_5d": row["label_pm_5d_net"] + 0.005,
            }
        ).to_parquet(list_dir / f"list_{fdate}.parquet", index=False)
        # 未成熟清单 (最后一天) 不应打分
        fdate2 = labeled["date"].iloc[-1].strftime("%Y%m%d")
        pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "pred_ret_1d": [0.01, 0.02],
                "pred_ret_3d": [0.01, 0.02],
                "pred_ret_5d": [0.01, 0.02],
            }
        ).to_parquet(list_dir / f"list_{fdate2}.parquet", index=False)

        scored = score_matured_forecasts(str(list_dir), panel, out_dir=str(out_dir))
        assert len(scored) == 1 and scored[0]["forecast_date"] == fdate
        summary = json.loads((out_dir / f"accuracy_{fdate}.json").read_text())
        assert summary["horizons"]["1"]["bias_1d"] == pytest.approx(0.005)
        assert (out_dir / f"detail_{fdate}.parquet").exists()
        # 幂等: 第二次不再重复打分
        assert score_matured_forecasts(str(list_dir), panel, out_dir=str(out_dir)) == []
