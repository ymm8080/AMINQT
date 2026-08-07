"""Tests for app/pipeline1/pred_smoothing — legacy 预测稳定性 (2026-08-06).

对齐 parallel _shortlist_t5_t10.ema_smooth: 每股 forecast 列 (pred_ret_*/prob_up*/pred_q50*)
用近 K 个可用交易日 raw 的衰减加权均值, 抑制单股相邻交易日预测/概率剧变.
"""

import pandas as pd
import pytest

from app.pipeline1.pred_smoothing import (
    ALPHA,
    load_raw_history,
    persist_raw_preds,
    smooth_preds,
)

FORECAST_COLS = [
    "pred_ret_1d",
    "pred_ret_3d",
    "prob_up",
    "prob_up_3d",
    "pred_q50_3d",
]


def _cand(symbol="600519", vals=None):
    vals = vals or {
        "pred_ret_1d": 0.05,
        "pred_ret_3d": 0.06,
        "prob_up": 0.50,
        "prob_up_3d": 0.55,
        "pred_q50_3d": 0.04,
    }
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "board": ["main"],
            "score": [1.5],
            **{c: [vals[c]] for c in FORECAST_COLS},
        }
    )


def test_persist_and_load_roundtrip(tmp_path, monkeypatch):
    from app.pipeline1 import pred_smoothing

    monkeypatch.setattr(pred_smoothing, "STOCK_LIST_DIR", tmp_path)
    fp = persist_raw_preds(_cand(), "20260805", "testmod")
    assert str(tmp_path / "legacy_preds_raw_20260805__testmod.csv") == fp
    hist = load_raw_history("20260806", "testmod")
    assert set(hist["symbol"]) == {"600519"}
    assert (hist["hist_date"] == "20260805").all()
    assert "pred_ret_3d" in hist.columns


def test_smooth_blends_today_and_history(tmp_path, monkeypatch):
    from app.pipeline1 import pred_smoothing

    monkeypatch.setattr(pred_smoothing, "STOCK_LIST_DIR", tmp_path)
    # 昨日: pred_ret_3d=0.05, prob_up_3d=0.50; 今日: 0.06 / 0.55
    persist_raw_preds(
        _cand(
            "600519",
            {
                "pred_ret_1d": 0.05,
                "pred_ret_3d": 0.05,
                "prob_up": 0.50,
                "prob_up_3d": 0.50,
                "pred_q50_3d": 0.04,
            },
        ),
        "20260805",
        "testmod",
    )
    res = _cand("600519")
    out = smooth_preds(res, "20260806", "testmod")

    w0, w1 = ALPHA, ALPHA * (1 - ALPHA)
    w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
    assert out["pred_ret_3d"].iloc[0] == pytest.approx(w0 * 0.06 + w1 * 0.05)
    assert out["prob_up_3d"].iloc[0] == pytest.approx(w0 * 0.55 + w1 * 0.50)
    assert out["pred_ret_1d"].iloc[0] == pytest.approx(0.05)  # 昨日今日同值 → 不变
    # score 不平滑
    assert out["score"].iloc[0] == pytest.approx(1.5)


def test_smooth_no_history_returns_raw(tmp_path, monkeypatch):
    from app.pipeline1 import pred_smoothing

    monkeypatch.setattr(pred_smoothing, "STOCK_LIST_DIR", tmp_path)
    res = _cand()
    pd.testing.assert_frame_equal(smooth_preds(res, "20260806", "testmod"), res)


def test_smooth_gap_symbol_without_history_unchanged(tmp_path, monkeypatch):
    from app.pipeline1 import pred_smoothing

    monkeypatch.setattr(pred_smoothing, "STOCK_LIST_DIR", tmp_path)
    persist_raw_preds(_cand("000002"), "20260805", "testmod")  # 历史没有 600519
    res = _cand("600519")
    pd.testing.assert_frame_equal(smooth_preds(res, "20260806", "testmod"), res)


def test_load_history_filters_module_and_date(tmp_path, monkeypatch):
    from app.pipeline1 import pred_smoothing

    monkeypatch.setattr(pred_smoothing, "STOCK_LIST_DIR", tmp_path)
    cases = [
        ("legacy_preds_raw_20260804__testmod.csv", "000001"),  # 昨日匹配 → 保留
        ("legacy_preds_raw_20260806__testmod.csv", "000002"),  # 今日 → 排除
        ("legacy_preds_raw_20260804__other.csv", "000003"),  # 模块不匹配 → 排除
        ("legacy_preds_raw_20260804.csv", "000004"),  # 无模块 → 排除
    ]
    for fname, sym in cases:
        pd.DataFrame({"symbol": [sym], **{c: [0.04] for c in FORECAST_COLS}}).to_csv(
            tmp_path / fname, index=False
        )
    hist = load_raw_history("20260806", "testmod")
    assert set(hist["symbol"]) == {"000001"}
    assert (hist["hist_date"] == "20260804").all()
