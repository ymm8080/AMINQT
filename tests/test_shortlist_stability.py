"""Tests for scripts/_shortlist_t5_t10 预测稳定性 (2026-08-06).

用户报告问题: 同一只股票相邻交易日预测(预期涨幅/达到概率)剧烈变化.
两层修复:
  Layer 1 校准器收缩 — per-stock 斜率向横截面收缩 (_PooledReg, SHRINK_KAPPA)
  Layer 2 输出级时间 EMA 平滑 — 近 SMOOTH_K 可用交易日 raw 预测衰减加权 (ema_smooth)
"""

import numpy as np
import pandas as pd
import pytest

from scripts._shortlist_t5_t10 import (
    HORIZONS,
    SHRINK_KAPPA,
    SMOOTH_ALPHA,
    _load_raw_history,
    _PooledReg,
    ema_smooth,
)


def test_pooled_reg_predict_matches_manual():
    slope, intercept = 0.03, 0.01
    reg = _PooledReg(slope, intercept)
    X = np.array([[0.5], [0.85], [1.0]])
    np.testing.assert_allclose(reg.predict(X), X @ np.array([slope]) + intercept)


def test_shrinkage_slope_is_convex_combo():
    """Layer 1 不变式: slope = λ·slope_per + (1-λ)·slope_cross, 位于两者之间 (λ=n/(n+κ))."""
    n = 90
    lam = n / (n + SHRINK_KAPPA)
    slope_per, slope_cross = 0.12, 0.03
    slope = lam * slope_per + (1 - lam) * slope_cross
    assert lam == pytest.approx(90 / (90 + SHRINK_KAPPA))
    assert slope_per > slope > slope_cross
    # n 越大收缩越弱 (更信任该股自身斜率); n 趋近 0 时收敛到横截面
    assert n / (n + SHRINK_KAPPA) > (n - 30) / (n - 30 + SHRINK_KAPPA)


def _res_row(symbol="000001", mag=0.05, prob=0.60):
    return pd.DataFrame(
        {
            "date": ["2026-08-07"],
            "board": ["main"],
            "symbol": [symbol],
            "systems": ["fusion"],
            "score": [0.8],
            **{f"pred_mag_{h}": [mag] for h in HORIZONS},
            **{f"pred_prob_{h}": [prob] for h in HORIZONS},
        }
    )


def test_ema_smooth_blends_today_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts._shortlist_t5_t10.STOCK_LIST_DIR", tmp_path)
    pd.DataFrame(
        {
            "symbol": ["000001"],
            "pred_mag_3d": [0.04],
            "pred_prob_3d": [0.50],
            "pred_mag_2d": [0.03],
            "pred_prob_2d": [0.52],
            "pred_mag_5d": [0.05],
            "pred_prob_5d": [0.50],
            "pred_mag_10d": [0.07],
            "pred_prob_10d": [0.50],
        }
    ).to_csv(tmp_path / "parallel_preds_raw_20260806__testmod.csv", index=False)

    res = _res_row()
    out = ema_smooth(res, pd.Timestamp("2026-08-07"), "testmod")

    w0, w1 = SMOOTH_ALPHA, SMOOTH_ALPHA * (1 - SMOOTH_ALPHA)
    w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
    assert out["pred_mag_3d"].iloc[0] == pytest.approx(w0 * 0.05 + w1 * 0.04)
    assert out["pred_prob_3d"].iloc[0] == pytest.approx(w0 * 0.60 + w1 * 0.50)
    # score 不平滑
    assert out["score"].iloc[0] == pytest.approx(0.8)


def test_ema_smooth_no_history_returns_raw(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts._shortlist_t5_t10.STOCK_LIST_DIR", tmp_path)
    res = _res_row()
    pd.testing.assert_frame_equal(
        ema_smooth(res, pd.Timestamp("2026-08-07"), "testmod"), res
    )


def test_ema_smooth_gap_symbol_without_history_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts._shortlist_t5_t10.STOCK_LIST_DIR", tmp_path)
    pd.DataFrame(
        {
            "symbol": ["000002"],
            "pred_mag_3d": [0.04],
            "pred_prob_3d": [0.50],
            "pred_mag_2d": [0.03],
            "pred_prob_2d": [0.52],
            "pred_mag_5d": [0.05],
            "pred_prob_5d": [0.50],
            "pred_mag_10d": [0.07],
            "pred_prob_10d": [0.50],
        }
    ).to_csv(tmp_path / "parallel_preds_raw_20260806__testmod.csv", index=False)
    res = _res_row(symbol="000001")  # 历史里没有 000001 → 原样
    pd.testing.assert_frame_equal(
        ema_smooth(res, pd.Timestamp("2026-08-07"), "testmod"), res
    )


def test_load_raw_history_filters_module_and_date(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts._shortlist_t5_t10.STOCK_LIST_DIR", tmp_path)
    cases = [
        ("parallel_preds_raw_20260805__testmod.csv", "000001"),  # 昨日匹配 → 保留
        ("parallel_preds_raw_20260807__testmod.csv", "000002"),  # 今日 → 排除
        ("parallel_preds_raw_20260805__other.csv", "000003"),  # 模块不匹配 → 排除
        ("parallel_preds_raw_20260805.csv", "000004"),  # 无模块 → 排除
    ]
    pred_cols = {f"{k}_{h}": 0.04 for h in HORIZONS for k in ("pred_mag", "pred_prob")}
    for fname, sym in cases:
        pd.DataFrame({"symbol": [sym], **pred_cols}).to_csv(
            tmp_path / fname, index=False
        )
    hist = _load_raw_history(pd.Timestamp("2026-08-07"), "testmod")
    assert set(hist["symbol"]) == {"000001"}
    assert (hist["hist_date"] == "20260805").all()


def test_c2c_latest_returns_per_symbol_mag():
    """pred_ret_{h} 数据源 (2026-08-09): _c2c_latest 用每股 score→label_pm_{h}_net
    校准, 决策日每股唯一 close-to-close 平均预期 (非 MFE)."""
    from scripts._shortlist_t5_t10 import _c2c_latest

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-06", periods=120)
    rows = []
    for i in range(10):
        slope = 0.08 + 0.01 * (i % 3)
        base = 0.5 + 0.04 * i
        for d in dates:
            sc = base + 0.02 * rng.normal()
            rows.append(
                {
                    "symbol": f"SYM{i:03d}",
                    "date": d,
                    "score": sc,
                    "label_pm_5d_net": slope * sc + 0.001 * rng.normal(),
                }
            )
    panel = {("main", "both"): pd.DataFrame(rows)}
    last = dates[-1]
    mag = _c2c_latest(panel, "main", "5d", last)
    assert set(mag.index) == {f"SYM{i:03d}" for i in range(10)}
    assert np.isfinite(mag).all()
    # 与 MFE 无关: 结果是 close-to-close 平均预期量级 (score 0.4~0.7 × slope ~0.08 → 2~6%)
    assert (mag.abs() < 0.15).all()
