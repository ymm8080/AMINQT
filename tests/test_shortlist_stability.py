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


def test_reject_reason_derivation(tmp_path):
    """某板当日短名单为空的原因推导 (2026-08-13): 有 OOS 落盘 → 选股门原因;
    连落盘都没有 → 无候选原因. 交付端据此在清单上标注 '未接受'."""
    from scripts._shortlist_t5_t10 import _reject_reason

    (tmp_path / "stocks_main_sniper_oos.csv").write_text("symbol\n000001\n")
    assert "选股/排名门" in _reject_reason(tmp_path, "main")
    assert _reject_reason(tmp_path, "dual") == (
        "该板块当日无候选 (池为空/未参与并行选股)"
    )
    (tmp_path / "stocks_dual_fusion_oos.csv").write_text("symbol\n300001\n")
    assert "选股/排名门" in _reject_reason(tmp_path, "dual")


def test_trailing_realized_takes_top_by_score_over_realized_days():
    """报告锚 (2026-08-14): _trailing_realized 取末 window 个已实现决策日内每日
    score top-top 的 label 均值; 未实现 (label NaN) 日不计数."""
    from scripts._shortlist_t5_t10 import _trailing_realized

    dates = pd.bdate_range("2025-01-06", periods=30)
    rows = []
    for i in range(20):
        sc = 0.2 + 0.04 * i  # score 随 i 单调 → top-10 by score == i 最大的 10 只
        for d in dates:
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "score": sc,
                    "label_pm_10d_net": 0.001 * i + 0.0005,
                }
            )
    frame = pd.DataFrame(rows)
    val = _trailing_realized(frame, "10d", top=10, window=10)
    expected = float(np.mean([0.001 * i + 0.0005 for i in range(10, 20)]))
    assert val == pytest.approx(expected)
    # 未实现 (label NaN) 日跳过: 只统计有 label 的决策日
    frame.loc[frame["date"] == dates[-1], "label_pm_10d_net"] = np.nan
    assert _trailing_realized(frame, "10d", top=10, window=10) == pytest.approx(
        expected
    )


def test_anchor_reported_shifts_mean_keeps_order_sync_mag10d():
    """_anchor_reported (2026-08-14): 每板块每视界把报告 pred_ret_{h} 均值平移到近窗
    (ANCHOR_WINDOW) top-ANCHOR_TOP 实得锚, 板块内排序 (单调) 不变, pred_mag_10d 同步
    为锚定后的 pred_ret_10d."""
    from scripts._shortlist_t5_t10 import (
        ANCHOR_TOP,
        ANCHOR_WINDOW,
        _anchor_reported,
        _trailing_realized,
    )

    dates = pd.bdate_range("2025-01-06", periods=40)
    panel_rows = []
    for i in range(15):
        sc = 0.3 + 0.04 * i
        for d in dates:
            panel_rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "score": sc,
                    "label_pm_3d_net": 0.015 + 0.0005 * i,
                    "label_pm_5d_net": 0.020 + 0.0005 * i,
                    "label_pm_10d_net": 0.025 + 0.0005 * i,
                }
            )
    panel = {("main", "both"): pd.DataFrame(panel_rows)}

    # 入选档 = 每板块 TOP-5 (score 最高 5 只 S10-S14), 报告值虚高且随 score 单调
    res = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": [str(dates[-1].date())],
                    "board": ["main"],
                    "cut": ["T-5"],
                    "symbol": [f"S{i:02d}"],
                    "score": [0.3 + 0.04 * i],
                    **{f"pred_mag_{h}": [0.04] for h in HORIZONS},
                    **{f"pred_prob_{h}": [0.5] for h in HORIZONS},
                    **{f"pred_ret_{h}": [0.04 + 0.01 * i] for h in HORIZONS},
                }
            )
            for i in range(10, 15)
        ],
        ignore_index=True,
    )
    anchored = _anchor_reported(res, panel)
    for h in HORIZONS:
        t_real = _trailing_realized(
            panel[("main", "both")], h, top=ANCHOR_TOP, window=ANCHOR_WINDOW
        )
        assert anchored[f"pred_ret_{h}"].mean() == pytest.approx(t_real, rel=1e-9)
    # pred_mag_10d 同步到锚定后的 pred_ret_10d
    assert (anchored["pred_mag_10d"] == anchored["pred_ret_10d"]).all()
    # 平移是每板块每视界常数: 锚定后 pred_ret_10d 仍随 score 单调 (每股都减同一 shift)
    assert anchored.sort_values("score")["pred_ret_10d"].is_monotonic_increasing
    shift = float(anchored["pred_ret_10d"].iloc[0] - (0.04 + 0.01 * 10))
    assert (
        anchored["pred_ret_10d"]
        - (0.04 + 0.01 * anchored["symbol"].str[1:].astype(int))
    ).abs().max() == pytest.approx(abs(shift))


def test_rank_and_truncate_keeps_only_top5_per_board():
    """2026-08-14 收紧: rank_and_truncate 每板块只留 pred_mag_10d 前 5, cut 统一 T-5."""
    from scripts._shortlist_t5_t10 import rank_and_truncate

    pref = {"main": "ma", "dual": "du"}
    rows = []
    for board, n in (("main", 8), ("dual", 7)):
        for i in range(n):
            rows.append(
                {
                    "board": board,
                    "symbol": f"{pref[board]}{i:04d}",
                    "pred_mag_10d": 0.05 + 0.001 * i,  # i 越大 mag 越高
                    "score": 0.5,
                }
            )
    out = rank_and_truncate(pd.DataFrame(rows))
    assert set(out["cut"]) == {"T-5"}
    for board, n in (("main", 8), ("dual", 7)):
        b = out[out["board"] == board]
        assert len(b) == 5
        # 保留 pred_mag_10d 最高的 5 只
        assert set(b["symbol"]) == {f"{pref[board]}{i:04d}" for i in range(n - 5, n)}


def test_rank_and_truncate_blend_key_when_pred_prob_present():
    """2026-08-15 A/B 定案: 概率闸附 pred_prob 后, 排名键 = pred_mag_10d × pred_prob.

    高 mag 低 prob 被低 mag 高 prob 反超 (纯 mag 顺序 A>C>B, blend 顺序 B>C>A).
    """
    from scripts._shortlist_t5_t10 import rank_and_truncate

    rows = [
        {
            "board": "main",
            "symbol": "A",
            "pred_mag_10d": 0.10,
            "pred_prob": 0.30,
            "score": 0.5,
        },
        {
            "board": "main",
            "symbol": "B",
            "pred_mag_10d": 0.08,
            "pred_prob": 0.90,
            "score": 0.5,
        },
        {
            "board": "main",
            "symbol": "C",
            "pred_mag_10d": 0.09,
            "pred_prob": 0.60,
            "score": 0.5,
        },
    ]
    out = rank_and_truncate(pd.DataFrame(rows))
    # blend: B=0.072 > C=0.054 > A=0.030
    assert out["symbol"].tolist() == ["B", "C", "A"]
    assert set(out["cut"]) == {"T-5"}


def test_rank_and_truncate_blend_nan_prob_sorts_last():
    """pred_prob 缺失 (fail-open 保留) → blend NaN → 排尾, 不优于有概率的股."""
    from scripts._shortlist_t5_t10 import rank_and_truncate

    rows = [
        {
            "board": "main",
            "symbol": "A",
            "pred_mag_10d": 0.10,
            "pred_prob": float("nan"),
            "score": 0.5,
        },
        {
            "board": "main",
            "symbol": "B",
            "pred_mag_10d": 0.01,
            "pred_prob": 0.50,
            "score": 0.5,
        },
    ]
    out = rank_and_truncate(pd.DataFrame(rows))
    assert out["symbol"].tolist() == ["B", "A"]


def test_rank_and_truncate_board_gate_down_falls_back_to_mag():
    """一板闸可用一板失效: 可用板 blend, 失效板 (pred_prob 全 NaN) 纯 mag 不随机,
    且失效板 rank_blend 归一为 mag 供 build_merged 全局排名."""
    from scripts._shortlist_t5_t10 import rank_and_truncate

    rows = [
        {
            "board": "main",
            "symbol": "mA",
            "pred_mag_10d": 0.10,
            "pred_prob": 0.30,
            "score": 0.5,
        },
        {
            "board": "main",
            "symbol": "mB",
            "pred_mag_10d": 0.08,
            "pred_prob": 0.90,
            "score": 0.5,
        },
        {
            "board": "dual",
            "symbol": "dA",
            "pred_mag_10d": 0.05,
            "pred_prob": float("nan"),
            "score": 0.5,
        },
        {
            "board": "dual",
            "symbol": "dB",
            "pred_mag_10d": 0.09,
            "pred_prob": float("nan"),
            "score": 0.5,
        },
    ]
    out = rank_and_truncate(pd.DataFrame(rows))
    m = out[out["board"] == "main"]
    d = out[out["board"] == "dual"]
    assert m["symbol"].tolist() == ["mB", "mA"]  # blend: 0.072 > 0.030
    assert d["symbol"].tolist() == ["dB", "dA"]  # 失效板纯 mag
    assert d["rank_blend"].tolist() == [0.09, 0.05]  # NaN 归一为 mag
