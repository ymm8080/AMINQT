"""展示层校准单测 (2026-08-29): pred_excess 超额列 + 概率再校准.

- _mkt_expected: 近窗全池等权日均值, NaN 标签排除, 空数据 NaN
- _anchor_reported: pred_excess_{h} = 锚定后 pred_ret_{h} − 市场基准
- _recal_factor: 完美校准→1, 高估→<1, 成熟不足收缩, 边界夹逼, 退化输入恒等
- _recal_probs: 板内常数乘法 (排序不变), 无面板板块不动, 开关关闭恒等
"""

import numpy as np
import pandas as pd
import pytest

from scripts import _shortlist_t5_t10 as mod
from scripts._shortlist_t5_t10 import (
    ABS_TARGET,
    ANCHOR_TOP,
    ANCHOR_WINDOW,
    HORIZONS,
    _anchor_reported,
    _mkt_expected,
    _recal_factor,
    _recal_probs,
    _trailing_realized,
)


# ---------------------------------------------------------------- _mkt_expected
def test_mkt_expected_daily_mean_then_window_mean():
    dates = pd.bdate_range("2025-01-06", periods=4)
    rows = []
    for i, d in enumerate(dates):
        rows += [
            {"symbol": "600001", "date": d, "label_pm_3d_net": 0.01 * (i + 1)},
            {"symbol": "B", "date": d, "label_pm_3d_net": 0.02 * (i + 1)},
            # NaN 标签 (未实现) 自动排除
            {"symbol": "C", "date": d, "label_pm_3d_net": np.nan},
        ]
    fr = pd.DataFrame(rows)
    # 全窗: 每日均值 = 0.015*(i+1), 再取均值 = 0.015*2.5
    assert _mkt_expected(fr, "3d") == pytest.approx(0.0375)
    # 只取近 2 日: 0.015*3, 0.015*4 → 0.0525
    assert _mkt_expected(fr, "3d", window=2) == pytest.approx(0.0525)


def test_mkt_expected_empty_is_nan():
    fr = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-06"]), "label_pm_3d_net": [np.nan]}
    )
    assert np.isnan(_mkt_expected(fr, "3d"))


# ---------------------------------------------------------------- anchor excess
def _anchor_fixture():
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
    return panel, res


def test_anchor_reported_pred_excess_equals_pred_minus_mkt():
    panel, res = _anchor_fixture()
    anchored = _anchor_reported(res, panel)
    fr = panel[("main", "both")]
    m = anchored["board"] == "main"
    for h in HORIZONS:
        exp = _mkt_expected(fr, h, window=ANCHOR_WINDOW)
        # 逐行: excess = 锚定后 pred_ret − 市场基准
        assert np.allclose(
            anchored.loc[m, f"pred_excess_{h}"],
            anchored.loc[m, f"pred_ret_{h}"] - exp,
        )
        # 锚定不变量 (回归保护): 均值 = 实得锚
        t_real = _trailing_realized(fr, h, top=ANCHOR_TOP, window=ANCHOR_WINDOW)
        assert anchored.loc[m, f"pred_ret_{h}"].mean() == pytest.approx(
            t_real, rel=1e-9
        )


def test_anchor_reported_excess_nan_without_panel():
    # 板块无面板帧 → 超额列保持 NaN, pred_ret 不动
    panel, res = _anchor_fixture()
    res_dual = res.copy()
    res_dual["board"] = "dual"
    anchored = _anchor_reported(res_dual, {"dual": None})
    for h in HORIZONS:
        assert anchored[f"pred_excess_{h}"].isna().all()
        assert anchored[f"pred_ret_{h}"].equals(res_dual[f"pred_ret_{h}"])


# ---------------------------------------------------------------- _recal_factor
def test_recal_factor_math():
    b = (0.2, 1.5)
    # 完美校准 → 1
    assert _recal_factor(0.5, 0.5, 30, 20, b) == pytest.approx(1.0)
    # 高估: 预测 0.5 实得 0.25, 成熟 → 0.5
    assert _recal_factor(0.5, 0.25, 30, 20, b) == pytest.approx(0.5)
    # 成熟不足收缩: N=10 → (1-0.5)*0.5 → 0.75
    assert _recal_factor(0.5, 0.25, 10, 20, b) == pytest.approx(0.75)
    # 下界夹逼: 实得 0
    assert _recal_factor(0.5, 0.0, 30, 20, b) == pytest.approx(0.2)
    # 上界夹逼: 低估 3x
    assert _recal_factor(0.1, 0.3, 30, 20, b) == pytest.approx(1.5)
    # 退化: pred 0 / NaN / N<=0 → 1
    assert _recal_factor(0.0, 0.5, 30, 20, b) == 1.0
    assert _recal_factor(np.nan, 0.5, 30, 20, b) == 1.0
    assert _recal_factor(0.5, np.nan, 0, 20, b) == 1.0


# ---------------------------------------------------------------- _recal_probs
def _recal_fixture():
    # 25 个成熟决策日 ≥ min_matured=20; pred_prob_10d=0.6, mfe_10d=0.01 (<0.06 未命中)
    hist_dates = pd.bdate_range("2026-08-03", periods=25)
    hist = pd.DataFrame(
        {
            "symbol": ["600001"] * 25,
            "hist_date": [d.strftime("%Y%m%d") for d in hist_dates],
            **{f"pred_prob_{h}": [0.6] * 25 for h in HORIZONS},
        }
    )
    lk_rows = []
    for d in hist_dates:
        lk_rows.append(
            {
                "symbol": "600001",
                "date": d,
                "board": "main",
                "mfe_3d": np.nan,  # 3d 未成熟 → 因子恒等, 不动
                "mfe_5d": np.nan,
                "mfe_10d": 0.01,
            }
        )
    lk = pd.DataFrame(lk_rows)
    res = pd.DataFrame(
        {
            "symbol": ["600001", "600002", "300001"],
            "board": ["main", "main", "dual"],
            "pred_prob_3d": [0.5, 0.4, 0.7],
            "pred_prob_5d": [0.5, 0.4, 0.7],
            "pred_prob_10d": [0.6, 0.5, 0.7],
        }
    )
    return hist, lk, res


def test_recal_probs_board_constant_keeps_order(monkeypatch):
    hist, lk, res = _recal_fixture()
    monkeypatch.setattr(
        mod,
        "PARALLEL_PROB_RECAL",
        {
            "enable": True,
            "min_matured": 20,
            "window_days": 42,
            "factor_bounds": (0.2, 1.5),
        },
    )
    monkeypatch.setattr(mod, "_load_raw_history", lambda sel, m: hist)
    monkeypatch.setattr(mod, "_mfe_lookup", lambda: lk)
    out = _recal_probs(res.copy(), pd.Timestamp("2026-08-29"), "fusion")
    # main 10d: hit=0/pred=0.6 → factor=0.2 (下界) → 板内同乘
    assert out.loc[[0, 1], "pred_prob_10d"].tolist() == pytest.approx(
        [0.6 * 0.2, 0.5 * 0.2]
    )
    # 板内排序不变
    assert out.loc[0, "pred_prob_10d"] > out.loc[1, "pred_prob_10d"]
    # 未成熟视界 (mfe NaN) 不动
    assert out["pred_prob_3d"].equals(res["pred_prob_3d"])
    assert out["pred_prob_5d"].equals(res["pred_prob_5d"])
    # dual 无面板数据 → 不动
    assert out.loc[2, "pred_prob_10d"] == 0.7
    # 命中口径与 ABS_TARGET 一致性: mfe 0.07 > 0.06 应记命中
    assert ABS_TARGET["10d"] == 0.06


def test_recal_probs_disabled_and_empty_hist(monkeypatch):
    hist, lk, res = _recal_fixture()
    monkeypatch.setattr(
        mod,
        "PARALLEL_PROB_RECAL",
        {
            "enable": False,
            "min_matured": 20,
            "window_days": 42,
            "factor_bounds": (0.2, 1.5),
        },
    )
    monkeypatch.setattr(mod, "_load_raw_history", lambda sel, m: hist)
    monkeypatch.setattr(mod, "_mfe_lookup", lambda: lk)
    out = _recal_probs(res.copy(), pd.Timestamp("2026-08-29"), "fusion")
    assert out.equals(res)
    # enable 但历史为空 → 恒等
    monkeypatch.setattr(
        mod,
        "PARALLEL_PROB_RECAL",
        {
            "enable": True,
            "min_matured": 20,
            "window_days": 42,
            "factor_bounds": (0.2, 1.5),
        },
    )
    monkeypatch.setattr(mod, "_load_raw_history", lambda sel, m: pd.DataFrame())
    out2 = _recal_probs(res.copy(), pd.Timestamp("2026-08-29"), "fusion")
    assert out2.equals(res)
