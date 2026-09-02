"""Tests for scripts/_rankkey_multiseed_sweep — 纯函数单测 (2026-09-01).

覆盖: TOP-N 截取/日净序列/多 seed 中位聚合/子窗均值/挑战者判据/10d 净标签矩阵.
判定规则 (预登记, 3 seed × 4 子窗): 中位 delta > 0 且 ≥2/3 seed 为正 且 ≥3/4 子窗为正.
"""

import numpy as np
import pandas as pd
import pytest

from scripts._rankkey_multiseed_sweep import (
    REG_EMBARGO,
    REG_HORIZON,
    _challenger_verdict,
    _daily_net,
    _daily_topn,
    _reg_labels_from_matrix,
    _seed_median_series,
    _sub_means,
)


def _mk_df():
    return pd.DataFrame(
        {
            "date": ["d1", "d1", "d1", "d1", "d2", "d2", "d2"],
            "symbol": list("abcdabc"),
            "score": [1.0, 3.0, 2.0, np.nan, 5.0, 4.0, 6.0],
            "realized_net": [0.01, 0.02, 0.03, 0.04, 0.10, -0.02, 0.06],
        }
    )


def test_daily_topn_ranks_desc_per_date():
    top = _daily_topn(_mk_df(), "score", 2)
    assert list(top.loc[top["date"] == "d1", "symbol"]) == ["b", "c"]
    assert list(top.loc[top["date"] == "d2", "symbol"]) == ["c", "a"]


def test_daily_topn_nan_score_ranks_last_not_picked():
    top = _daily_topn(_mk_df(), "score", 2)
    assert "d" not in set(top["symbol"])


def test_daily_net_mean_per_date():
    top = _daily_topn(_mk_df(), "score", 2)
    net = _daily_net(top)
    assert net["d1"] == pytest.approx((0.02 + 0.03) / 2)
    assert net["d2"] == pytest.approx((0.06 + 0.10) / 2)


def test_seed_median_series_ignores_missing_day_per_seed():
    s42 = pd.Series({"d1": 0.01, "d2": 0.03})
    s43 = pd.Series({"d1": 0.02, "d2": 0.05})
    s44 = pd.Series({"d1": 0.04})  # d2 缺 → 该日中位只取 42/43 两 seed
    med = _seed_median_series({42: s42, 43: s43, 44: s44})
    assert med["d1"] == pytest.approx(0.02)
    assert med["d2"] == pytest.approx(0.04)


def test_sub_means_splits_contiguous_quarters():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    s = pd.Series(np.arange(8, dtype=float), index=idx)
    subs = _sub_means(s, n_sub=4)
    assert subs == pytest.approx([0.5, 2.5, 4.5, 6.5])


def test_sub_means_empty_window_is_nan():
    s = pd.Series([1.0], index=pd.date_range("2026-01-01", periods=1))
    subs = _sub_means(s, n_sub=4)
    assert len(subs) == 4
    assert np.isnan(subs[3])


def test_challenger_verdict_pass_requires_all_three_conditions():
    sub_ok = [0.01, 0.02, 0.03, 0.04]
    assert _challenger_verdict({42: 0.05, 43: 0.03, 44: 0.01}, sub_ok)["pass"]
    r = _challenger_verdict({42: 0.05, 43: 0.03, 44: 0.01}, sub_ok)
    assert r["seeds_pos"] == 3 and r["subs_pos"] == 4


def test_challenger_verdict_fails_when_median_delta_zero():
    d = {42: 0.05, 43: 0.00, 44: -0.05}
    assert not _challenger_verdict(d, [0.01, 0.02, 0.03, 0.04])["pass"]


def test_challenger_verdict_fails_at_one_positive_seed():
    d = {42: 0.05, 43: -0.01, 44: -0.02}
    assert not _challenger_verdict(d, [0.01, 0.02, 0.03, 0.04])["pass"]


def test_challenger_verdict_fails_at_two_positive_subwindows():
    d = {42: 0.05, 43: 0.03, 44: 0.01}
    assert not _challenger_verdict(d, [0.01, 0.02, -0.03, -0.04])["pass"]


def test_challenger_verdict_ignores_nan_subwindows_in_denominator():
    d = {42: 0.05, 43: 0.03, 44: 0.01}
    r = _challenger_verdict(d, [0.01, 0.02, 0.03, np.nan])
    assert r["pass"] and r["subs_pos"] == 3 and r["n_subs_valid"] == 3


def test_reg_labels_matrix_buy_next_sell_horizon11():
    """row j: buy=cal[j+1], sell=cal[j+11], 净=sell/buy-1-cost."""
    cost = 0.002
    P = np.linspace(1.0, 1.0 + 0.1 * 14, 15, dtype=float)[None, :]
    j = np.array([0])
    lab = _reg_labels_from_matrix(P, np.array([0]), j, cost, horizon=REG_HORIZON)
    buy, sell = P[0, 1], P[0, 11]
    assert lab[0] == pytest.approx(sell / buy - 1 - cost)


def test_reg_labels_matrix_nan_when_horizon_exceeds_calendar():
    P = np.ones((1, 15))
    j = np.array([4])  # 4+11 = 15 越界
    lab = _reg_labels_from_matrix(P, np.array([0]), j, 0.002, horizon=REG_HORIZON)
    assert np.isnan(lab[0])


def test_reg_labels_matrix_nan_on_bad_price():
    P = np.array([[1.0, np.nan] + [2.0] * 13])  # buy 价 (j+1=1) NaN
    j = np.array([0])
    lab = _reg_labels_from_matrix(P, np.array([0]), j, 0.002, horizon=REG_HORIZON)
    assert np.isnan(lab[0])


def test_reg_embargo_matches_label_horizon():
    """训练掩码 idx < pos-REG_EMBARGO: pos-11 行标签用到 pos 日价格, 必须排除."""
    assert REG_EMBARGO == REG_HORIZON == 11
