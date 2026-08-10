"""环境市场分类器 + 否决层测试 (2026-08-10).

覆盖: 三态分类 (ice/range/hot) / 历史基线 / 无历史回退 / 策略映射 / 极端冰点否决 /
面板涨停/跌停近似 (按 board 分档) / 空面板安全回退.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.market_environment import (
    classify_market_state,
    is_veto,
    sentiment_from_panel,
    sentiment_history_from_panel,
    state_policy,
)


def _history(up_mean: float = 60.0, n: int = 60) -> pd.DataFrame:
    """近 n 日情绪历史: 涨停家数恒为 up_mean (除最后一天)."""
    dates = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame(
        {
            "date": dates,
            "count_limit_up": np.full(n, up_mean),
            "count_limit_down": np.full(n, 5.0),
            "market_turnover": np.full(n, 8e11),
        }
    )


def _panel(**overrides) -> pd.DataFrame:
    """小面板: 昨日基线 + 今日若干涨停/跌停/平稳股."""
    n_hist = 10
    base = []
    for i in range(n_hist):
        base.append(
            {
                "symbol": "BASE",
                "date": pd.Timestamp(f"2025-01-{i+1:02d}"),
                "close": 100.0,
                "pre_close": 100.0,
                "amount": 1e9,
                "board": "main",
            }
        )
    rows = list(base)
    # 今日: 10 只主板涨停 (10%), 3 只主板跌停, 2 只双创涨停 (20%), 3 只平稳
    today = pd.Timestamp("2025-01-20")
    for i in range(10):
        rows.append(
            {
                "symbol": f"UP{i}",
                "date": today,
                "close": 110.0,
                "pre_close": 100.0,
                "amount": 1e9,
                "board": "main",
            }
        )
    for i in range(3):
        rows.append(
            {
                "symbol": f"DN{i}",
                "date": today,
                "close": 90.0,
                "pre_close": 100.0,
                "amount": 1e9,
                "board": "main",
            }
        )
    for i in range(2):
        rows.append(
            {
                "symbol": f"CY{i}",
                "date": today,
                "close": 120.0,
                "pre_close": 100.0,
                "amount": 1e9,
                "board": "dual",
            }
        )
    for i in range(3):
        rows.append(
            {
                "symbol": f"FL{i}",
                "date": today,
                "close": 101.0,
                "pre_close": 100.0,
                "amount": 1e9,
                "board": "main",
            }
        )
    df = pd.DataFrame(rows)
    for k, v in overrides.items():
        df[k] = v
    return df


class TestClassify:
    def test_ice_when_up_far_below_baseline(self):
        hist = _history(up_mean=100.0)
        assert classify_market_state({"count_limit_up": 20, "count_limit_down": 1}, hist) == "ice"

    def test_ice_when_limitdown_dominates(self):
        hist = _history(up_mean=60.0)
        assert classify_market_state({"count_limit_up": 30, "count_limit_down": 100}, hist) == "ice"

    def test_hot_when_up_high_and_ratio_good(self):
        hist = _history(up_mean=50.0)
        assert classify_market_state({"count_limit_up": 120, "count_limit_down": 5}, hist) == "hot"

    def test_range_when_normal(self):
        hist = _history(up_mean=60.0)
        assert classify_market_state({"count_limit_up": 55, "count_limit_down": 10}, hist) == "range"

    def test_no_history_fallback_by_ratio(self):
        assert classify_market_state({"count_limit_up": 1, "count_limit_down": 10}) == "ice"
        assert classify_market_state({"count_limit_up": 10, "count_limit_down": 1}) == "hot"
        assert classify_market_state({"count_limit_up": 3, "count_limit_down": 3}) == "range"

    def test_short_history_falls_back(self):
        assert classify_market_state({"count_limit_up": 1, "count_limit_down": 10}, _history(n=5)) == "ice"


class TestPolicy:
    def test_ice_maps_to_bear(self):
        p = state_policy("ice")
        assert p["market_state"] == "bear"
        assert p["cap_position"] == 0.5
        assert p["veto"] is False

    def test_hot_and_range_normal(self):
        for s in ("hot", "range"):
            p = state_policy(s)
            assert p["market_state"] == "range"
            assert p["cap_position"] == 1.0


class TestVeto:
    def test_extreme_ice_veto(self):
        # 涨停 < 10 且 跌停 ≥ 涨停 → 否决
        assert is_veto("ice", {"count_limit_up": 5, "count_limit_down": 6}) is True
        assert is_veto("ice", {"count_limit_up": 12, "count_limit_down": 30}) is False

    def test_non_ice_never_veto(self):
        assert is_veto("range", {"count_limit_up": 5, "count_limit_down": 6}) is False
        assert is_veto("hot", {"count_limit_up": 5, "count_limit_down": 6}) is False


class TestPanelSentiment:
    def test_board_aware_thresholds(self):
        sent = sentiment_from_panel(_panel())
        assert sent["count_limit_up"] == 12  # 10 main + 2 dual
        assert sent["count_limit_down"] == 3

    def test_history_length(self):
        hist = sentiment_history_from_panel(_panel(), n=5)
        assert len(hist) == 5
        assert {"count_limit_up", "count_limit_down", "market_turnover"} <= set(hist.columns)

    def test_empty_panel_safe(self):
        sent = sentiment_from_panel(pd.DataFrame())
        assert sent == {}
