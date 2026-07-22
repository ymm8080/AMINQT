# -*- coding: utf-8 -*-
"""RightSideFilter 单元测试 (P10.8b, ARCH §5.13.7.B)."""

import numpy as np
import pandas as pd
import pytest

from app.core.right_side_filter import RightSideFilter


def _make_trend_df(
    n: int = 60, slope: float = 0.1, amount: float = 1e8, noise: float = 0.005
) -> pd.DataFrame:
    """构造已知趋势的合成日线 (slope>0 上行, slope<0 下行)."""
    rng = np.random.default_rng(7)
    t = np.arange(n)
    close = 10.0 + slope * t + rng.normal(0.0, noise, n)
    close = np.maximum(close, 0.5)
    volume = amount / close
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
            "amount": np.full(n, amount),
        }
    )


@pytest.fixture()
def filt() -> RightSideFilter:
    return RightSideFilter()


class TestIsUptrend:
    def test_uptrend_passes(self, filt):
        df = _make_trend_df(slope=0.1)
        assert filt.is_uptrend(df) is True

    def test_downtrend_fails(self, filt):
        df = _make_trend_df(slope=-0.1)
        assert filt.is_uptrend(df) is False

    def test_sideways_fails(self, filt):
        df = _make_trend_df(slope=0.0, noise=0.02)
        assert filt.is_uptrend(df) is False

    def test_insufficient_data_fails(self, filt):
        df = _make_trend_df(n=15, slope=0.1)
        assert filt.is_uptrend(df) is False

    def test_low_amount_fails(self, filt):
        df = _make_trend_df(slope=0.1, amount=1e7)  # 1000 万 < 5000 万
        assert filt.is_uptrend(df) is False

    def test_amount_fallback_close_times_volume(self, filt):
        """amount 列缺失时用 close*volume 近似."""
        df = _make_trend_df(slope=0.1).drop(columns=["amount"])
        assert filt.is_uptrend(df) is True

    def test_market_requirement(self):
        f = RightSideFilter(require_market_above_ma20=True)
        df = _make_trend_df(slope=0.1)
        assert f.is_uptrend(df, market_above_ma20=True) is True
        assert f.is_uptrend(df, market_above_ma20=False) is False

    def test_market_not_required_by_default(self, filt):
        df = _make_trend_df(slope=0.1)
        assert filt.is_uptrend(df, market_above_ma20=False) is True

    def test_ma_alignment_broken_fails(self, filt):
        """尾部急跌破坏多头排列 → 非上行."""
        df = _make_trend_df(slope=0.1)
        df.loc[df.index[-3:], "close"] = df["close"].iloc[-4] * 0.90
        df.loc[df.index[-3:], "amount"] = 1e8
        assert filt.is_uptrend(df) is False

    def test_nan_in_close_handled(self, filt):
        df = _make_trend_df(slope=0.1)
        df.loc[10, "close"] = np.nan
        result = filt.is_uptrend(df)
        assert isinstance(result, bool)


class TestBatchFilter:
    def test_batch_returns_dict(self, filt):
        pool = {
            "UP": _make_trend_df(slope=0.1),
            "DOWN": _make_trend_df(slope=-0.1),
            "ILLIQ": _make_trend_df(slope=0.1, amount=1e6),
        }
        result = filt.batch_filter(pool)
        assert result == {"UP": True, "DOWN": False, "ILLIQ": False}

    def test_batch_empty_df(self, filt):
        result = filt.batch_filter({"EMPTY": pd.DataFrame()})
        assert result == {"EMPTY": False}

    def test_custom_ma_periods(self):
        f = RightSideFilter(ma_short=5, ma_mid=10, ma_long=15)
        df = _make_trend_df(slope=0.1)
        assert f.is_uptrend(df) is True
