# -*- coding: utf-8 -*-
"""P12: MarketContext 大盘因子测试 (6 列 / merge 无错位 / 空值处理)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.market_context import MarketContext


def make_index(days=60, seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    close = 3000 * np.cumprod(1 + rng.normal(0.0003, 0.01, days))
    return pd.DataFrame({"date": dates, "close": close})


class TestMarketContext:
    def test_six_factor_columns(self):
        ctx = MarketContext()
        ctx.load_from_df(make_index())
        factors = ctx.compute_factors()
        assert len(MarketContext.get_factor_columns()) == 6
        for col in MarketContext.get_factor_columns():
            assert col in factors.columns
        assert len(factors) == 60

    def test_factor_values_sane(self):
        ctx = MarketContext()
        ctx.load_from_df(make_index())
        factors = ctx.compute_factors()
        assert factors["market_above_ma20"].isin([0.0, 1.0]).all()
        assert (factors["market_volatility"] >= 0).all()
        assert factors["market_return_1d"].abs().max() < 0.2

    def test_merge_no_misalignment(self):
        """按日期 merge: 因子值与当日指数计算值一致, 不错位."""
        idx = make_index()
        ctx = MarketContext()
        ctx.load_from_df(idx)
        factors = ctx.compute_factors()
        stock = pd.DataFrame({"date": idx["date"], "close": 100.0})
        merged = ctx.merge_to_stock(stock)
        for col in MarketContext.get_factor_columns():
            pd.testing.assert_series_equal(
                merged[col].reset_index(drop=True),
                factors[col].reset_index(drop=True), check_names=False)

    def test_missing_dates_ffill(self):
        """个股缺交易日 → ffill 前值, 不留 NaN."""
        idx = make_index()
        ctx = MarketContext()
        ctx.load_from_df(idx)
        ctx.compute_factors()
        stock = pd.DataFrame({"date": [idx["date"][0], idx["date"][30]], "close": 1.0})
        merged = ctx.merge_to_stock(stock)
        assert not merged.isna().any().any()
