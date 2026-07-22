# -*- coding: utf-8 -*-
"""intraday_factor_engine 单元测试 (ARCH §5.10, 25 维)."""

import numpy as np
import pandas as pd
import pytest

from app.core.intraday_factor_engine import (
    ALL_FACTOR_NAMES,
    MARKET_FACTOR_NAMES,
    SECTOR_FACTOR_NAMES,
    STOCK_FACTOR_NAMES,
    TOTAL_DIM,
    IntradayFactorEngine,
)


def _make_bars(n=48, start_price=10.0, step=0.05, base_vol=1000.0, last5_vol=2000.0):
    """构造线性上涨 + 末 5 根放量的 5 分钟 K 线."""
    close = start_price + step * np.arange(n)
    open_ = close - step / 2
    high = close + 0.02
    low = open_ - 0.02
    volume = np.full(n, base_vol)
    volume[-5:] = last5_vol
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def engine():
    return IntradayFactorEngine({"prev_close": 9.80})


@pytest.fixture
def stock_bars():
    return _make_bars()


class TestStockFactors:
    def test_sixteen_dims_all_finite(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        assert set(out.keys()) == set(STOCK_FACTOR_NAMES)
        assert len(out) == 16
        assert all(np.isfinite(v) for v in out.values())

    def test_momentum_5_exact(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        c = stock_bars["close"].to_numpy()
        expected = c[-1] / c[-6] - 1.0
        assert out["intraday_momentum_5"] == pytest.approx(expected, rel=1e-9)
        expected15 = c[-1] / c[-16] - 1.0
        assert out["intraday_momentum_15"] == pytest.approx(expected15, rel=1e-9)

    def test_uptrend_signatures(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        assert out["intraday_momentum_5"] > 0
        assert out["intraday_trend_strength"] > 0
        assert out["intraday_buy_pressure"] > 0
        assert out["intraday_position_in_range"] > 0.9  # 收盘接近日内最高
        # 末 5 根放量 → 量比 > 1
        assert out["intraday_volume_ratio"] > 1.0

    def test_volume_ratio_exact(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        # v5 = 2000, v20 = (5*2000 + 15*1000)/20 = 1250
        assert out["intraday_volume_ratio"] == pytest.approx(1.6, rel=1e-9)

    def test_vwap_dev(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        c = stock_bars["close"].to_numpy()
        v = stock_bars["volume"].to_numpy()
        vwap = float((c * v).sum() / v.sum())
        expected = (c[-1] - vwap) / vwap
        assert out["intraday_vwap_dev"] == pytest.approx(expected, rel=1e-9)

    def test_empty_bars_returns_zeros(self, engine):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = engine.compute_stock_factors(empty)
        assert len(out) == 16
        assert all(v == 0.0 for v in out.values())

    def test_missing_column_raises(self, engine):
        bad = pd.DataFrame({"close": [1.0]})
        with pytest.raises(KeyError):
            engine.compute_stock_factors(bad)

    def test_snapshot_appended(self, engine, stock_bars):
        base = engine.compute_stock_factors(stock_bars)
        snap = {"price": 99.0, "volume": 5000.0, "amount": 99.0 * 5000.0}
        with_snap = engine.compute_stock_factors(stock_bars, snap)
        # 快照把最后价拉到 99 → 动量/区间位置显著变化
        assert with_snap["intraday_momentum_5"] > base["intraday_momentum_5"]
        assert with_snap["intraday_position_in_range"] == pytest.approx(1.0)

    def test_open_strength_uses_prev_close(self, engine, stock_bars):
        out = engine.compute_stock_factors(stock_bars)
        # gap = open_0/prev_close - 1 > 0 (open_0=9.975, prev_close=9.80)
        assert out["intraday_open_strength"] > 0


class TestMarketFactors:
    def test_five_dims(self, engine):
        out = engine.compute_market_factors(_make_bars())
        assert set(out.keys()) == set(MARKET_FACTOR_NAMES)
        assert len(out) == 5
        assert all(np.isfinite(v) for v in out.values())
        assert out["market_5min_momentum"] > 0

    def test_breadth_default_and_config(self):
        bars = _make_bars()
        out = IntradayFactorEngine().compute_market_factors(bars)
        assert out["market_5min_breadth"] == 1.0
        out = IntradayFactorEngine({"market_breadth": 0.3}).compute_market_factors(bars)
        assert out["market_5min_breadth"] == 0.3

    def test_breadth_column_overrides(self):
        bars = _make_bars()
        bars["breadth"] = 0.7
        out = IntradayFactorEngine({"market_breadth": 0.3}).compute_market_factors(bars)
        assert out["market_5min_breadth"] == 0.7

    def test_empty_bars(self, engine):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = engine.compute_market_factors(empty)
        assert len(out) == 5
        assert out["market_5min_momentum"] == 0.0


class TestSectorFactors:
    def test_four_dims(self, engine, stock_bars):
        sector = _make_bars(step=0.02)
        out = engine.compute_sector_factors(sector, stock_bars)
        assert set(out.keys()) == set(SECTOR_FACTOR_NAMES)
        assert all(np.isfinite(v) for v in out.values())

    def test_dev_uses_stock_bars(self, engine, stock_bars):
        # 个股涨速 (step=0.05) > 板块 (step=0.01) → dev > 0
        sector = _make_bars(step=0.01)
        out = engine.compute_sector_factors(sector, stock_bars)
        assert out["sector_5min_dev"] > 0

    def test_flow_proxy(self, engine):
        sector = _make_bars()  # close > open 每根 → 净流入代理 > 0
        out = engine.compute_sector_factors(sector)
        assert out["sector_5min_flow"] > 0

    def test_empty_bars(self, engine):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        out = engine.compute_sector_factors(empty)
        assert all(v == 0.0 for v in out.values())


class TestCompute5Min:
    def test_vector_shape_and_finite(self, engine, stock_bars):
        vec = engine.compute_5min(
            stock_bars, _make_bars(step=0.02), _make_bars(step=0.01)
        )
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (TOTAL_DIM,) == (25,)
        assert np.isfinite(vec).all()

    def test_vector_order(self, engine, stock_bars):
        index_bars = _make_bars(step=0.02)
        sector_bars = _make_bars(step=0.01)
        vec = engine.compute_5min(stock_bars, index_bars, sector_bars)
        stock = engine.compute_stock_factors(stock_bars)
        market = engine.compute_market_factors(index_bars)
        sector = engine.compute_sector_factors(sector_bars, stock_bars)
        expected = np.array(
            [stock[k] for k in STOCK_FACTOR_NAMES]
            + [market[k] for k in MARKET_FACTOR_NAMES]
            + [sector[k] for k in SECTOR_FACTOR_NAMES]
        )
        assert np.allclose(vec, expected)

    def test_factor_name_registry(self):
        assert len(ALL_FACTOR_NAMES) == 25
        assert len(set(ALL_FACTOR_NAMES)) == 25  # 无重名

    def test_all_empty_still_25_finite(self, engine):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        vec = engine.compute_5min(empty, empty, empty)
        assert vec.shape == (25,)
        assert np.isfinite(vec).all()
