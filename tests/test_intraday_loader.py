# -*- coding: utf-8 -*-
"""intraday_loader 单元测试 (akshare 缺失 — monkeypatch 假模块)."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from app.core.intraday_loader import IntradayLoader


def _fake_minute_df():
    """伪造 akshare stock_zh_a_minute 返回 (英文列, day 为时间)."""
    n = 10
    return pd.DataFrame(
        {
            "day": pd.date_range("2026-07-21 09:31", periods=n, freq="1min"),
            "open": np.linspace(10, 11, n),
            "high": np.linspace(10.1, 11.1, n),
            "low": np.linspace(9.9, 10.9, n),
            "close": np.linspace(10.05, 11.05, n),
            "volume": np.full(n, 1000.0),
        }
    )


def _fake_hist_min_df():
    """伪造 akshare stock_zh_a_hist_min_em 返回 (中文列)."""
    n = 96  # 2 天 × 48 根
    return pd.DataFrame(
        {
            "时间": pd.date_range("2026-07-20 09:35", periods=n, freq="5min"),
            "开盘": np.linspace(10, 12, n),
            "收盘": np.linspace(10.05, 12.05, n),
            "最高": np.linspace(10.1, 12.1, n),
            "最低": np.linspace(9.95, 11.95, n),
            "成交量": np.full(n, 5000.0),
            "成交额": np.full(n, 50000.0),
        }
    )


@pytest.fixture
def fake_akshare(monkeypatch):
    """注入假 akshare 模块, 带调用计数."""
    fake = types.ModuleType("akshare")
    fake.calls = {"minute": 0, "hist_min": 0}

    def stock_zh_a_minute(symbol, period="1", adjust=""):
        fake.calls["minute"] += 1
        return _fake_minute_df()

    def stock_zh_a_hist_min_em(
        symbol, period="5", start_date=None, end_date=None, adjust=""
    ):
        fake.calls["hist_min"] += 1
        return _fake_hist_min_df()

    fake.stock_zh_a_minute = stock_zh_a_minute
    fake.stock_zh_a_hist_min_em = stock_zh_a_hist_min_em
    monkeypatch.setitem(sys.modules, "akshare", fake)
    return fake


@pytest.fixture
def loader(tmp_path):
    return IntradayLoader(cache_dir=str(tmp_path / "5min"))


class TestRealtime:
    def test_load_and_normalize(self, loader, fake_akshare):
        df = loader.load_realtime("sh600000")
        assert list(df.columns) == [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        assert len(df) == 10
        assert df["datetime"].is_monotonic_increasing
        assert fake_akshare.calls["minute"] == 1

    def test_memory_cache_per_day(self, loader, fake_akshare):
        loader.load_realtime("sh600000")
        loader.load_realtime("sh600000")
        assert fake_akshare.calls["minute"] == 1  # 第二次走缓存
        loader.load_realtime("sh600000", use_cache=False)
        assert fake_akshare.calls["minute"] == 2  # 强制刷新

    def test_clear_cache(self, loader, fake_akshare):
        loader.load_realtime("sh600000")
        loader.clear_realtime_cache("sh600000")
        loader.load_realtime("sh600000")
        assert fake_akshare.calls["minute"] == 2

    def test_cache_keyed_by_symbol(self, loader, fake_akshare):
        loader.load_realtime("sh600000")
        loader.load_realtime("sz000001")
        assert fake_akshare.calls["minute"] == 2


class TestHistoryMin:
    def test_fetch_and_parquet_cache(self, loader, fake_akshare, tmp_path):
        df = loader.load_history_min("600000", period="5")
        assert len(df) == 96
        assert fake_akshare.calls["hist_min"] == 1
        # Parquet 已落盘
        cache_file = tmp_path / "5min" / "600000_5min.parquet"
        assert cache_file.exists()
        # 第二次调用走缓存, 不再请求 akshare
        df2 = loader.load_history_min("600000", period="5")
        assert fake_akshare.calls["hist_min"] == 1
        assert len(df2) == 96

    def test_chinese_columns_normalized(self, loader, fake_akshare):
        df = loader.load_history_min("600000", period="5")
        assert {"datetime", "open", "close", "high", "low", "volume", "amount"} <= set(
            df.columns
        )
        assert not {"时间", "开盘", "收盘"} & set(df.columns)

    def test_start_end_filter(self, loader, fake_akshare):
        loader.load_history_min("600000", period="5")  # 落盘
        df = loader.load_history_min(
            "600000",
            period="5",
            start="2026-07-20 12:00",
            end="2026-07-20 15:00",
        )
        assert (df["datetime"] >= pd.Timestamp("2026-07-20 12:00")).all()
        assert (df["datetime"] <= pd.Timestamp("2026-07-21 00:00")).all()
        assert 0 < len(df) < 96


class TestAkshareMissing:
    def test_realtime_raises_runtime_error(self, loader, monkeypatch):
        monkeypatch.setitem(sys.modules, "akshare", None)  # import → ImportError
        with pytest.raises(RuntimeError, match="pip install akshare"):
            loader.load_realtime("sh600000")

    def test_history_raises_runtime_error(self, loader, monkeypatch):
        monkeypatch.setitem(sys.modules, "akshare", None)
        with pytest.raises(RuntimeError, match="pip install akshare"):
            loader.load_history_min("600000")
