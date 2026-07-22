# -*- coding: utf-8 -*-
"""P10 看板测试: 数据服务层纯函数 + 图表组件 + 页面导入."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.list_generator import SCHEMA_FIELDS
from app.streamlit import data_service as ds


class TestDataService:
    def test_demo_list_schema_compatible(self):
        df = ds.demo_list()
        for col in SCHEMA_FIELDS:
            assert col in df.columns, col
        assert (df["schema_version"] == "1.0").all()
        assert df["prob_up"].between(0, 1).all()

    def test_demo_ohlc_and_intraday(self):
        df = ds.demo_ohlc("600519", days=60)
        assert len(df) == 60
        assert (df["high"] >= df["low"]).all()
        intra = ds.demo_intraday("600519")
        assert len(intra) == 120

    def test_list_roundtrip(self, tmp_path):
        df = ds.demo_list()
        df.to_parquet(tmp_path / "list_20260722.parquet", index=False)
        assert ds.list_available_dates(str(tmp_path)) == ["20260722"]
        loaded, date = ds.load_latest_list(str(tmp_path))
        assert date == "20260722" and len(loaded) == len(df)
        assert ds.load_list("20990101", str(tmp_path)) is None

    def test_watchlist_toggle(self, tmp_path):
        path = str(tmp_path / "watchlist.json")
        assert ds.toggle_watchlist("600519", "贵州茅台", path) is True
        items = ds.load_watchlist(path)
        assert items[0]["symbol"] == "600519"
        assert ds.toggle_watchlist("600519", path=path) is False
        assert ds.load_watchlist(path) == []

    def test_yaml_roundtrip(self, tmp_path):
        path = str(tmp_path / "cfg.yaml")
        ds.save_yaml({"a": 1, "b": [1, 2]}, path)
        assert ds.load_yaml(path) == {"a": 1, "b": [1, 2]}
        assert ds.load_yaml(str(tmp_path / "none.yaml")) == {}

    def test_tuning_report_missing(self, tmp_path):
        assert ds.load_tuning_report(str(tmp_path / "none.json")) is None


class TestComponents:
    def test_chart_builders(self):
        from app.streamlit.components import (
            drawdown_chart,
            equity_curve,
            factor_radar,
            intraday_chart,
            kline_chart,
        )

        ohlc = ds.demo_ohlc("600519", days=80)
        assert kline_chart(ohlc) is not None
        assert intraday_chart(ds.demo_intraday("600519"), prev_close=100) is not None
        nav = pd.DataFrame({"date": ohlc["date"], "nav": np.linspace(1e6, 1.1e6, 80)})
        assert equity_curve(nav) is not None
        assert drawdown_chart(nav) is not None
        assert factor_radar({"MACD": 0.5, "RSI": 0.3}) is not None


class TestPageImports:
    def test_pages_importable(self):
        import app.streamlit.page_backtest  # noqa: F401
        import app.streamlit.page_config  # noqa: F401
        import app.streamlit.page_selection  # noqa: F401
        import app.streamlit.page_trading  # noqa: F401

    def test_entry_importable(self):
        import app.streamlit_app  # noqa: F401
