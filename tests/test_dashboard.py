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
        assert (df["schema_version"] == "1.4").all()
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


class TestStockPredictionQuery:
    """STOCK_LIST_DIR 预测文件 → 日期清单 (交付族, 多日期可拼)."""

    def _mkfile(self, tmp_path, fname: str, df: pd.DataFrame) -> None:
        df.to_csv(tmp_path / fname, index=False)

    def test_list_prediction_files_parsing(self, tmp_path):
        self._mkfile(
            tmp_path, "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000001"], "board": ["main"]}),
        )
        self._mkfile(
            tmp_path, "legacy_preds_raw_20260804.csv",
            pd.DataFrame({"symbol": ["000002"]}),
        )
        self._mkfile(
            tmp_path, "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame({"symbol": ["000003"], "date": ["2026-08-06"],
                          "board": ["main"], "systems": ["sniper"]}),
        )
        self._mkfile(
            tmp_path, "slowbull_pool_main_20260731__slow_bull_v1_0.csv",
            pd.DataFrame({"symbol": ["000004"], "board": ["main"]}),
        )
        (tmp_path / "STOCK LIST 20260806.csv").write_text(
            "x,y\n1,2\n", encoding="utf-8"
        )
        files = ds.list_prediction_files(str(tmp_path))
        parsed = {(f["family"], f["date"], f["module"]) for f in files}
        assert ("legacy", "20260805", "modA") in parsed
        assert ("legacy_raw", "20260804", "na") in parsed
        assert ("parallel", "20260806", "modB") in parsed
        assert ("slow_bull", "20260731", "slow_bull_v1_0") in parsed
        assert len(files) == 4  # 非预测文件被跳过

    def test_list_prediction_dates_sorted_desc(self, tmp_path):
        self._mkfile(
            tmp_path, "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        self._mkfile(
            tmp_path, "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame({"symbol": ["000001"], "date": ["2026-08-06"]}),
        )
        self._mkfile(
            tmp_path, "slowbull_pool_main_20260731__slow_bull_v1_0.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        assert ds.list_prediction_dates(str(tmp_path)) == [
            "20260806", "20260805", "20260731",
        ]

    def test_load_stock_list_on_date(self, tmp_path):
        # 交付族 + raw 同 symbol+module → 只列交付族, raw 全排除 (含仅存在于 raw 的股票)
        self._mkfile(
            tmp_path, "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame({
                "symbol": ["000001", "000002"], "board": ["main", "main"],
                "score": [0.5, 0.4],
                "pred_ret_2d": [0.01, 0.02], "pred_ret_3d": [0.02, 0.03],
                "pred_ret_5d": [0.03, 0.04],
                "prob_up": [0.55, 0.53], "prob_up_2d": [0.56, 0.54],
                "prob_up_3d": [0.57, 0.55], "prob_up_5d": [0.58, 0.56],
            }),
        )
        self._mkfile(
            tmp_path, "legacy_preds_raw_20260806__modA.csv",
            pd.DataFrame({
                "symbol": ["000001", "000004"],  # 000004 仅存在于 raw → 不显示
                "pred_ret_2d": [0.011, 0.0], "pred_ret_3d": [0.021, 0.0],
                "pred_ret_5d": [0.031, 0.0],
                "prob_up": [0.551, 0.5], "prob_up_2d": [0.561, 0.5],
                "prob_up_3d": [0.571, 0.5], "prob_up_5d": [0.581, 0.5],
            }),
        )
        # parallel 多 cut → 去重保留 rk 最小
        self._mkfile(
            tmp_path, "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame({
                "date": ["2026-08-06", "2026-08-06"], "board": ["main", "main"],
                "symbol": ["000003", "000003"], "systems": ["fusion", "fusion"],
                "score": [0.8, 0.8], "rk": [1, 2],
                "pred_mag_2d": [0.02, 0.02], "pred_prob_2d": [0.52, 0.52],
                "pred_mag_3d": [0.03, 0.03], "pred_prob_3d": [0.53, 0.53],
                "pred_mag_5d": [0.04, 0.04], "pred_prob_5d": [0.54, 0.54],
                "pred_mag_10d": [0.05, 0.05], "pred_prob_10d": [0.55, 0.55],
            }),
        )
        # 其它日期不应混入
        self._mkfile(
            tmp_path, "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000009"]}),
        )
        rows = ds.load_stock_list_on_date("20260806", list_dir=str(tmp_path))
        assert set(rows["date"]) == {"2026-08-06"}
        assert len(rows) == 3  # 000001/000002 legacy + 000003 parallel
        bysym = {r["symbol"]: r for _, r in rows.iterrows()}
        assert bysym["000001"]["family"] == "legacy"
        assert bysym["000001"]["gain_3d"] == 0.02  # 交付族优先, 非 raw 0.021
        assert bysym["000002"]["family"] == "legacy"
        assert bysym["000003"]["family"] == "parallel"
        assert bysym["000003"]["rk"] == 1  # 多 cut 去重保留 rk 最小
        assert "000004" not in bysym  # 仅存在于 raw → 排除
        assert ds.load_stock_list_on_date("20991231", list_dir=str(tmp_path)).empty

    def test_load_stock_list_on_dates(self, tmp_path):
        self._mkfile(
            tmp_path, "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        self._mkfile(
            tmp_path, "parallel_shortlist_20260805__modB.csv",
            pd.DataFrame({"symbol": ["000002"], "date": ["2026-08-05"]}),
        )
        rows = ds.load_stock_list_on_dates(
            ["20260806", "20260805"], list_dir=str(tmp_path)
        )
        assert set(rows["date"]) == {"2026-08-05", "2026-08-06"}
        assert len(rows) == 2
        assert ds.load_stock_list_on_dates(["20991231"], list_dir=str(tmp_path)).empty


class TestPageImports:
    def test_pages_importable(self):
        import app.streamlit.page_backtest
        import app.streamlit.page_config
        import app.streamlit.page_selection
        import app.streamlit.page_trading  # noqa: F401

    def test_entry_importable(self):
        import app.streamlit_app  # noqa: F401
