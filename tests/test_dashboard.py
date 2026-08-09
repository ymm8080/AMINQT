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

    def test_stock_names_covers_pool(self):
        # 真实名称映射 (sw 静态分类), 覆盖清单 + 知名股
        names = ds.stock_names()
        assert len(names) > 1000
        assert names["600519"] == "贵州茅台"
        pool = pd.read_parquet("data/lists/list_20260807.parquet")
        pool["symbol"] = pool["symbol"].astype(str)
        assert pool["symbol"].map(names).notna().all()

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


class TestModulePerfRecent:
    def test_recent_module_ids_per_model(self):
        from app.streamlit import module_perf as mp

        df = pd.DataFrame(
            {
                "module_id": [
                    "legacy·v1",
                    "legacy·v2",
                    "legacy·v3",
                    "parallel·v1",
                    "parallel·v2",
                ],
                "family": ["legacy", "legacy", "legacy", "parallel", "parallel"],
                "date": [
                    "2026-08-01",
                    "2026-08-02",
                    "2026-08-03",
                    "2026-08-01",
                    "2026-08-02",
                ],
            }
        )
        # 每个 family 取最新 n 个版本 (按交付日降序)
        assert mp.recent_module_ids_per_model(df, n=2) == [
            "legacy·v3",
            "legacy·v2",
            "parallel·v2",
            "parallel·v1",
        ]


class TestSelectionPriorityEdit:
    """选股池表格日内买入复选框 → priority.json 的对账逻辑."""

    def test_apply_priority_edits(self):
        from app.streamlit.page_selection import _apply_priority_edits

        edited = {0: {"日内买入": True}, 2: {"日内买入": False}, 1: {"name": "改错列"}}
        symbols = ["600519", "000001", "300750"]
        assert _apply_priority_edits(edited, symbols) == {
            "600519": True,
            "300750": False,
        }

    def test_apply_priority_edits_ignores_out_of_range(self):
        from app.streamlit.page_selection import _apply_priority_edits

        assert _apply_priority_edits({9: {"日内买入": True}}, ["600519"]) == {}

    def test_priority_toggles(self):
        from app.streamlit.page_selection import _priority_toggles

        desired = {"600519": True, "000001": False, "300750": True}
        current = {"600519"}
        # 600519 已标记→不动; 000001 未标记且目标未勾选→不动; 300750 未标记→需勾选
        assert _priority_toggles(desired, current) == ["300750"]
        # 目标与当前一致 → 不 toggle (幂等, 防编辑事件累积重复处理)
        assert _priority_toggles({"600519": True}, {"600519"}) == []
        assert _priority_toggles({"000001": False}, set()) == []


class TestStockPredictionQuery:
    """STOCK_LIST_DIR 预测文件 → 个股预测历史 (含模块标签) + 日期清单 (交付族, 多日期可拼)."""

    def _mkfile(self, tmp_path, fname: str, df: pd.DataFrame) -> None:
        df.to_csv(tmp_path / fname, index=False)

    def test_list_prediction_files_parsing(self, tmp_path):
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000001"], "board": ["main"]}),
        )
        self._mkfile(
            tmp_path,
            "legacy_preds_raw_20260804.csv",
            pd.DataFrame({"symbol": ["000002"]}),
        )
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame(
                {
                    "symbol": ["000003"],
                    "date": ["2026-08-06"],
                    "board": ["main"],
                    "systems": ["sniper"],
                }
            ),
        )
        self._mkfile(
            tmp_path,
            "slowbull_pool_main_20260731__slow_bull_v1_0.csv",
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

    def test_list_prediction_files_numeric_module(self, tmp_path):
        """纯数字模块 (如当前模型 tag `20260807`) 不能被贪婪 prefix 吃掉."""
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260807__20260807.csv",
            pd.DataFrame({"symbol": ["000001"], "board": ["main"]}),
        )
        files = ds.list_prediction_files(str(tmp_path))
        assert len(files) == 1
        assert files[0]["family"] == "legacy"
        assert files[0]["date"] == "20260807"
        assert files[0]["module"] == "20260807"

    def test_load_history_symbol_filter_module_kept(self, tmp_path):
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame(
                {
                    "symbol": ["000001", "999999"],
                    "board": ["main", "main"],
                    "score": [0.5, 0.4],
                    "pred_ret_2d": [0.01, 0.0],
                    "pred_ret_3d": [0.02, 0.0],
                    "pred_ret_5d": [0.03, 0.0],
                    "prob_up": [0.55, 0.5],
                    "prob_up_2d": [0.56, 0.5],
                    "prob_up_3d": [0.57, 0.5],
                    "prob_up_5d": [0.58, 0.5],
                }
            ),
        )
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame(
                {
                    "date": ["2026-08-06"],
                    "board": ["main"],
                    "symbol": ["000001"],
                    "systems": ["fusion+sniper"],
                    "score": [0.8],
                    "rk": [1],
                    "pred_mag_2d": [0.02],
                    "pred_prob_2d": [0.52],
                    "pred_mag_3d": [0.03],
                    "pred_prob_3d": [0.53],
                    "pred_mag_5d": [0.04],
                    "pred_prob_5d": [0.54],
                    "pred_mag_10d": [0.05],
                    "pred_prob_10d": [0.55],
                }
            ),
        )
        hist = ds.load_stock_prediction_history("000001", list_dir=str(tmp_path))
        assert len(hist) == 2  # legacy + parallel, 同日不同模块都保留
        bymod = {r["module"]: r for _, r in hist.iterrows()}
        assert bymod["modA"]["family"] == "legacy"
        assert bymod["modA"]["gain_3d"] == 0.02
        assert bymod["modA"]["prob_3d"] == 0.57
        assert bymod["modB"]["family"] == "parallel"
        assert bymod["modB"]["system"] == "fusion+sniper"
        assert bymod["modB"]["gain_10d"] == 0.05
        assert set(hist["date"]) == {"2026-08-06"}  # 日期统一 YYYY-MM-DD
        assert ds.load_stock_prediction_history("123456", list_dir=str(tmp_path)).empty

    def test_last_five_dates_limit_and_dedup(self, tmp_path):
        for i in range(1, 9):  # 20260801..20260808
            self._mkfile(
                tmp_path,
                f"legacy_stocklist_2026080{i}__modA.csv",
                pd.DataFrame(
                    {
                        "symbol": ["000001"],
                        "board": ["main"],
                        "score": [0.5],
                        "pred_ret_2d": [0.01],
                        "pred_ret_3d": [0.02],
                        "pred_ret_5d": [0.03],
                        "prob_up": [0.55],
                        "prob_up_2d": [0.56],
                        "prob_up_3d": [0.57],
                        "prob_up_5d": [0.58],
                    }
                ),
            )
        # 同 date+module 的 raw 底稿 → 去重保留交付族 (gain_3d=0.02, 非 raw 0.021)
        self._mkfile(
            tmp_path,
            "legacy_preds_raw_20260808__modA.csv",
            pd.DataFrame(
                {
                    "symbol": ["000001"],
                    "pred_ret_2d": [0.011],
                    "pred_ret_3d": [0.021],
                    "pred_ret_5d": [0.031],
                    "prob_up": [0.551],
                    "prob_up_2d": [0.561],
                    "prob_up_3d": [0.571],
                    "prob_up_5d": [0.581],
                }
            ),
        )
        # 同 symbol+module 多 cut 行 → 去重保留 rk 最小一行
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260808__modB.csv",
            pd.DataFrame(
                {
                    "date": ["2026-08-08", "2026-08-08"],
                    "board": ["main", "main"],
                    "symbol": ["000001", "000001"],
                    "systems": ["fusion", "fusion"],
                    "score": [0.8, 0.8],
                    "rk": [1, 2],
                    "pred_mag_2d": [0.02, 0.02],
                    "pred_prob_2d": [0.52, 0.52],
                    "pred_mag_3d": [0.03, 0.03],
                    "pred_prob_3d": [0.53, 0.53],
                    "pred_mag_5d": [0.04, 0.04],
                    "pred_prob_5d": [0.54, 0.54],
                    "pred_mag_10d": [0.05, 0.05],
                    "pred_prob_10d": [0.55, 0.55],
                }
            ),
        )
        hist = ds.load_stock_prediction_history(
            "000001", list_dir=str(tmp_path), max_dates=5
        )
        assert sorted(set(hist["date"])) == [
            "2026-08-04",
            "2026-08-05",
            "2026-08-06",
            "2026-08-07",
            "2026-08-08",
        ]
        aug8 = hist[hist["date"] == "2026-08-08"]
        assert len(aug8) == 2
        assert set(aug8["family"]) == {"legacy", "parallel"}
        par = aug8[aug8["family"] == "parallel"].iloc[0]
        assert par["rk"] == 1
        leg = aug8[aug8["family"] == "legacy"].iloc[0]
        assert leg["gain_3d"] == 0.02  # 交付族优先, 非 raw

    def test_list_prediction_dates_sorted_desc(self, tmp_path):
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame({"symbol": ["000001"], "date": ["2026-08-06"]}),
        )
        self._mkfile(
            tmp_path,
            "slowbull_pool_main_20260731__slow_bull_v1_0.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        assert ds.list_prediction_dates(str(tmp_path)) == [
            "20260806",
            "20260805",
            "20260731",
        ]

    def test_load_stock_list_on_date(self, tmp_path):
        # 交付族 + raw 同 symbol+module → 只列交付族, raw 全排除 (含仅存在于 raw 的股票)
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame(
                {
                    "symbol": ["000001", "000002"],
                    "board": ["main", "main"],
                    "score": [0.5, 0.4],
                    "pred_ret_2d": [0.01, 0.02],
                    "pred_ret_3d": [0.02, 0.03],
                    "pred_ret_5d": [0.03, 0.04],
                    "prob_up": [0.55, 0.53],
                    "prob_up_2d": [0.56, 0.54],
                    "prob_up_3d": [0.57, 0.55],
                    "prob_up_5d": [0.58, 0.56],
                }
            ),
        )
        self._mkfile(
            tmp_path,
            "legacy_preds_raw_20260806__modA.csv",
            pd.DataFrame(
                {
                    "symbol": ["000001", "000004"],  # 000004 仅存在于 raw → 不显示
                    "pred_ret_2d": [0.011, 0.0],
                    "pred_ret_3d": [0.021, 0.0],
                    "pred_ret_5d": [0.031, 0.0],
                    "prob_up": [0.551, 0.5],
                    "prob_up_2d": [0.561, 0.5],
                    "prob_up_3d": [0.571, 0.5],
                    "prob_up_5d": [0.581, 0.5],
                }
            ),
        )
        # parallel 多 cut → 去重保留 rk 最小
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260806__modB.csv",
            pd.DataFrame(
                {
                    "date": ["2026-08-06", "2026-08-06"],
                    "board": ["main", "main"],
                    "symbol": ["000003", "000003"],
                    "systems": ["fusion", "fusion"],
                    "score": [0.8, 0.8],
                    "rk": [1, 2],
                    "pred_mag_2d": [0.02, 0.02],
                    "pred_prob_2d": [0.52, 0.52],
                    "pred_mag_3d": [0.03, 0.03],
                    "pred_prob_3d": [0.53, 0.53],
                    "pred_mag_5d": [0.04, 0.04],
                    "pred_prob_5d": [0.54, 0.54],
                    "pred_mag_10d": [0.05, 0.05],
                    "pred_prob_10d": [0.55, 0.55],
                }
            ),
        )
        # 其它日期不应混入
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260805__modA.csv",
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
            tmp_path,
            "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame({"symbol": ["000001"]}),
        )
        self._mkfile(
            tmp_path,
            "parallel_shortlist_20260805__modB.csv",
            pd.DataFrame({"symbol": ["000002"], "date": ["2026-08-05"]}),
        )
        rows = ds.load_stock_list_on_dates(
            ["20260806", "20260805"], list_dir=str(tmp_path)
        )
        assert set(rows["date"]) == {"2026-08-05", "2026-08-06"}
        assert len(rows) == 2
        assert ds.load_stock_list_on_dates(["20991231"], list_dir=str(tmp_path)).empty

    def test_legacy_10d_normalized(self):
        # legacy 交付文件含 pred_ret_10d/prob_up_10d, 归一化必须保留 (看板 10d 预期)
        df = pd.DataFrame(
            {
                "symbol": ["000001"],
                "board": ["main"],
                "pred_ret_2d": [0.01],
                "pred_ret_3d": [0.02],
                "pred_ret_5d": [0.03],
                "pred_ret_10d": [0.05],
                "prob_up_2d": [0.51],
                "prob_up_3d": [0.52],
                "prob_up_5d": [0.53],
                "prob_up_10d": [0.55],
            }
        )
        norm = ds._normalize_pred_rows("legacy", "20260807", "modA", df)
        assert norm["gain_10d"].iloc[0] == 0.05
        assert norm["prob_10d"].iloc[0] == 0.55

    def test_parallel_gain_prefers_pred_ret_average_over_pred_mag_mfe(self):
        # 2026-08-09 用户: 看板预期列显示平均预测 (pred_ret_, close-to-close), 非 MFE 最大.
        # 有 pred_ret_ 列 → 用平均; 无 → 回退 pred_mag_.
        with_avg = pd.DataFrame(
            {
                "symbol": ["000001"],
                "board": ["main"],
                "systems": ["fusion"],
                "score": [0.8],
                "pred_mag_3d": [0.12],
                "pred_mag_5d": [0.17],
                "pred_mag_10d": [0.28],
                "pred_ret_3d": [0.012],
                "pred_ret_5d": [0.025],
                "pred_ret_10d": [0.03],
                "pred_prob_3d": [0.53],
                "pred_prob_5d": [0.54],
                "pred_prob_10d": [0.55],
            }
        )
        norm_avg = ds._normalize_pred_rows("parallel", "20260807", "modA", with_avg)
        assert norm_avg["gain_3d"].iloc[0] == 0.012  # 平均预测优先
        assert norm_avg["gain_5d"].iloc[0] == 0.025
        assert norm_avg["gain_10d"].iloc[0] == 0.03
        assert norm_avg["prob_3d"].iloc[0] == 0.53  # 概率不变

        # 旧交付无 pred_ret_ → 回退 pred_mag_ (兼容)
        no_avg = with_avg.drop(columns=["pred_ret_3d", "pred_ret_5d", "pred_ret_10d"])
        norm_fb = ds._normalize_pred_rows("parallel", "20260807", "modA", no_avg)
        assert norm_fb["gain_3d"].iloc[0] == 0.12
        assert norm_fb["gain_5d"].iloc[0] == 0.17
        assert norm_fb["gain_10d"].iloc[0] == 0.28

    def test_load_official_run_shortlist(self, tmp_path):
        # 选股看板股票池源: 最新交付短名单, 去 *_raw, 默认最新日期
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260805__modA.csv",
            pd.DataFrame({"symbol": ["000001"], "board": ["main"]}),
        )
        self._mkfile(
            tmp_path,
            "legacy_stocklist_20260806__modA.csv",
            pd.DataFrame({"symbol": ["000002"], "board": ["main"]}),
        )
        self._mkfile(
            tmp_path,
            "legacy_preds_raw_20260806__modA.csv",
            pd.DataFrame({"symbol": ["000004"]}),
        )
        # 默认最新日期 (20260806), 且不含 raw
        df, date = ds.load_official_run_shortlist(list_dir=str(tmp_path))
        assert date == "20260806"
        assert df["symbol"].tolist() == ["000002"]
        # 指定旧日期
        df2, date2 = ds.load_official_run_shortlist("20260805", list_dir=str(tmp_path))
        assert date2 == "20260805"
        assert df2["symbol"].tolist() == ["000001"]
        # 空目录 → (None, None)
        assert ds.load_official_run_shortlist(list_dir=str(tmp_path / "empty")) == (
            None,
            None,
        )


class TestPageImports:
    def test_pages_importable(self):
        import app.streamlit.page_config
        import app.streamlit.page_selection
        import app.streamlit.page_trading  # noqa: F401

    def test_entry_importable(self):
        import app.streamlit_app  # noqa: F401
