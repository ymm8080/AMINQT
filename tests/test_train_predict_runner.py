# -*- coding: utf-8 -*-
"""训练/预测编排 (panel_builder + train_runner + predict_runner) 测试."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.pipeline1.dual_track_trainer as dtt
from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.list_generator import SCHEMA_FIELDS
from app.pipeline1.panel_builder import assemble_panel, enrich_panel
from app.pipeline1.predict_runner import find_bundles, run_prediction
from app.pipeline1.train_runner import prepare_board_frame, run_training
from tests.test_daily_pipeline import make_panel


# ---------------- 数据源级联 + 硬超时 ----------------
class TestFetchCascade:
    def test_fallback_order_and_down_flag(self, tmp_path):
        """akshare 失败 → sina; 失败后本次运行内跳过该源."""
        from app.pipeline1.data_supply import DataSupplyChain

        supply = DataSupplyChain(cache_dir=str(tmp_path))
        calls = []
        good = pd.DataFrame({"symbol": ["600519"], "date": [pd.Timestamp("2026-07-24")]})

        def fail(symbol, start, end):
            calls.append("fail")
            raise ConnectionError("boom")

        supply._akshare_fetch_hist = fail
        supply._sina_fetch_hist = lambda s, a, b: (calls.append("sina"), good)[1]
        df = supply._default_fetch_hist("600519", "2023-01-01", "2026-07-24")
        assert len(df) == 1 and calls == ["fail", "sina"]
        # 第二次: akshare 已被标记 down, 直走 sina
        supply._default_fetch_hist("600519", "2023-01-01", "2026-07-24")
        assert calls == ["fail", "sina", "sina"]

    def test_all_sources_fail_raises(self, tmp_path):
        from app.pipeline1.data_supply import DataSupplyChain, DataSupplyError

        supply = DataSupplyChain(cache_dir=str(tmp_path))
        for name in ("_akshare_fetch_hist", "_sina_fetch_hist", "_baostock_fetch_hist"):
            setattr(supply, name, lambda s, a, b: (_ for _ in ()).throw(ConnectionError("x")))
        with pytest.raises(DataSupplyError, match="全部数据源失败"):
            supply._default_fetch_hist("600519", "2023-01-01", "2026-07-24")

    def test_with_timeout_on_hang(self):
        import time

        from app.pipeline1.data_supply import _with_timeout

        with pytest.raises(TimeoutError):
            _with_timeout(lambda: time.sleep(5), timeout=0.2)

    def test_with_timeout_passthrough_exception(self):
        from app.pipeline1.data_supply import _with_timeout

        with pytest.raises(ValueError, match="inner"):
            _with_timeout(lambda: (_ for _ in ()).throw(ValueError("inner")), timeout=1)


# ---------------- 窗口深度自适应 (B11) ----------------
class TestAdaptiveWindow:
    def test_shallow_data_uses_transition_window(self, tmp_path):
        """3 年数据经步骤1过滤后 ~490 日 < 750 → 过渡窗口, es/calib 不为空."""
        dtt.LGB_PARAMS_REG["n_estimators"] = 10
        dtt.LGB_PARAMS_CLS["n_estimators"] = 10
        dtt.ES_PATIENCE = 3
        panel = make_panel(days=500)  # 500 日面板 (< 750 窗口)
        df = panel.copy()
        df["label_1d"] = df.groupby("symbol")["close_hfq"].shift(-1) / df["close_hfq"] - 1
        df["label_3d"] = df.groupby("symbol")["close_hfq"].shift(-3) / df["close_hfq"] - 1
        df["label_5d"] = df.groupby("symbol")["close_hfq"].shift(-5) / df["close_hfq"] - 1
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1", "f2"])
        assert len(trained["segs"]["es"]) > 0
        assert len(trained["segs"]["calib"]) > 0
        assert len(trained["segs"]["test"]) > 0

    def test_too_shallow_raises(self, tmp_path):
        panel = make_panel(days=160)  # < 130+50 下限
        df = panel.copy()
        df["label_1d"] = 0.01
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="深度不足"):
            trainer.train_window(df, "main", ["f1", "f2"])


# ---------------- enrich_panel ----------------
class TestEnrichPanel:
    def _mini(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["600519", "600519", "300750", "300750", "688981"],
                "date": pd.to_datetime(
                    ["2026-07-20", "2026-07-21"] * 2 + ["2026-07-20"]
                ),
                "turnover_rate": [3.0, 3.1, 5.0, 5.2, 4.0],
            }
        )

    def test_defaults(self):
        out = enrich_panel(self._mini())
        board = dict(zip(out["symbol"], out["board"]))
        assert board == {"600519": "main", "300750": "GEM", "688981": "STAR"}
        assert out["is_st"].eq(False).all()
        assert out["is_suspended"].eq(False).all()
        days = out.sort_values(["symbol", "date"]).groupby("symbol")["list_days"]
        assert all(list(g) == [1, 2] for _, g in days if len(g) == 2)
        assert out["industry"].eq("UNKNOWN").all()
        assert (out["free_float_turnover_rate"] == out["turnover_rate"]).all()

    def test_maps(self):
        out = enrich_panel(
            self._mini(),
            industry_map={"600519": "白酒"},
            name_map={"300750": "ST宁德"},
        )
        assert out.loc[out["symbol"] == "600519", "industry"].iloc[0] == "白酒"
        assert out.loc[out["symbol"] == "300750", "is_st"].iloc[0]
        assert not out.loc[out["symbol"] == "600519", "is_st"].iloc[0]


# ---------------- assemble_panel ----------------
class TestAssemblePanel:
    def test_backfill_enrich_cache(self, tmp_path):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=30)

        def fake_hist(symbol, start, end):
            close = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
            return pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "open_hfq": close,
                    "high_hfq": close * 1.01,
                    "low_hfq": close * 0.99,
                    "close_hfq": close,
                    "volume": 1e6,
                    "amount": 1e8,
                    "turnover_rate": 2.0,
                    "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                }
            )

        supply = DataSupplyChain(
            cache_dir=str(tmp_path / "supply"), fetcher_hist=fake_hist
        )
        panel = assemble_panel(
            supply,
            ["600519", "300750"],
            end="2026-07-24",
            years=3,
            cache_dir=str(tmp_path / "panels"),
        )
        assert set(panel["symbol"]) == {"600519", "300750"}
        assert {"board", "is_st", "list_days", "industry"} <= set(panel.columns)
        # 缓存落盘 (WORM 日期后缀) + 二次调用读缓存
        cached = list((tmp_path / "panels").glob("panel_20260724_3y.parquet"))
        assert len(cached) == 1
        panel2 = assemble_panel(
            supply,
            ["600519", "300750"],
            end="2026-07-24",
            years=3,
            cache_dir=str(tmp_path / "panels"),
        )
        assert len(panel2) == len(panel)


# ---------------- train → predict 全链路 ----------------
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """合成面板 → run_training (tiny LGB) → 模型包目录."""
    tmp = tmp_path_factory.mktemp("train_predict")
    dtt.LGB_PARAMS_REG["n_estimators"] = 10
    dtt.LGB_PARAMS_CLS["n_estimators"] = 10
    dtt.ES_PATIENCE = 3
    panel = make_panel()  # 760 日双股主板面板
    results = run_training(
        panel,
        "w30",
        model_dir=str(tmp / "models"),
        registry_path=str(tmp / "registry"),
        use_ic_screen=False,
    )
    return {"tmp": tmp, "panel": panel, "results": results}


class TestRunTraining:
    def test_bundles_saved_with_oos(self, trained):
        res = trained["results"]
        assert "main" in res  # make_panel 全主板 → dual 无样本跳过
        assert (trained["tmp"] / "models" / "main_w30.pkl").exists()
        assert "ics" in res["main"]["oos"]
        assert res["main"]["n_features"] > 0

    def test_prepare_board_frame_labels(self, trained):
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        df = prepare_board_frame(trained["panel"], FeatureEngineV35())
        for col in ("label_1d", "label_1d_net", "label_pm_1d", "label_pain"):
            assert col in df.columns
        # 近端 6 日标签已掩码 (未成熟)
        tail = df.groupby("symbol").tail(6)
        assert tail["label_1d"].isna().all()


class TestFindBundles:
    def test_latest_and_tag(self, trained):
        import os

        mdir = str(trained["tmp"] / "models")
        expected = {"main": os.path.join(mdir, "main_w30.pkl")}
        assert find_bundles(mdir, tag="w30") == expected
        assert find_bundles(mdir) == expected
        assert find_bundles(mdir, tag="nonexistent") == {}

    def test_missing_dir(self, tmp_path):
        assert find_bundles(str(tmp_path / "nope")) == {}


class TestRunPrediction:
    def test_emits_schema_list(self, trained):
        from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline

        mdir = str(trained["tmp"] / "models")
        # 测试仅 2 只股: 放宽流动性安全阀 (生产默认 50/15 针对全市场)
        cleaner = CleaningPipeline(CleaningConfig(valve_full=2, valve_reduced=1))
        result = run_prediction(
            trained["panel"],
            "20260720",
            find_bundles(mdir),
            list_dir=str(trained["tmp"] / "lists"),
            cleaner=cleaner,
        )
        assert result["mode"] in ("normal", "empty")
        if result["mode"] == "normal":
            assert list(result["list"].columns) == SCHEMA_FIELDS

    def test_no_bundles_raises(self, trained):
        with pytest.raises(RuntimeError, match="模型包"):
            run_prediction(trained["panel"], "20260720", {})
