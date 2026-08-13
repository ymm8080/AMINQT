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
from app.pipeline1.train_runner import (
    prepare_board_frame,
    run_training,
    select_features,
)
from tests.test_daily_pipeline import make_panel


# ---------------- 数据源级联 + 硬超时 ----------------
class TestFetchCascade:
    def test_fallback_order_and_down_flag(self, tmp_path):
        """akshare 失败 → sina; 连续失败后本次运行内跳过该源."""
        from app.pipeline1.data_supply import DataSupplyChain

        supply = DataSupplyChain(cache_dir=str(tmp_path))
        calls = []
        good = pd.DataFrame(
            {"symbol": ["600519"], "date": [pd.Timestamp("2026-07-24")]}
        )

        def fail(symbol, start, end):
            calls.append("fail")
            raise ConnectionError("boom")

        supply._tushare_fetch_hist = fail
        supply._akshare_fetch_hist = fail
        supply._sina_fetch_hist = lambda s, a, b: (calls.append("sina"), good)[1]
        # 连续失败 _MAX_CONSECUTIVE_FAILS 次后源被标记 down
        for _ in range(supply._MAX_CONSECUTIVE_FAILS):
            df = supply._default_fetch_hist("600519", "2023-01-01", "2026-07-24")
            assert len(df) == 1
        # 源已 down → 直走 sina (不再调 fail)
        calls.clear()
        supply._default_fetch_hist("600519", "2023-01-01", "2026-07-24")
        assert calls == ["sina"]

    def test_all_sources_fail_raises(self, tmp_path):
        from app.pipeline1.data_supply import DataSupplyChain, DataSupplyError

        supply = DataSupplyChain(cache_dir=str(tmp_path))
        # 源链以 tushare 打头: 4 个源全部 mock 失败
        for name in (
            "_tushare_fetch_hist",
            "_akshare_fetch_hist",
            "_sina_fetch_hist",
            "_baostock_fetch_hist",
        ):
            setattr(
                supply,
                name,
                lambda s, a, b: (_ for _ in ()).throw(ConnectionError("x")),
            )
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
        df["label_1d"] = (
            df.groupby("symbol")["close_hfq"].shift(-1) / df["close_hfq"] - 1
        )
        df["label_2d"] = (
            df.groupby("symbol")["close_hfq"].shift(-2) / df["close_hfq"] - 1
        )
        df["label_3d"] = (
            df.groupby("symbol")["close_hfq"].shift(-3) / df["close_hfq"] - 1
        )
        df["label_5d"] = (
            df.groupby("symbol")["close_hfq"].shift(-5) / df["close_hfq"] - 1
        )
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1", "f2"])
        assert len(trained["segs"]["es"]) > 0
        assert len(trained["segs"]["calib"]) > 0
        assert len(trained["segs"]["test"]) > 0

    def test_too_shallow_raises(self, tmp_path):
        panel = make_panel(days=140)  # < 100+50 下限 (es+calib+test=100, train>=50)
        df = panel.copy()
        df["label_1d"] = 0.01
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="深度不足"):
            trainer.train_window(df, "main", ["f1", "f2"])

    def test_sparse_symbol_does_not_shrink_window(self, tmp_path):
        """多股面板中个股只剩 2 天 (新股/被过滤) 不影响窗口深度 (按日期并集)."""
        dtt.LGB_PARAMS_REG["n_estimators"] = 10
        dtt.LGB_PARAMS_CLS["n_estimators"] = 10
        dtt.ES_PATIENCE = 3
        panel = make_panel(days=500)
        sparse = panel[panel["symbol"] == "600519"].tail(2).copy()
        sparse["symbol"] = "000999"  # 只有 2 天数据的"新股"
        df = pd.concat([panel, sparse], ignore_index=True)
        for k in (1, 2, 3, 5):
            df[f"label_{k}d"] = (
                df.groupby("symbol")["close_hfq"].shift(-k) / df["close_hfq"] - 1
            )
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1", "f2"])  # 不应 raise
        assert len(trained["segs"]["es"]) > 0


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
        board = dict(zip(out["symbol"], out["board"], strict=False))
        assert board == {"600519": "main", "300750": "GEM", "688981": "STAR"}
        assert out["is_suspended"].eq(False).all()
        assert out["industry"].eq("UNKNOWN").all()
        assert (out["free_float_turnover_rate"] == out["turnover_rate"]).all()

    def test_maps(self):
        out = enrich_panel(
            self._mini(),
            industry_map={"600519": "白酒"},
            name_map={"300750": "ST宁德"},
        )
        assert out.loc[out["symbol"] == "600519", "industry"].iloc[0] == "白酒"


# ---------------- assemble_panel ----------------
class TestAssemblePanel:
    def test_backfill_enrich_cache(self, tmp_path, monkeypatch):
        # Mock enrich_alt_data to avoid network calls (AKShare/tushare) hanging in CI
        from app.pipeline1 import panel_builder

        monkeypatch.setattr(
            panel_builder, "enrich_alt_data", lambda panel, *a, **kw: panel
        )
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
        assert {"board", "is_suspended", "industry"} <= set(panel.columns)
        # 缓存落盘 (WORM 日期后缀 + universe 哈希) + 二次调用读缓存
        cached = list((tmp_path / "panels").glob("panel_20260724_3y_*.parquet"))
        assert len(cached) == 1
        panel2 = assemble_panel(
            supply,
            ["600519", "300750"],
            end="2026-07-24",
            years=3,
            cache_dir=str(tmp_path / "panels"),
        )
        assert len(panel2) == len(panel)
        # 不同 universe 不得共享缓存 (缓存键含 universe 哈希)
        assemble_panel(
            supply,
            ["600519"],
            end="2026-07-24",
            years=3,
            cache_dir=str(tmp_path / "panels"),
        )
        assert len(list((tmp_path / "panels").glob("panel_20260724_3y_*.parquet"))) == 2


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


class TestSelectFeaturesBruteInjection:
    """select_features 的 BruteForce 后注入: 单次 join (而非逐族 N 次) 不改变列值与对齐."""

    def test_single_join_injects_missing_brute_col(self, monkeypatch):
        from types import SimpleNamespace

        import app.pipeline1.train_runner as tr
        from app.pipeline1.feature_selector import BruteForceGenerator

        rows = []
        for s in ("600000", "600001"):
            for t in range(25):
                rows.append(
                    {
                        "symbol": s,
                        "date": pd.Timestamp("2026-01-05") + pd.Timedelta(days=t),
                        "f1": float(t),
                    }
                )
        df = pd.DataFrame(rows)

        gen = BruteForceGenerator()
        brute = gen.generate_family(df, "pct_change", raw_cols=["f1"], dtype="float32")
        assert len(brute.columns) > 0, "pct_change 族至少应生成 1 列"
        brute_col = brute.columns[0]
        selected = ["f1", brute_col]

        fake_sel = SimpleNamespace(
            config={"main": {"pipeline": "bruteforce_dedup"}},
            select=lambda df, board: selected,
        )
        monkeypatch.setattr(
            tr,
            "apply_event_scope_screens",
            lambda selected, df, screens=None: selected,
        )

        cols, aug = select_features(df, "main", "tag", fake_sel, registry=None)
        assert brute_col in cols
        assert brute_col in aug.columns
        # 单次 join 值/索引对齐与逐族 join 完全一致 (index 即原 df 索引)
        pd.testing.assert_series_equal(
            aug[brute_col], brute[brute_col], check_names=False
        )


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
            assert list(result["list"].columns)[: len(SCHEMA_FIELDS)] == SCHEMA_FIELDS

    def test_no_bundles_raises(self, trained):
        with pytest.raises(RuntimeError, match="模型包"):
            run_prediction(trained["panel"], "20260720", {})


# ---------------- 跨视界权重 (1d/2d/3d/5d/10d) + 加权 OOS ----------------
class TestHorizonWeights:
    def test_horizons_and_weights_cover_1_2_3_5_10(self):
        from app.pipeline1.label_engine import LABEL_HORIZONS, LABEL_WEIGHTS

        # 2026-08-08 用户裁决: 弃 2d 权重 (0), 3d 最小化 vs 5d, 10d 最高 (资金窗口)
        assert LABEL_HORIZONS == (1, 2, 3, 5, 10)
        assert set(LABEL_WEIGHTS) == {1, 2, 3, 5, 10}
        assert LABEL_WEIGHTS[2] == 0.0  # 2d 权重剔除
        assert LABEL_WEIGHTS[3] < LABEL_WEIGHTS[5]  # 3d 相对 5d 最小化
        assert sum(LABEL_WEIGHTS.values()) == pytest.approx(1.0)

    def test_validate_oos_weighted_ic_formula(self, tmp_path):
        from app.pipeline1.label_engine import LABEL_WEIGHTS

        dtt.LGB_PARAMS_REG["n_estimators"] = 5
        dtt.LGB_PARAMS_CLS["n_estimators"] = 5
        dtt.ES_PATIENCE = 2
        df = make_panel(days=760).copy()
        for k in (1, 2, 3, 5, 10):
            df[f"label_{k}d"] = (
                df.groupby("symbol")["close_hfq"].shift(-k) / df["close_hfq"] - 1
            )
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5, 10):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1", "f2"])
        oos = trainer.validate_oos(trained)
        assert "weighted_ic" in oos
        expected = sum(
            LABEL_WEIGHTS[k] * oos["ics"].get(f"{k}d_reg", 0.0) for k in LABEL_WEIGHTS
        ) / sum(LABEL_WEIGHTS.values())
        assert oos["weighted_ic"] == pytest.approx(expected, abs=1e-6)


# ---------------- run_training 逐板块分期 (内存减半, 模型不变) ----------------
class TestRunTrainingSequentialBoards:
    """回归: run_training 必须逐板块 prepare→select→train→释放,
    而非先攒齐两板块特征帧再训 (内存峰值减半, 最终模型相同).

    顺序约束 (分期关键):
        prepare(main) → select(main) → train(main) → prepare(dual) → select(dual) → train(dual)
    即 main 训完才允许开始 dual 的特征计算, 保证任意时刻至多一份增强帧在内存.
    """

    def test_boards_trained_sequentially_not_staged(self, monkeypatch):
        import app.pipeline1.dual_track_trainer as dtt_mod
        import app.pipeline1.train_runner as tr
        from app.pipeline1 import cleaning_pipeline as cln

        panel = pd.DataFrame(
            {
                "symbol": ["a", "b"],
                "date": ["2026-08-07", "2026-08-07"],
                "board": ["main", "dual"],
            }
        )
        main_df = panel[panel["board"] == "main"].copy()
        dual_df = panel[panel["board"] == "dual"].copy()

        monkeypatch.setattr(
            cln.CleaningPipeline,
            "run_train",
            lambda self, df, board=None: (main_df, dual_df),
        )
        # FeatureEngineV35 仅实例化, 不做真实特征 (prepare 已被 mock)
        monkeypatch.setattr(tr, "FeatureEngineV35", lambda: object())

        events = []

        def fake_prepare(
            board_df,
            features,
            float_shares_map=None,
            cross_sectional_rank=False,
            registry=None,
        ):
            events.append(("prepare", board_df["board"].iloc[0]))
            return board_df.copy()

        def fake_select(df, board, tag, selector=None, registry=None, fallback_boards=None):
            events.append(("select", board))
            return ["f1", "f2"], df

        def fake_weekly(self, panels, feature_cols_by_board, tag, resume=False):
            events.append(("train", next(iter(panels))))
            assert len(panels) == 1, "每板块应单独训练: 不同时持有两板块增强帧"
            board = next(iter(panels))
            return {
                board: {
                    "path": f"models/pipeline1/{board}_20260809.pkl",
                    "oos": {"weighted_ic": 0.05, "pass": True},
                    "switched": True,
                }
            }

        monkeypatch.setattr(tr, "prepare_board_frame", fake_prepare)
        monkeypatch.setattr(tr, "select_features", fake_select)
        monkeypatch.setattr(dtt_mod.DualTrackTrainer, "weekly_retrain", fake_weekly)

        results = run_training(
            panel,
            "20260809",
            model_dir="models/pipeline1",
            use_ic_screen=False,
            use_registry=False,
        )

        expected = [
            ("prepare", "main"),
            ("select", "main"),
            ("train", "main"),
            ("prepare", "dual"),
            ("select", "dual"),
            ("train", "dual"),
        ]
        assert events == expected, f"分期顺序错误: {events}"
        assert set(results) == {"main", "dual"}
        assert results["main"]["n_features"] == 2
        assert results["dual"]["switched"] is True
