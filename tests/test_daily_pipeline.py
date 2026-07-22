# -*- coding: utf-8 -*-
"""Pipeline-1 每日选股编排 e2e 测试 (合成数据 + 真实小模型)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.pipeline1.dual_track_trainer as dtt
from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.data_supply import DataSupplyChain, DataSupplyError
from app.pipeline1.list_generator import SCHEMA_FIELDS


def make_panel(symbols=("600519", "601318"), days=760, seed=21) -> pd.DataFrame:
    """760 交易日 (>720 窗口) 双股面板, 含 f1/f2 伪特征."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    frames = []
    for sym in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "board": "main",
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "close_hfq": close,
                    "open_hfq": close,
                    "high_hfq": close * 1.01,
                    "low_hfq": close * 0.99,
                    "volume": 1e7,
                    "amount": 1e9,
                    "turnover_rate": 5.0,
                    "free_float_turnover_rate": 5.0,
                    "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                    "is_suspended": False,
                    "is_st": False,
                    "industry": "白酒" if sym == "600519" else "保险",
                    "list_days": 1000,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    rng2 = np.random.default_rng(5)
    df["f1"] = rng2.normal(size=len(df))
    df["f2"] = rng2.normal(size=len(df))
    return df


class _StubFeatures:
    """跳过真实特征工程, 直接透传 (f1/f2 已在面板上)."""

    def build(self, df, float_shares_map=None):
        return df


def _train_bundle(tmp_path, panel):
    """用 720 窗口真实训练 tiny LightGBM 并保存模型包 (含校准器)."""
    dtt.LGB_PARAMS_REG["n_estimators"] = 10
    dtt.LGB_PARAMS_CLS["n_estimators"] = 10
    dtt.ES_PATIENCE = 3
    df = panel.copy()
    df["label_1d"] = df.groupby("symbol")["close_hfq"].shift(-1) / df["close_hfq"] - 1
    df["label_3d"] = df.groupby("symbol")["close_hfq"].shift(-3) / df["close_hfq"] - 1
    df["label_5d"] = df.groupby("symbol")["close_hfq"].shift(-5) / df["close_hfq"] - 1
    df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
    trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
    trained = trainer.train_window(df, "main", ["f1", "f2"])
    return trainer.save(trained, "test")


@pytest.fixture()
def pipeline(tmp_path):
    panel = make_panel()
    bundle = _train_bundle(tmp_path, panel)
    pipe = DailySelectionPipeline(
        supply=DataSupplyChain(cache_dir=str(tmp_path / "cache")),
        bundle_paths={"main": bundle},
        list_dir=str(tmp_path / "lists"),
    )
    pipe.features = _StubFeatures()
    # 测试仅 2 只股: 放宽流动性安全阀阈值 (生产默认 50/15 针对全市场)
    pipe.cleaner = CleaningPipeline(CleaningConfig(valve_full=2, valve_reduced=1))
    return pipe, panel


class TestDailyPipeline:
    def test_run_emits_schema_list(self, pipeline):
        pipe, panel = pipeline
        result = pipe.run("20260720", panel=panel)
        assert result["mode"] == "normal"
        lst = result["list"]
        assert list(lst.columns) == SCHEMA_FIELDS
        assert 0 < len(lst) <= 2
        assert (lst["schema_version"] == "1.0").all()

    def test_list_persisted_and_yesterday_carryover(self, pipeline):
        """清单持久化 + 次日 is_in_yesterday_list 回填 (Holding Bonus)."""
        pipe, panel = pipeline
        r1 = pipe.run("20260720", panel=panel)
        assert pipe.load_list("20260720") is not None
        yesterday_symbols = set(r1["list"]["symbol"])
        r2 = pipe.run("20260721", panel=panel)
        assert r2["mode"] == "normal"
        carried = pipe._load_yesterday("20260721")
        assert set(carried["symbol"]) == yesterday_symbols

    def test_supply_failure_triggers_guard(self, tmp_path):
        """数据供应链失败 → 三档降级 (第1档: 沿用昨日/告警)."""

        class FailSupply(DataSupplyChain):
            def _assemble_check(self):
                raise DataSupplyError("network down")

        pipe = DailySelectionPipeline(
            supply=FailSupply(cache_dir=str(tmp_path / "c")),
            bundle_paths={},
            list_dir=str(tmp_path / "l"),
        )
        result = pipe.run("20260720", panel=None)  # _assemble_panel 抛 DataSupplyError
        assert result["mode"] == "reuse_yesterday"
        # 连续失败升级
        assert pipe.guard.on_failure()["mode"] == "sell_only"
        assert pipe.guard.on_failure()["mode"] == "manual_intervention"

    def test_empty_trigger_forces_empty_list(self, pipeline):
        from app.pipeline1.list_generator import MarketEnv

        pipe, panel = pipeline
        r = pipe.run("20260720", panel=panel, env=MarketEnv(hs300_drop_today=0.031))
        assert r["empty"] and len(r["list"]) == 0

    def test_is_retrain_day(self):
        cal = ["20260529", "20260601", "20260602", "20260701"]
        assert DailySelectionPipeline.is_retrain_day("20260601", cal)
        assert not DailySelectionPipeline.is_retrain_day("20260602", cal)
        assert DailySelectionPipeline.is_retrain_day("20260701", cal)
