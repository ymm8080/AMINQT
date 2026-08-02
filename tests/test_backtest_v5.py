# -*- coding: utf-8 -*-
"""V5.0/V5.2 回测模块单元测试."""

import os
import json

import pandas as pd
import pytest

from app.backtest.config_manager import BacktestConfig, ConfigManager


@pytest.fixture
def sample_config():
    return BacktestConfig(
        initial_capital=100000.0,
        trigger_pct=0.03,
        position_mode="squad",
        prob_threshold=0.55,
        holding_period=2,
    )


@pytest.fixture
def sample_pred_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stocks = ["000001", "000002", "000003"]
    rows = []
    for d in dates:
        for s in stocks:
            rows.append({
                "date": d, "stock": s,
                "score_h1": 0.7, "prob_up_h1": 0.6, "pred_ret_h1": 0.02,
                "score_h2": 0.7, "prob_up_h2": 0.6, "pred_ret_h2": 0.02,
                "score_h4": 0.7, "prob_up_h4": 0.6, "pred_ret_h4": 0.02,
                "board": "main",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    stocks = ["000001", "000002", "000003"]
    rows = []
    for d in dates:
        for s in stocks:
            close = 10.0
            rows.append({
                "date": d, "stock": s,
                "open": 10.0, "high": 10.5, "low": 9.8, "close": close,
                "volume": 2000000, "amount": 20000000.0,
                "up_limit": 11.0, "down_limit": 9.0,
                "is_st": False, "is_halt": False,
                "pre_close": 9.9, "circ_mv": 2000000000.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_market_df():
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame({"date": dates, "index_close": [3000, 3010, 3005, 2990, 3015, 3020]})


class TestConfigManager:
    def test_default_config(self):
        config = BacktestConfig()
        assert config.initial_capital > 0
        assert config.trigger_pct > 0

    def test_config_hash_deterministic(self, sample_config):
        h1 = ConfigManager.hash(sample_config)
        h2 = ConfigManager.hash(sample_config)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_config_hash_differs(self):
        c1 = BacktestConfig(initial_capital=100000)
        c2 = BacktestConfig(initial_capital=200000)
        assert ConfigManager.hash(c1) != ConfigManager.hash(c2)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = "backtest:\n  initial_capital: 500000.0\n  trigger_pct: 0.05\n  position_mode: sniper\n"
        p = tmp_path / "test_config.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        config = ConfigManager.load(str(p))
        assert config.initial_capital == 500000.0
        assert config.trigger_pct == 0.05

    def test_load_missing_file(self):
        config = ConfigManager.load("nonexistent.yaml")
        assert config.initial_capital == 100000.0

    def test_v50_params_exist(self):
        config = BacktestConfig()
        assert hasattr(config, "volume_confirm_ratio")
        assert hasattr(config, "market_drop_limit")
        assert hasattr(config, "down_limit_max_days")

    def test_v52_atr_params_exist(self):
        config = BacktestConfig()
        assert hasattr(config, "stop_loss_atr_mult")
        assert hasattr(config, "atr_period")


class TestReportGenerator:
    def test_generate_text_report(self, sample_config, tmp_path):
        from app.backtest.report_generator import ReportGenerator
        reporter = ReportGenerator(config=sample_config, output_dir=str(tmp_path))
        result_df = pd.DataFrame({"nav": [100000, 101000], "daily_pnl_pct": [0, 0.01]})
        trades_df = pd.DataFrame({
            "entry_date": [pd.Timestamp("2024-01-01")],
            "exit_date": [pd.Timestamp("2024-01-02")],
            "stock": ["000001"], "entry_price": [10.0],
            "exit_price": [10.5], "quantity": [1000],
            "pnl": [500], "pnl_pct": [0.05],
            "exit_reason": ["expired"],
        })
        metrics = {"total_return": 0.02, "sharpe_ratio": 1.5, "max_drawdown": -0.01, "win_rate": 0.5, "num_trades": 1}
        basepath = reporter.generate("squad", result_df, trades_df, metrics, data_version_hash="sha256:test")
        assert os.path.exists(basepath + ".txt")
        assert os.path.exists(basepath + ".json")
        assert os.path.exists(basepath + ".html")

    def test_json_serializable(self, sample_config, tmp_path):
        from app.backtest.report_generator import ReportGenerator
        reporter = ReportGenerator(config=sample_config, output_dir=str(tmp_path))
        basepath = reporter.generate("test",
            pd.DataFrame({"nav": [100000], "daily_pnl_pct": [0]}),
            pd.DataFrame(), {"total_return": 0.0})
        with open(basepath + ".json", encoding="utf-8") as f:
            data = json.load(f)
        assert data["meta"]["mode"] == "test"

    def test_empty_trades(self, sample_config, tmp_path):
        from app.backtest.report_generator import ReportGenerator
        reporter = ReportGenerator(config=sample_config, output_dir=str(tmp_path))
        basepath = reporter.generate("empty",
            pd.DataFrame({"nav": [100000], "daily_pnl_pct": [0]}),
            pd.DataFrame(), {"total_return": 0.0, "num_trades": 0})
        assert os.path.exists(basepath + ".txt")


class TestBacktestEngine:
    def test_engine_init(self, sample_config, sample_pred_df, sample_price_df):
        from app.backtest.engine import BacktestEngine
        trade_dates = sorted(sample_price_df["date"].unique())
        engine = BacktestEngine(
            config=sample_config, pred_df=sample_pred_df,
            price_df=sample_price_df, trade_dates=trade_dates,
            data_version_hash="sha256:test")
        assert engine.config.initial_capital == 100000.0

    def test_engine_run(self, sample_config, sample_pred_df, sample_price_df):
        from app.backtest.engine import BacktestEngine
        trade_dates = sorted(sample_price_df["date"].unique())
        engine = BacktestEngine(
            config=sample_config, pred_df=sample_pred_df,
            price_df=sample_price_df, trade_dates=trade_dates,
            data_version_hash="sha256:test")
        result = engine.run(horizon=2)
        assert isinstance(result, pd.DataFrame)
        assert "nav" in result.columns

    def test_engine_metrics(self, sample_config, sample_pred_df, sample_price_df):
        from app.backtest.engine import BacktestEngine
        trade_dates = sorted(sample_price_df["date"].unique())
        engine = BacktestEngine(
            config=sample_config, pred_df=sample_pred_df,
            price_df=sample_price_df, trade_dates=trade_dates,
            data_version_hash="sha256:test")
        engine.run(horizon=2)
        metrics = engine.get_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics

    def test_market_drop_no_data(self, sample_config, sample_pred_df, sample_price_df):
        from app.backtest.engine import BacktestEngine
        trade_dates = sorted(sample_price_df["date"].unique())
        engine = BacktestEngine(
            config=sample_config, pred_df=sample_pred_df,
            price_df=sample_price_df, trade_dates=trade_dates,
            data_version_hash="sha256:test")
        assert engine._check_market_drop(
            pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-01")) is True

    def test_market_drop_with_data(self, sample_config, sample_pred_df, sample_price_df, sample_market_df):
        from app.backtest.engine import BacktestEngine
        trade_dates = sorted(sample_price_df["date"].unique())
        engine = BacktestEngine(
            config=sample_config, pred_df=sample_pred_df,
            price_df=sample_price_df, trade_dates=trade_dates,
            data_version_hash="sha256:test", market_df=sample_market_df)
        result = engine._check_market_drop(
            sample_market_df["date"].iloc[1], sample_market_df["date"].iloc[0])
        assert isinstance(result, bool)

    def test_holding_has_down_limit_days(self):
        from app.backtest.engine import Holding
        h = Holding(
            stock="000001", entry_date=pd.Timestamp("2024-01-01"),
            entry_price_fen=1000, quantity=100, horizon=2,
            mode="squad", prob_up=0.6, pred_ret=0.02, score=0.7)
        assert hasattr(h, "down_limit_days")
        assert h.down_limit_days == 0

    def test_trade_has_is_swap(self):
        from app.backtest.engine import Trade
        t = Trade(
            entry_date=pd.Timestamp("2024-01-01"),
            exit_date=pd.Timestamp("2024-01-02"),
            stock="000001", entry_price_fen=1000, exit_price_fen=1050,
            quantity=100, pnl_fen=5000, pnl_pct=0.05,
            exit_reason="expired", horizon=2, mode="squad")
        assert hasattr(t, "is_swap")
        assert t.is_swap is False


class TestPipeline:
    def test_pipeline_init(self):
        from app.backtest.pipeline import BacktestPipeline
        pipeline = BacktestPipeline(
            config_path="config/backtest_config.yaml",
            pred_path="data/pred.csv", price_path="data/price.csv",
            modes=["squad"], horizon=2)
        assert pipeline.horizon == 2
