"""Backtest module coverage tests: SignalEvaluator, DataValidator,
ComparativeAnalyzer, DataLoader."""

import pandas as pd
import pytest

from app.backtest.config_manager import BacktestConfig

# ── Shared fixtures ──


@pytest.fixture
def cfg():
    return BacktestConfig(
        initial_capital=100000.0,
        trigger_pct=0.03,
        prob_threshold=0.50,
        holding_period=2,
        signal_horizons=[1, 2],
        signal_simulate_trigger=True,
        volume_confirm_ratio=1.5,
    )


@pytest.fixture
def pred_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stocks = ["000001", "000002", "000003", "000004", "000005"]
    rows = []
    for d in dates:
        for i, s in enumerate(stocks):
            rows.append(
                {
                    "date": d,
                    "stock": s,
                    "score_h1": 0.6 + i * 0.02,
                    "prob_up_h1": 0.55 + i * 0.01,
                    "pred_ret_h1": 0.01 + i * 0.005,
                    "score_h2": 0.6 + i * 0.02,
                    "prob_up_h2": 0.55 + i * 0.01,
                    "pred_ret_h2": 0.015 + i * 0.005,
                    "score_h4": 0.6 + i * 0.02,
                    "prob_up_h4": 0.55 + i * 0.01,
                    "pred_ret_h4": 0.02 + i * 0.005,
                    "board": "main",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def price_df():
    dates = pd.date_range("2024-01-01", periods=7, freq="B")
    stocks = ["000001", "000002", "000003", "000004", "000005"]
    rows = []
    for d in dates:
        for i, s in enumerate(stocks):
            base = 10.0 + i
            rows.append(
                {
                    "date": d,
                    "stock": s,
                    "open": base,
                    "high": base * 1.05,
                    "low": base * 0.97,
                    "close": base * 1.02,
                    "volume": 2000000 + i * 100000,
                    "amount": 20000000.0 + i * 1000000,
                    "up_limit": base * 1.1,
                    "down_limit": base * 0.9,
                    "is_st": False,
                    "is_halt": False,
                    "pre_close": base * 0.99,
                    "circ_mv": 2000000000.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trade_dates(price_df):
    return sorted(price_df["date"].unique())


# ════════════════════════════════════════════════════
# SignalEvaluator
# ════════════════════════════════════════════════════


class TestSignalEvaluator:
    """SignalEvaluator 信号质量评估."""

    def test_calc_rank_ic(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        ic_df = ev.calc_rank_ic("score_h2", horizon=2, simulate_trigger=False)
        assert isinstance(ic_df, pd.DataFrame)
        assert "rank_ic" in ic_df.columns or ic_df.empty

    def test_calc_rank_ic_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        empty_pred = pd.DataFrame(columns=["date", "stock", "score_h2"])
        empty_price = pd.DataFrame(columns=["date", "stock", "open", "high", "close"])
        ev = SignalEvaluator(empty_pred, empty_price, [], cfg)
        ic_df = ev.calc_rank_ic()
        assert ic_df.empty

    def test_calc_topk_hit_rate(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        result = ev.calc_topk_hit_rate("score_h2", "prob_up_h2", horizon=2)
        assert "top1_hit_rate" in result
        assert "top5_hit_rate" in result
        assert isinstance(result["top1_hit_rate"], float)

    def test_calc_topk_hit_rate_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(
            pd.DataFrame(columns=["date", "stock"]),
            pd.DataFrame(columns=["date", "stock"]),
            [],
            cfg,
        )
        result = ev.calc_topk_hit_rate()
        assert result["top1_hit_rate"] == 0.0

    def test_calc_trigger_stats(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        stats = ev.calc_trigger_stats("score_h2", "prob_up_h2", horizon=2)
        assert "trigger_rate" in stats
        assert "open_trigger_rate" in stats
        assert isinstance(stats["trigger_rate"], float)

    def test_calc_trigger_stats_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(
            pd.DataFrame(columns=["date", "stock"]),
            pd.DataFrame(columns=["date", "stock"]),
            [],
            cfg,
        )
        stats = ev.calc_trigger_stats()
        assert stats["trigger_rate"] == 0.0

    def test_calc_prediction_bias(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        bias = ev.calc_prediction_bias("pred_ret_h2", horizon=2)
        assert "mae" in bias
        assert "rmse" in bias
        assert "wmape" in bias
        assert isinstance(bias["mae"], float)

    def test_calc_prediction_bias_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(
            pd.DataFrame(columns=["date", "stock"]),
            pd.DataFrame(columns=["date", "stock"]),
            [],
            cfg,
        )
        bias = ev.calc_prediction_bias()
        assert bias["mae"] == 0.0

    def test_calc_prob_calibration(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        cal = ev.calc_prob_calibration("prob_up_h2", horizon=2)
        assert isinstance(cal, pd.DataFrame)
        assert "prob_bin" in cal.columns
        assert len(cal) > 0

    def test_calc_prob_calibration_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(
            pd.DataFrame(columns=["date", "stock"]),
            pd.DataFrame(columns=["date", "stock"]),
            [],
            cfg,
        )
        cal = ev.calc_prob_calibration()
        assert cal.empty

    def test_calc_gap_risk(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        gap = ev.calc_gap_risk("score_h2")
        assert isinstance(gap, float)

    def test_calc_gap_risk_empty(self, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(
            pd.DataFrame(columns=["date", "stock"]),
            pd.DataFrame(columns=["date", "stock"]),
            [],
            cfg,
        )
        assert ev.calc_gap_risk() == 0.0

    def test_calc_volume_confirmation_rate(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        rate = ev.calc_volume_confirmation_rate("score_h2")
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_run_full_report(self, pred_df, price_df, trade_dates, cfg):
        from app.backtest.signal_evaluator import SignalEvaluator

        ev = SignalEvaluator(pred_df, price_df, trade_dates, cfg)
        report = ev.run_full_report()
        assert isinstance(report, dict)
        assert "h1" in report or "h2" in report


# ════════════════════════════════════════════════════
# DataValidator
# ════════════════════════════════════════════════════


class TestDataValidator:
    """DataValidator 数据完整性校验."""

    def test_validate_prices_clean(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        errors = dv.validate_prices()
        assert isinstance(errors, list)

    def test_validate_prices_bad_close(self, pred_df):
        from app.backtest.data_validator import DataValidator

        bad_price = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "stock": ["000001"] * 3,
                "open": [10, 10, 10],
                "high": [11, 11, 11],
                "low": [9, 9, 9],
                "close": [10, -1, 10],
                "volume": [1000, 1000, 1000],
            }
        )
        dv = DataValidator(bad_price, pred_df)
        errors = dv.validate_prices()
        assert any(e[0] == "E003" for e in errors)

    def test_validate_prices_high_low(self, pred_df):
        from app.backtest.data_validator import DataValidator

        bad_price = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "stock": ["000001"] * 3,
                "open": [10, 10, 10],
                "high": [9, 11, 11],
                "low": [11, 9, 9],
                "close": [10, 10, 10],
                "volume": [1000, 1000, 1000],
            }
        )
        dv = DataValidator(bad_price, pred_df)
        errors = dv.validate_prices()
        assert any(e[0] == "E004" for e in errors)

    def test_validate_prices_halt_inferred(self, pred_df):
        from app.backtest.data_validator import DataValidator

        bad_price = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "stock": ["000001"] * 3,
                "open": [10, 10, 10],
                "high": [11, 11, 11],
                "low": [9, 9, 9],
                "close": [10, 10, 10],
                "volume": [1000, 0, 1000],
            }
        )
        dv = DataValidator(bad_price, pred_df)
        errors = dv.validate_prices()
        assert any(e[0] == "E005" for e in errors)

    def test_validate_predictions_clean(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        errors = dv.validate_predictions()
        assert isinstance(errors, list)

    def test_validate_predictions_out_of_range(self, price_df):
        from app.backtest.data_validator import DataValidator

        bad_pred = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "stock": ["000001"] * 3,
                "prob_up_h1": [0.5, 1.5, 0.6],
                "prob_up_h2": [0.5, 0.6, 0.7],
                "prob_up_h4": [0.5, 0.6, 0.7],
            }
        )
        dv = DataValidator(price_df, bad_pred)
        errors = dv.validate_predictions()
        assert any(e[0] == "E002" for e in errors)

    def test_check_adjusted_prices_ok(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        assert dv.check_adjusted_prices() is False

    def test_check_adjusted_prices_unadjusted(self, pred_df):
        from app.backtest.data_validator import DataValidator

        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        unadjusted = pd.DataFrame(
            {
                "date": dates,
                "stock": ["000001"] * 5,
                "close": [10, 7, 11, 10, 10],
            }
        )
        dv = DataValidator(unadjusted, pred_df)
        assert dv.check_adjusted_prices() is True

    def test_check_pit_data(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        warnings = dv.check_pit_data()
        assert isinstance(warnings, list)

    def test_check_pit_data_missing_board(self, price_df):
        from app.backtest.data_validator import DataValidator

        bad_pred = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=3), "stock": ["000001"] * 3}
        )
        dv = DataValidator(price_df, bad_pred)
        warnings = dv.check_pit_data()
        assert any("board" in w for w in warnings)

    def test_infer_halt_status(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        result = dv.infer_halt_status()
        assert isinstance(result, pd.DataFrame)
        assert "is_halt" in result.columns

    def test_infer_halt_status_vol_zero(self, pred_df):
        from app.backtest.data_validator import DataValidator

        price = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "stock": ["000001"] * 3,
                "open": [10, 10, 10],
                "high": [11, 11, 11],
                "low": [9, 9, 9],
                "close": [10, 10, 10],
                "volume": [1000, 0, 500],
            }
        )
        dv = DataValidator(price, pred_df)
        result = dv.infer_halt_status()
        assert result.loc[result["volume"] == 0, "is_halt"].iloc[0] == 1

    def test_filter_ipo_stocks(self, pred_df):
        from app.backtest.data_validator import DataValidator

        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        price = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "stock": ["000001"] * 10 + ["000002"] * 10,
                "close": range(20),
            }
        )
        dv = DataValidator(price, pred_df)
        filtered = dv.filter_ipo_stocks(min_listing_days=5)
        assert len(filtered) < len(price)

    def test_run_all_checks(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        dv = DataValidator(price_df, pred_df)
        issues = dv.run_all_checks()
        assert isinstance(issues, list)
        assert any(i[0] == "E008" for i in issues)
        assert any(i[0] == "E010" for i in issues)

    def test_run_all_checks_with_market(self, price_df, pred_df):
        from app.backtest.data_validator import DataValidator

        market = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "index_close": [3000, 3010, 3005],
            }
        )
        dv = DataValidator(price_df, pred_df, market_df=market)
        issues = dv.run_all_checks()
        assert not any(i[0] == "E010" for i in issues)


# ════════════════════════════════════════════════════
# ComparativeAnalyzer
# ════════════════════════════════════════════════════


class TestComparativeAnalyzer:
    """ComparativeAnalyzer Squad vs Sniper 对比."""

    def _make_ca(self, benchmark=None):
        from app.backtest.comparative_analyzer import ComparativeAnalyzer

        squad_result = pd.DataFrame({"nav": [100000, 101000, 102000, 100500]})
        sniper_result = pd.DataFrame({"nav": [100000, 102000, 103000, 101000]})
        squad_trades = pd.DataFrame({"pnl": [500, -200, 300]})
        sniper_trades = pd.DataFrame({"pnl": [1000, -500, 800]})
        squad_metrics = {
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.02,
            "total_return": 0.03,
        }
        sniper_metrics = {
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.04,
            "total_return": 0.05,
        }
        return ComparativeAnalyzer(
            squad_result,
            sniper_result,
            squad_trades,
            sniper_trades,
            squad_metrics,
            sniper_metrics,
            benchmark,
        )

    def test_calc_concentration_risk_ratio(self):
        ca = self._make_ca()
        ratio = ca.calc_concentration_risk_ratio()
        assert isinstance(ratio, float)
        assert ratio > 0

    def test_calc_concentration_risk_ratio_zero_dd(self):
        from app.backtest.comparative_analyzer import ComparativeAnalyzer

        ca = ComparativeAnalyzer(
            pd.DataFrame({"nav": [100000]}),
            pd.DataFrame({"nav": [100000]}),
            pd.DataFrame(),
            pd.DataFrame(),
            {"max_drawdown": 0.0},
            {"max_drawdown": 0.0},
        )
        assert ca.calc_concentration_risk_ratio() == 0.0

    def test_calc_jensen_alpha_no_benchmark(self):
        ca = self._make_ca()
        alpha = ca.calc_jensen_alpha(pd.DataFrame({"nav": [100000, 101000]}))
        assert alpha == 0.0

    def test_calc_jensen_alpha_with_benchmark(self):
        bench = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=4),
                "csi1000_close": [3000, 3010, 3005, 3020],
            }
        )
        ca = self._make_ca(benchmark=bench)
        alpha = ca.calc_jensen_alpha(
            pd.DataFrame({"nav": [100000, 101000, 102000, 103000]})
        )
        assert isinstance(alpha, float)

    def test_compare_nav_curves(self):
        ca = self._make_ca()
        desc = ca.compare_nav_curves()
        assert "squad_final_nav" in desc
        assert "sniper_final_nav" in desc

    def test_compare_nav_curves_empty(self):
        from app.backtest.comparative_analyzer import ComparativeAnalyzer

        ca = ComparativeAnalyzer(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {},
            {},
        )
        desc = ca.compare_nav_curves()
        assert "squad_final_nav=0.00" in desc

    def test_generate_comparison_report(self):
        ca = self._make_ca()
        report = ca.generate_comparison_report()
        assert "concentration_risk_ratio" in report
        assert "squad_sharpe" in report
        assert "sniper_sharpe" in report
        assert "recommendation" in report
        assert report["recommendation"] in (
            "concentrated",
            "diversified",
            "tighten_stop",
        )


# ════════════════════════════════════════════════════
# DataLoader
# ════════════════════════════════════════════════════


class TestDataLoader:
    """DataLoader 数据加载与对齐."""

    def _make_pred_csv(self, path):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        stocks = ["000001", "000002"]
        rows = []
        for d in dates:
            for s in stocks:
                rows.append(
                    {
                        "date": d,
                        "stock": s,
                        "score_h1": 0.6,
                        "prob_up_h1": 0.55,
                        "pred_ret_h1": 0.01,
                        "score_h2": 0.6,
                        "prob_up_h2": 0.55,
                        "pred_ret_h2": 0.01,
                        "score_h4": 0.6,
                        "prob_up_h4": 0.55,
                        "pred_ret_h4": 0.01,
                        "board": "main",
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    def _make_price_csv(self, path):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        stocks = ["000001", "000002"]
        rows = []
        for d in dates:
            for s in stocks:
                rows.append(
                    {
                        "date": d,
                        "stock": s,
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.8,
                        "close": 10.2,
                        "volume": 2000000,
                        "amount": 20000000.0,
                        "up_limit": 11.0,
                        "down_limit": 9.0,
                        "is_st": False,
                        "is_halt": False,
                        "pre_close": 9.9,
                        "circ_mv": 2000000000.0,
                    }
                )
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_load_csv(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        loader = DataLoader(pred_path, price_path)
        pred_df, price_df, bench_df, market_df = loader.load()
        assert len(pred_df) > 0
        assert len(price_df) > 0
        assert "avg_amount_20d" in price_df.columns

    def test_read_file_not_found(self):
        from app.backtest.data_loader import DataLoader

        with pytest.raises(FileNotFoundError):
            DataLoader._read_file("nonexistent.csv", "test")

    def test_load_parquet(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.parquet")
        price_path = str(tmp_path / "price.parquet")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        # Convert to parquet
        pd.read_csv(pred_path).to_parquet(pred_path)
        pd.read_csv(price_path).to_parquet(price_path)
        loader = DataLoader(pred_path, price_path)
        pred_df, price_df, _, _ = loader.load()
        assert len(pred_df) > 0

    def test_validate_missing_cols(self, pred_df):
        from app.backtest.data_loader import DataLoader

        bad_df = pd.DataFrame({"date": [1], "stock": ["x"]})
        with pytest.raises(ValueError, match="缺失必要列"):
            loader = DataLoader("a", "b")
            loader._validate(bad_df, ["date", "stock", "missing_col"], "test")

    def test_get_trade_dates(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        loader = DataLoader(pred_path, price_path)
        loader.load()
        dates = loader.get_trade_dates()
        assert len(dates) > 0
        assert all(isinstance(d, pd.Timestamp) for d in dates)

    def test_get_next_trade_date(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        loader = DataLoader(pred_path, price_path)
        loader.load()
        dates = loader.get_trade_dates()
        next_d = loader.get_next_trade_date(dates[0], 1)
        assert next_d == dates[1]
        beyond = loader.get_next_trade_date(dates[0], 999)
        assert beyond is None

    def test_get_data_version_hash(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        loader = DataLoader(pred_path, price_path)
        loader.load()
        h = loader.get_data_version_hash()
        assert h.startswith("sha256:")

    def test_load_with_market(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        market_path = str(tmp_path / "market.csv")
        self._make_pred_csv(pred_path)
        self._make_price_csv(price_path)
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=5),
                "index_close": range(3000, 3005),
            }
        ).to_csv(market_path, index=False)
        loader = DataLoader(pred_path, price_path, market_path=market_path)
        pred_df, price_df, _, market_df = loader.load()
        assert len(market_df) > 0

    def test_load_missing_pred_cols(self, tmp_path):
        from app.backtest.data_loader import DataLoader

        pred_path = str(tmp_path / "pred.csv")
        price_path = str(tmp_path / "price.csv")
        # pred with missing columns
        pd.DataFrame({"date": [1], "stock": ["x"]}).to_csv(pred_path, index=False)
        self._make_price_csv(price_path)
        loader = DataLoader(pred_path, price_path)
        with pytest.raises(ValueError, match="缺失必要列"):
            loader.load()
