# -*- coding: utf-8 -*-
"""BacktestEngine 测试 (P9, ARCH §7)."""

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine import BacktestEngine, BacktestResult

DATES = pd.date_range("2024-01-01", periods=6, freq="B")


def _df(closes, **cols):
    df = pd.DataFrame({"date": DATES, "close": closes})
    for name, values in cols.items():
        df[name] = values
    return df


class TestKnownReturn:
    def test_single_signal_known_return(self):
        # 100 -> 110: 首日信号, 持有 1 日 → 收益 10%
        data = {"AAA": _df([100, 110, 121, 133.1, 146.41, 161.051],
                           signal=[True, False, False, False, False, False])}
        engine = BacktestEngine({"holding_days": 1})
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        assert result.total_return == pytest.approx(0.10, abs=1e-9)
        assert result.max_drawdown == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(result.sharpe)

    def test_no_signal_zero_return(self):
        data = {"AAA": _df([100, 110, 121, 133.1, 146.41, 161.051],
                           signal=[False] * 6)}
        engine = BacktestEngine()
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        assert result.total_return == pytest.approx(0.0)
        assert result.sharpe == pytest.approx(0.0)

    def test_score_threshold_signal(self):
        # 无 signal 列时按 score > threshold 生成信号
        data = {"AAA": _df([100, 110, 121, 133.1, 146.41, 161.051],
                           score=[0.9, 0.1, 0.1, 0.1, 0.1, 0.1])}
        engine = BacktestEngine({"score_threshold": 0.5, "holding_days": 1})
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        assert result.total_return == pytest.approx(0.10, abs=1e-9)

    def test_drawdown_on_falling_price(self):
        data = {"AAA": _df([100, 90, 81, 72.9, 65.61, 59.049],
                           signal=[True, True, False, False, False, False])}
        engine = BacktestEngine({"holding_days": 1})
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        # 两日各 -10%: equity 1.0 -> 0.9 -> 0.81
        assert result.total_return == pytest.approx(-0.19, abs=1e-9)
        # 峰谷回撤: 0.81 / 0.9 - 1 = -0.10
        assert result.max_drawdown == pytest.approx(-0.10, abs=1e-9)

    def test_empty_data_zero_metrics(self):
        engine = BacktestEngine()
        result = engine.run("selection", ["MISSING"], "2024-01-01",
                            "2024-01-10", data={})
        assert result == BacktestResult(gaps={})


class TestIC:
    def _ic_data(self, score_a, score_b):
        # 每日 A 前向收益 > B 前向收益 (单调上涨且 A 涨幅恒大于 B)
        a = _df([100, 102, 104.04, 106.1208, 108.243216, 110.40808032],
                score=[score_a] * 6)
        b = _df([100, 101, 102.01, 103.0301, 104.060701, 105.10140701],
                score=[score_b] * 6)
        return {"AAA": a, "BBB": b}

    def test_ic_positive(self):
        engine = BacktestEngine({"holding_days": 1})
        result = engine.run("selection", ["AAA", "BBB"],
                            "2024-01-01", "2024-01-10",
                            data=self._ic_data(0.9, 0.1))
        # 每日截面 score 排序与前向收益排序完全一致 → spearman = 1
        assert result.ic == pytest.approx(1.0, abs=1e-9)
        assert result.icir > 0

    def test_ic_negative(self):
        engine = BacktestEngine({"holding_days": 1})
        result = engine.run("selection", ["AAA", "BBB"],
                            "2024-01-01", "2024-01-10",
                            data=self._ic_data(0.1, 0.9))
        # score 排序与收益排序完全相反 → spearman = -1
        assert result.ic == pytest.approx(-1.0, abs=1e-9)

    def test_ic_no_future_factor_leak(self):
        # IC 使用 holding_days 日前向收益作标签; 改标签窗口 IC 仍确定
        engine = BacktestEngine({"holding_days": 2})
        result = engine.run("selection", ["AAA", "BBB"],
                            "2024-01-01", "2024-01-10",
                            data=self._ic_data(0.9, 0.1))
        assert result.ic == pytest.approx(1.0, abs=1e-9)


class TestCtrlRatio:
    def test_ctrl_long_short(self):
        # 周线上升股涨 2%/日, 下降股涨 1%/日 → 多空 > 0
        rising = _df([100, 102, 104.04, 106.1208, 108.243216, 110.40808032],
                     score=[0.9] * 6, ctrl_ratio=[0.5] * 6,
                     ctrl_weekly_rising=[True] * 6)
        falling = _df([100, 101, 102.01, 103.0301, 104.060701, 105.10140701],
                      score=[0.1] * 6, ctrl_ratio=[0.2] * 6,
                      ctrl_weekly_rising=[False] * 6)
        engine = BacktestEngine({"holding_days": 1})
        result = engine.run("selection", ["AAA", "BBB"],
                            "2024-01-01", "2024-01-10",
                            data={"AAA": rising, "BBB": falling})
        assert result.ctrl_ratio_weekly_rising_return > 0
        assert result.ctrl_ratio_long_short_return > 0
        assert result.ctrl_ratio_long_short_return == pytest.approx(
            result.ctrl_ratio_weekly_rising_return
            - result.ctrl_ratio_weekly_falling_return)
        # ctrl_ratio 与收益同向 → ctrl IC > 0
        assert result.ctrl_ratio_ic == pytest.approx(1.0, abs=1e-9)

    def test_no_ctrl_columns_defaults(self):
        data = {"AAA": _df([100, 110, 121, 133.1, 146.41, 161.051],
                           signal=[True] + [False] * 5)}
        engine = BacktestEngine()
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        assert result.ctrl_ratio_ic == 0.0
        assert result.ctrl_ratio_long_short_return == 0.0


class TestGaps:
    def test_compute_gaps(self):
        data = {"AAA": _df([100, 110, 121, 133.1, 146.41, 161.051],
                           signal=[True] + [False] * 5)}
        engine = BacktestEngine({
            "holding_days": 1,
            "targets": {"target_sharpe": 2.0, "target_total_return": 0.5,
                        "target_max_drawdown": 0.05},
        })
        result = engine.run("selection", ["AAA"], "2024-01-01", "2024-01-10",
                            data=data)
        assert result.gaps["total_return_gap"] == pytest.approx(0.10 - 0.5)
        assert result.gaps["sharpe_gap"] == pytest.approx(result.sharpe - 2.0)
        # 无回撤, 优于目标 5% → gap = (|0| - 0.05) * -1 = +0.05 (正=优于目标)
        assert result.gaps["max_drawdown_gap"] == pytest.approx(0.05)
        assert "ic_gap" not in result.gaps  # 未配置 target_ic

    def test_gaps_empty_targets(self):
        engine = BacktestEngine()
        assert engine.compute_gaps(BacktestResult()) == {}
