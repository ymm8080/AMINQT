# -*- coding: utf-8 -*-
"""回测协议 + 参数调优器测试."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.backtest_v35 import (
    BacktestEngineV35,
    BacktestProtocol,
)
from app.pipeline1.param_tuner import ParamTuner
from app.rules.config import Config


def make_panel(
    symbols=("600519", "601318", "600000"), days=120, seed=9
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    frames = []
    for sym in symbols:
        # 温和上行趋势, 让清单策略有正期望
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, days))
        open_ = close * (1 + rng.normal(0, 0.003, days))
        pre_close = np.concatenate([[close[0]], close[:-1]])
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "pre_close": pre_close,
                    "board": "main",
                    "industry": "白酒" if sym == "600519" else "保险",
                    "amount": 1e9,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_lists(panel: pd.DataFrame) -> dict:
    """每日清单: 固定给全部候选, score 随机."""
    rng = np.random.default_rng(3)
    lists = {}
    for d, g in panel.groupby("date"):
        lists[d] = pd.DataFrame(
            {
                "symbol": g["symbol"].values,
                "score": rng.uniform(0, 1, len(g)),
                "prob_up": 0.60,
                "industry": g["industry"].values,
            }
        )
    return lists


class TestBacktestProtocol:
    def test_run_and_metrics_keys(self):
        panel = make_panel()
        eng = BacktestEngineV35(panel)
        result = eng.run(make_lists(panel))
        m = result["metrics"]
        for key in (
            "total_return",
            "annual_return",
            "net_excess_annual",
            "max_drawdown",
            "sharpe",
        ):
            assert key in m
        assert len(result["nav_curve"]) == 120
        assert len(result["trades"]) > 0

    def test_costs_reduce_return(self):
        """换手成本: 有成本收益 < 无成本收益 (协议 §4)."""
        panel = make_panel()
        lists = make_lists(panel)
        with_cost = BacktestEngineV35(panel).run(lists)["metrics"]["total_return"]
        free = BacktestProtocol(slippage=0, commission=0, stamp_tax=0)
        no_cost = BacktestEngineV35(panel, free).run(lists)["metrics"]["total_return"]
        assert with_cost < no_cost

    def test_max_hold_days_forces_exit(self):
        """持仓约束: max_hold_days=2 → 所有卖出 reason 无 '持仓满3日'."""
        panel = make_panel()
        proto = BacktestProtocol(max_hold_days=2)
        result = BacktestEngineV35(panel, proto).run(make_lists(panel))
        sells = result["trades"][result["trades"]["side"] == "sell"]
        assert len(sells) > 0
        assert not sells["reason"].str.contains("持仓满3日").any()

    def test_limit_up_no_buy(self):
        """协议 §2: T+1 一字涨停买单放弃."""
        panel = make_panel(symbols=("600519",), days=10)
        d1 = panel["date"].unique()[1]
        row = panel[panel["date"] == d1].index[0]
        lu = round(panel.loc[row, "pre_close"] * 1.10, 2)
        panel.loc[row, "open"] = lu  # 一字涨停开盘
        lists = {
            panel["date"].unique()[0]: pd.DataFrame(
                {
                    "symbol": ["600519"],
                    "score": [1.0],
                    "prob_up": [0.6],
                    "industry": ["白酒"],
                }
            )
        }
        result = BacktestEngineV35(panel).run(lists)
        buys = result["trades"][result["trades"]["side"] == "buy"]
        assert not ((buys["date"] == d1) & (buys["symbol"] == "600519")).any()


class TestParamTuner:
    def test_grid_search_and_report(self, tmp_path):
        panel = make_panel(days=120)
        lists = make_lists(panel)
        tuner = ParamTuner(panel, lists, report_dir=str(tmp_path))
        report = tuner.grid_search(
            ["max_hold_days", "prob_exit"], oos_ratio=0.3, top_k=3
        )
        assert report["n_combos"] == 4 * 5  # 2/3/4/5 × 0.40..0.60 step .05
        assert "best_params" in report and "oos_score" in report
        assert len(report["leaderboard"]) == 3
        import os

        assert os.path.exists(report["report_path"])

    def test_apply_to_config(self):
        cfg = Config()
        ParamTuner.apply_to_config(
            {"max_hold_days": 4, "prob_exit": 0.55, "not_a_param": 999}, cfg
        )
        assert cfg.max_hold_days == 4
        assert cfg.prob_exit == 0.55
        assert not hasattr(cfg, "not_a_param")
