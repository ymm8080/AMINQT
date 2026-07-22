# -*- coding: utf-8 -*-
"""P12 端到端集成测试 (V3.5 全链路): 选股 → 标记 → 盘中交易 → 回测 → 调参.

链路覆盖 (IMPLEMENTATION_PLAN v2.8 P12):
  1. DailySelectionPipeline.run  → 清单 schema V1.0 (P14)
  2. RuleEngine.after_close      → STEP2/3/4 标记 (P15, CompositeFeed 真实指标源)
  3. RuleEngine.on_tick          → 盘中订单 (P1-P12 状态机)
  4. BacktestEngineV35.run       → V3.5 回测协议绩效 (P9)
  5. ParamTuner.grid_search      → 规则参数回测调优 + OOS 复验 (用户需求)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators.chip_distribution import ChipDistribution, ChipFeed
from app.indicators.indicator_feed import CompositeFeed
from app.indicators.yimeng_dingdi import YimengFeed, yimeng_dingdi
from app.indicators.zhuli_lasheng import zhuli_lasheng
from app.pipeline1.backtest_v35 import BacktestEngineV35
from app.pipeline1.list_generator import SCHEMA_FIELDS
from app.pipeline1.param_tuner import ParamTuner
from app.rules.config import Config
from app.rules.rule_engine import (
    Action,
    Candidate,
    PortfolioState,
    Position,
    RuleEngine,
    Tick,
)
from tests.test_daily_pipeline import _StubFeatures, _train_bundle, make_panel
from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.data_supply import DataSupplyChain


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """全链路夹具: 面板 + 训练模型包 + Pipeline + 真实 CompositeFeed."""
    tmp = tmp_path_factory.mktemp("e2e")
    panel = make_panel()
    bundle = _train_bundle(tmp, panel)

    pipe = DailySelectionPipeline(
        supply=DataSupplyChain(cache_dir=str(tmp / "cache")),
        bundle_paths={"main": bundle},
        list_dir=str(tmp / "lists"),
    )
    pipe.features = _StubFeatures()
    pipe.cleaner = CleaningPipeline(CleaningConfig(valve_full=2, valve_reduced=1))

    # 真实指标源 (NECESSARY INDICATOR 复刻)
    df = panel[panel["symbol"] == "600519"].copy()
    feed = CompositeFeed(
        yimeng=YimengFeed({"600519": yimeng_dingdi(df.copy())}),
        chip=ChipFeed(
            {"600519": ChipDistribution().build(df.copy(), float_shares=1e8)}
        ),
        capital=None,
        zhuli_hist={"600519": zhuli_lasheng(df.copy())},
        daily_hist={"600519": df},
        prob_provider=lambda code: 0.60,
    )
    return {"panel": panel, "pipe": pipe, "feed": feed, "tmp": tmp}


class TestEndToEndV35:
    def test_full_workflow(self, chain):
        panel, pipe, feed = chain["panel"], chain["pipe"], chain["feed"]

        # ── 1. Pipeline-1 选股: 清单 schema V1.0 ──
        r1 = pipe.run("20260720", panel=panel)
        assert r1["mode"] == "normal"
        lst = r1["list"]
        assert list(lst.columns) == SCHEMA_FIELDS and len(lst) > 0

        # ── 2. 规则引擎盘后标记 (真实 CompositeFeed) ──
        eng = RuleEngine(feed, Config())
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        cands = [
            Candidate(
                str(r["symbol"]),
                "白酒",
                float(r["pred_ret_1d"]),
                float(r["prob_up"]),
                turnover_today=12,
                daily_closes=[100, 101, 99, 102, 103],
            )
            for _, r in lst.iterrows()
        ]
        marks = eng.after_close(1, cands, pf)
        assert len(marks["pool"]) == len(lst)  # 全部进池
        assert isinstance(marks["watch"], list)

        # ── 3. 盘中交易: tick → 订单 ──
        sym = str(lst.iloc[0]["symbol"])
        if sym != "600519":  # CompositeFeed 只有 600519
            pytest.skip("清单首选非 600519, 跳过盘中段 (指标源仅单股)")
        for c in cands:
            c.flag_daily_buy = True
        seq = [100.0, 99.6, 99.0, 98.4, 98.2, 98.3, 98.9, 99.4, 99.8]
        times = [
            "09:32",
            "09:34",
            "09:36",
            "09:38",
            "09:40",
            "09:42",
            "09:44",
            "09:46",
            "09:48",
        ]
        orders = []
        for t, px in zip(times, seq):
            out = eng.on_tick(
                2,
                {sym: Tick(t, px, volume=1.6, turnover=8, big_order_net=5e6)},
                cands,
                pf,
                {sym: 100.0},
                0.62,
                [],
            )
            orders += out
        assert any(o.action == Action.BUY for o in orders)  # 形态买入成交
        assert sym in pf.bought_today

        # ── 4. 持仓 + 卖出状态机 (P1 硬止损) ──
        pf.positions[sym] = Position(
            sym, "白酒", cost=99.4, weight=0.10, buy_day=1, sellable=True
        )
        out = eng.on_tick(
            3, {sym: Tick("10:00", 95.0, volume=1.0)}, [], pf, {sym: 99.4}, 0.5, []
        )
        assert any(o.action == Action.SELL_ALL and o.priority == "P1" for o in out)

        # ── 5. 回测 (V3.5 协议) ──
        lists = {
            d: pd.DataFrame(
                {
                    "symbol": g["symbol"].values,
                    "score": np.linspace(1, 0.5, len(g)),
                    "prob_up": 0.60,
                    "industry": g["industry"].values,
                }
            )
            for d, g in list(panel.groupby("date"))[-30:]
        }
        bt = BacktestEngineV35(panel).run(lists)
        m = bt["metrics"]
        assert m["n_days"] == len(panel["date"].unique())
        assert -1.0 <= m["max_drawdown"] <= 0
        assert len(bt["trades"]) > 0

        # ── 6. 参数调优: 网格 + OOS 复验 + 写回 Config ──
        tuner = ParamTuner(panel, lists, report_dir=str(chain["tmp"]))
        report = tuner.grid_search(["max_hold_days", "prob_exit"], top_k=2)
        assert "best_params" in report
        cfg = ParamTuner.apply_to_config(report["best_params"], Config())
        assert cfg.max_hold_days in (2, 3, 4, 5)

    def test_no_import_errors(self):
        """P12 验证: 全部新模块可导入."""
        import app.pipeline1.daily_pipeline  # noqa: F401
        import app.pipeline1.predictor  # noqa: F401
        import app.pipeline1.backtest_v35  # noqa: F401
        import app.pipeline1.param_tuner  # noqa: F401
        import app.indicators.indicator_feed  # noqa: F401
        import app.rules.rule_engine  # noqa: F401
