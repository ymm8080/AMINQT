"""P20 V5.1 批3: 券商适配层 / 对账 / 半自动 / 配置冻结与紧急开关."""

from __future__ import annotations

import pandas as pd
import pytest

from app.intraday.v51.broker_adapter import (
    MiniQMTAdapter,
    Order,
    SimulationAdapter,
)
from app.intraday.v51.config import (
    ConfigFrozenError,
    EmergencySwitch,
    ProdConfig,
)
from app.intraday.v51.reconcile import slippage_report, trade_match_rate
from app.intraday.v51.semi_auto import SemiAutoDesk, ticket_from_order
from app.intraday.v51.worm_logger import WormLogger


class TestBrokerAdapter:
    def test_simulation_fill_and_positions(self):
        sim = SimulationAdapter({"600519": 1e9})
        sim.place_order(Order("600519", "buy", 1000, 100.0, "B2"))
        sim.place_order(Order("600519", "sell", 400, 101.0, "S2"))
        assert sim.positions()["600519"] == 600
        assert len(sim.fills) == 2
        assert sim.fills[0].slippage == 0.0

    def test_simulation_rejects_over_adv(self):
        sim = SimulationAdapter({"600519": 1e6})  # ADV 100万 → 单笔上限 1万
        with pytest.raises(ValueError):
            sim.place_order(Order("600519", "buy", 1000, 100.0))

    def test_auction_fill(self):
        sim = SimulationAdapter()
        f = sim.place_auction_order(Order("600519", "sell", 1000, 90.0, "S8"))
        assert f.auction

    def test_miniqmt_dry_run(self):
        mq = MiniQMTAdapter(dry_run=True)
        f = mq.place_order(Order("600519", "buy", 100, 100.0))
        assert f.filled_qty == 0  # 只记录意图
        with pytest.raises(NotImplementedError):
            MiniQMTAdapter(dry_run=False)


class TestReconcile:
    def test_match_rate(self):
        bt = pd.DataFrame([
            {"symbol": "A", "side": "buy", "rule": "B2"},
            {"symbol": "A", "side": "sell", "rule": "S1"},
            {"symbol": "B", "side": "buy", "rule": "B1"},
        ])
        live_same = bt.copy()
        assert trade_match_rate(bt, live_same)["pass"]
        live_missing = bt.iloc[:2]
        r = trade_match_rate(bt, live_missing)
        assert r["match_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert not r["pass"] and r["missing_in_live"]

    def test_slippage_report(self):
        fills = pd.DataFrame({
            "symbol": ["A", "B", "C"],
            "slippage": [0.0006, 0.0011, 0.0040],
        })
        assumed = pd.Series({"A": 0.0005, "B": 0.0010, "C": 0.0015})
        r = slippage_report(fills, assumed)
        assert r["alert"] and r["exceed_symbols"] == ["C"]  # 0.004 > 2×0.0015
        assert r["mean_assumed"] > 0


class TestSemiAuto:
    def test_signal_flow(self, tmp_path):
        desk = SemiAutoDesk(WormLogger(str(tmp_path)))
        desk.push_signal(ticket_from_order(
            "2026-07-25", Order("600519", "buy", 1000, 100.0, "B2")))
        assert len(desk.pending()) == 1
        desk.confirm_fill("2026-07-25", "600519", 1000, 100.1)
        assert len(desk.pending()) == 0
        desk.report_deviation("2026-07-25", "600000", "ignored_signal", "流动性差")
        with pytest.raises(AssertionError):
            desk.report_deviation("2026-07-25", "X", "bad_reason")
        # WORM: signal/order/manual 三类留痕
        recs = desk.worm.read_day("2026-07-25")
        assert {r["type"] for r in recs} == {"signal", "order", "manual"}


class TestProdConfigAndSwitch:
    def test_freeze_audit(self):
        cfg = ProdConfig()
        cfg.generate({"min_net_edge": 0.005, "profile": "C"})
        assert cfg.get("min_net_edge") == 0.005
        with pytest.raises(ConfigFrozenError):
            cfg.set("min_net_edge", 0.008)  # 盘中禁止热更新
        cfg.new_day()
        cfg.generate({"min_net_edge": 0.008})  # 次日盘前可重新生成
        assert cfg.get("min_net_edge") == 0.008

    def test_emergency_switch(self, tmp_path):
        sw = EmergencySwitch(WormLogger(str(tmp_path)))
        assert sw.allow_buy()
        sw.activate("2026-07-25", "close_only", "op01", "大盘闪崩")
        assert not sw.allow_buy() and sw.allow_sell()
        sw.activate("2026-07-25", "force_liquidate", "op01", "系统性风险")
        assert sw.must_liquidate()
        sw.activate("2026-07-25", "resume", "op01", "风险解除")
        assert sw.allow_buy() and not sw.must_liquidate()
        with pytest.raises(AssertionError):
            sw.activate("2026-07-25", "pause", "", "")  # 缺操作员/原因
        recs = sw.worm.read_day("2026-07-25")
        assert len(recs) == 3 and all(r["type"] == "manual" for r in recs)
