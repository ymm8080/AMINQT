# -*- coding: utf-8 -*-
"""规则引擎 v2 测试 (rule_engine_v2 四场景复现 + P11/P12 + PeakTracker 无未来函数)."""

from __future__ import annotations


from app.rules.config import Config, TUNABLE_BOUNDS, board_of, price_limit
from app.rules.peak_tracker import PeakTracker
from app.rules.rule_engine import (
    Action,
    Candidate,
    PortfolioState,
    Position,
    RuleEngine,
    Tick,
)


class MockFeed:
    """Mock IndicatorFeed (协议注入测试)."""

    def __init__(self, data: dict):
        self.d = data

    def _get(self, code, key, default):
        return self.d.get(code, {}).get(key, default)

    def control_ratio(self, code):
        return self._get(code, "control", 35.0)

    def had_accumulation_peak(self, code, lookback):
        return self._get(code, "acc_peak", True)

    def red_above_blue_since_peak(self, code):
        return self._get(code, "red_above", True)

    def red_blue_distance_min(self, code):
        return self._get(code, "rb_min", True)

    def control_weekly_up(self, code):
        return self._get(code, "wk_up", True)

    def bottom_breakout_volume(self, code):
        return self._get(code, "brk", False)

    def recent_shadow_lines(self, code):
        return self._get(code, "shadow", False)

    def red_bar_rising_and_majority(self, code):
        return self._get(code, "red_bar", True)

    def profit_chip_ratio(self, code):
        return self._get(code, "chip", 55.0)

    def latest_prob_up(self, code):
        return self._get(code, "prob", 0.60)


def make_engine():
    feed = MockFeed(
        {
            "600519": {
                "control": 42,
                "acc_peak": True,
                "red_above": True,
                "rb_min": True,
            },
            "300750": {"control": 25, "wk_up": True, "rb_min": False},
            "601318": {"control": 38, "rb_min": True, "chip": 35},
        }
    )
    return RuleEngine(feed), feed


def make_candidates():
    return [
        Candidate(
            "600519",
            "白酒",
            0.03,
            0.60,
            max_daily_gain_10d=9.0,
            turnover_today=12,
            daily_closes=[100, 101, 99, 102, 103],
        ),
        Candidate(
            "300750",
            "电池",
            0.025,
            0.58,
            max_daily_gain_10d=3.0,
            turnover_today=20,
            daily_closes=[50] * 5,
        ),
        Candidate(
            "601318", "保险", 0.02, 0.57, turnover_today=55, daily_closes=[40] * 5
        ),
    ]


class TestConfig:
    def test_board_and_limit(self):
        assert board_of("600519") == "MAIN" and price_limit("600519") == 10.0
        assert board_of("300750") == "GEM" and price_limit("300750") == 20.0
        assert board_of("830799") == "BSE" and price_limit("830799") == 30.0

    def test_tunable_bounds_cover_marked_fields(self):
        cfg = Config()
        for name in TUNABLE_BOUNDS:
            assert hasattr(cfg, name), name


class TestAfterClose:
    def test_step1_to_step4(self):
        """盘后场景: 600519/601318 自选标记 (控盘>30%+case1); 300750 控盘不足无标记;
        601318 换手 55>50 → 无日线买点."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        r = eng.after_close(1, make_candidates(), pf)
        assert {c.code for c in r["watch"]} == {"600519", "601318"}
        assert "300750" not in {c.code for c in r["watch"]}
        assert {c.code for c in r["daily_buy"]} == {"600519"}

    def test_daily_sell_mark(self):
        """收盘低于4日前 → 日线卖出标记 (次日执行)."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        pf.positions["601318"] = Position(
            "601318", "保险", cost=40.0, weight=0.10, buy_day=0
        )
        cands = make_candidates()
        cands[2].daily_closes = [44, 43, 42, 41, 40]  # 40 < 44
        r = eng.after_close(1, cands, pf)
        assert any(
            o.code == "601318" and "收盘低于4日前" in o.reason
            for o in r["daily_sell_marks"]
        )

    def test_prob_decay_mark_p12(self):
        """P12 双保险: 盘后 prob < 0.50 → 日线卖出标记."""
        eng, feed = make_engine()
        feed.d["601318"] = {"chip": 55, "prob": 0.45}
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        pf.positions["601318"] = Position(
            "601318", "保险", cost=40.0, weight=0.10, buy_day=0
        )
        r = eng.after_close(1, make_candidates(), pf)
        assert any("概率衰减" in o.reason for o in r["daily_sell_marks"])


class TestAuction:
    def test_p2_gap_sell(self):
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        pf.positions["601318"] = Position(
            "601318", "保险", cost=40.0, weight=0.10, buy_day=0
        )
        from app.rules.rule_engine import Order

        marks = [Order(1, "AFTER_CLOSE", "601318", Action.SELL_ALL, priority="P9")]
        orders = eng.on_auction(2, pf, {"601318": 42.4}, {"601318": 40.0}, marks)
        assert len(orders) == 1 and orders[0].priority == "P2"
        assert "601318" in pf.sold_today


class TestPeakTracker:
    def test_right_side_confirmation(self):
        """峰需回落0.8%×2根才确认; 无确认前不进 confirmed_peaks (因果性)."""
        tr = PeakTracker(Config())
        for px in [100, 101, 102]:  # 上升, 候选峰=102
            tr.update(px)
        assert tr.confirmed_peaks == []  # 无回落 → 未确认
        tr.update(102 * (1 - 0.009))  # 回落0.9% 第1根
        assert tr.confirmed_peaks == []
        tr.update(102 * (1 - 0.010))  # 持续第2根 → 确认
        assert tr.confirmed_peaks == [102]

    def test_three_peaks_descending(self):
        tr = PeakTracker(Config())
        tr.confirmed_peaks = [110, 105, 103]
        assert tr.three_peaks_descending()
        tr.confirmed_peaks = [110, 105, 106]
        assert not tr.three_peaks_descending()


class TestSellStateMachine:
    def _pf_with_pos(self, sellable=True):
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        pf.positions["600519"] = Position(
            "600519", "白酒", cost=100.0, weight=0.10, buy_day=1, sellable=sellable
        )
        return pf

    def _tick(self, price, time="10:00", volume=1.0, turnover=10):
        return {"600519": Tick(time, price, volume=volume, turnover=turnover)}

    def test_p1_hard_stop_unconditional(self):
        """P1 日内跌4%: 无日线标记也执行."""
        eng, _ = make_engine()
        pf = self._pf_with_pos()
        out = eng.on_tick(2, self._tick(95.5), [], pf, {"600519": 100.0}, 0.5, [])
        sells = [o for o in out if o.action == Action.SELL_ALL]
        assert sells and sells[0].priority == "P1"

    def test_p1_blocked_by_t1(self):
        """T+1 拦截: 当日新买跌-4.5% → P1 被 L0 拦截 → HOLD 挂单."""
        eng, _ = make_engine()
        pf = self._pf_with_pos(sellable=False)
        out = eng.on_tick(2, self._tick(95.5), [], pf, {"600519": 100.0}, 0.5, [])
        holds = [o for o in out if o.action == Action.HOLD]
        assert holds and "T+1" in holds[0].reason
        assert not any(o.action == Action.SELL_ALL for o in out)

    def test_p11_trailing_stop(self):
        """P11 移动止盈: 冲高108回落至103 (回撤≥4% 且曾盈利) → SELL_ALL."""
        eng, feed = make_engine()
        feed.d["600519"]["prob"] = 0.60  # P12 不抢跑
        pf = self._pf_with_pos()
        pf.positions["600519"].high_since_buy = 108.0
        out = eng.on_tick(2, self._tick(103.0), [], pf, {"600519": 100.0}, 0.5, [])
        sells = [o for o in out if o.action == Action.SELL_ALL]
        assert sells and sells[0].priority == "P11"

    def test_p11_not_triggered_when_unprofitable(self):
        """P11: high<=cost (未曾盈利) 不触发."""
        eng, feed = make_engine()
        feed.d["600519"]["prob"] = 0.60
        pf = self._pf_with_pos()
        pf.positions["600519"].high_since_buy = 99.0  # 成本 100
        out = eng.on_tick(2, self._tick(97.0), [], pf, {"600519": 100.0}, 0.5, [])
        assert not any(o.priority == "P11" for o in out)

    def test_p12_prob_decay(self):
        """P12 概率衰减: prob 0.45 < 0.50 → SELL_ALL (盘中)."""
        eng, feed = make_engine()
        feed.d["600519"] = {"chip": 55, "prob": 0.45}
        pf = self._pf_with_pos()
        out = eng.on_tick(2, self._tick(101.0), [], pf, {"600519": 100.0}, 0.5, [])
        sells = [o for o in out if o.action == Action.SELL_ALL]
        assert sells and sells[0].priority == "P12"

    def test_p7_half_sell_once(self):
        """P7 高换手减半: chg 7.5% (≥7 且 <8, 不触发 P6) → 减半, 且只执行一次."""
        eng, feed = make_engine()
        feed.d["600519"]["prob"] = 0.60
        pf = self._pf_with_pos()
        tick = self._tick(107.5, volume=1.5, turnover=45)
        out1 = eng.on_tick(2, tick, [], pf, {"600519": 100.0}, 0.5, [])
        assert any(o.action == Action.SELL_HALF for o in out1)
        out2 = eng.on_tick(2, tick, [], pf, {"600519": 100.0}, 0.5, [])
        assert not any(o.action == Action.SELL_HALF for o in out2)

    def test_p10_warn_no_sell(self):
        """P10 浮盈≥20% (相对成本): WARN 不自动卖 (chg 小, 不抢 P3)."""
        eng, feed = make_engine()
        feed.d["600519"]["prob"] = 0.60
        pf = self._pf_with_pos()
        out = eng.on_tick(2, self._tick(121.0), [], pf, {"600519": 118.0}, 0.5, [])
        warns = [o for o in out if o.action == Action.WARN]
        assert warns and warns[0].priority == "P10"

    def test_sold_today_no_rebuy(self):
        """当日卖出 → 当日禁止再买入."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        pf.sold_today.add("600519")
        c = Candidate("600519", "白酒", 0.03, 0.60)
        c.flag_daily_buy = True
        tick = {"600519": Tick("10:00", 105.0, volume=2.0)}
        out = eng.on_tick(2, tick, [c], pf, {"600519": 100.0}, 0.62, [])
        assert not any(o.action == Action.BUY for o in out)


class TestBuyPath:
    def test_path_a_gap_volume(self):
        """路径A: 跳空>4% + 放量 → 日内买点标记."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        c = Candidate("600519", "白酒", 0.03, 0.60)
        c.flag_daily_buy = True
        tick = {"600519": Tick("09:32", 105.0, volume=1.6)}
        eng.on_tick(2, tick, [c], pf, {"600519": 100.0}, 0.5, [])
        assert c.flag_intraday_buy

    def test_path_b_time_cutoff(self):
        """路径B: 10:40 后不触发."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        c = Candidate("600519", "白酒", 0.03, 0.60)
        c.flag_daily_buy = True
        tick = {"600519": Tick("10:41", 101.0, volume=1.2, big_order_net=5e6)}
        eng.on_tick(2, tick, [c], pf, {"600519": 100.0}, 0.62, [])
        assert not c.flag_intraday_buy

    def test_entry_shape_down_then_up(self):
        """入场形态: 前10分钟下行 → 低峰确认回升 → BUY (v2 场景复现)."""
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        c = Candidate("600519", "白酒", 0.03, 0.60)
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
        buys = []
        for t, px in zip(times, seq):
            ticks = {"600519": Tick(t, px, volume=1.6, turnover=8, big_order_net=5e6)}
            out = eng.on_tick(2, ticks, [c], pf, {"600519": 100.0}, 0.62, [])
            buys += [o for o in out if o.action == Action.BUY]
        assert len(buys) == 1 and buys[0].priority == "L4-形态"


class TestPortfolioGate:
    def test_gates(self):
        eng, _ = make_engine()
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6, cooldown_left=2)
        assert eng.portfolio_gate(pf)[0] == 0.0
        pf.cooldown_left = 0
        pf.nav, pf.peak_nav = 0.89e6, 1e6  # 回撤 -11%
        assert eng.portfolio_gate(pf)[0] == 0.0
        pf.nav = 0.94e6  # 回撤 -6%
        assert eng.portfolio_gate(pf)[0] == 0.60
        pf.nav = 1e6
        pf.today_pnl_pct = -0.025  # 浮亏熔断
        assert eng.portfolio_gate(pf)[0] == 0.0
        pf.today_pnl_pct = 0.0
        assert eng.portfolio_gate(pf)[0] == 0.95
