"""P20 V5.1 日内引擎核心测试: T+1/时段/成本/B1-B8/S1-S8/资金纪律/WORM/灯/四态."""

from __future__ import annotations

import pytest

from app.intraday.v51.buy_engine import (
    Bar,
    BuyContext,
    b1_vwap_pullback,
    b2_evening_strength,
    limit_order_price,
    trigger,
)
from app.intraday.v51.cost_model import round_trip_cost, slippage_tier
from app.intraday.v51.fund_manager import FundManager
from app.intraday.v51.position import Position
from app.intraday.v51.sell_engine import SellContext
from app.intraday.v51.sell_engine import trigger as sell_trigger
from app.intraday.v51.sessions import buy_window_open, sell_window_open
from app.intraday.v51.state_machine import ParamStateMachine, shadow_gate
from app.intraday.v51.traffic_light import IntradayTrafficLight, plateau_drift
from app.intraday.v51.worm_logger import WormLogger


# ============================================================
# T+1 状态机 (铁律 #0)
# ============================================================
class TestPosition:
    def test_t1_lock_and_settle(self):
        pos = Position(
            "600519",
            total_qty=100,
            sellable_qty=0,
            entry_price=100.0,
            entry_date="2026-07-25",
        )
        assert not pos.can_sell()  # 当日买入锁死 (满仓的代价)
        assert pos.on_sell(100) == 0  # 物理不可卖
        pos.settle_overnight()
        assert pos.can_sell() and pos.hold_days == 1
        assert pos.on_sell(100) == 100

    def test_max_price_anchor(self):
        pos = Position("600519", 100, 100, 100.0, "2026-07-25")
        pos.on_bar(105.0)
        pos.on_bar(103.0)
        assert pos.max_price_since_entry == 105.0


# ============================================================
# 时段 (单点引用 + 熊市接管)
# ============================================================
class TestSessions:
    def test_windows(self):
        assert not buy_window_open("09:30")  # 开盘噪声期
        assert buy_window_open("09:50", signal_grade="A")  # 早盘 A 级
        assert not buy_window_open("09:50", signal_grade="B")  # 早盘仅 A 级
        assert not buy_window_open("11:00")  # 盘中只卖不买
        assert buy_window_open("14:40")  # 尾盘主窗口 (不限等级)
        assert sell_window_open("14:45") and not sell_window_open("13:00")

    def test_bear_override(self):
        assert not buy_window_open("14:40", bear_state="DEFENSE")  # 只卖不买
        assert not buy_window_open(
            "09:50", bear_state="RECOVERY", signal_grade="A"
        )  # 首周早盘关闭
        assert buy_window_open("14:40", bear_state="RECOVERY")  # 尾盘仍开
        assert not buy_window_open(
            "09:50", signal_grade="A", hs300_change=-0.015
        )  # 单边下跌禁用早盘


# ============================================================
# 成本模型 (同源)
# ============================================================
class TestCostModel:
    def test_slippage_tier_shared_with_pipeline1(self):
        from app.pipeline1.label_engine import slippage_tier as p1_tier

        assert slippage_tier is p1_tier  # 单点同源 (检查清单 #12)

    def test_round_trip(self):
        c = round_trip_cost(adv_20d=6e8, order_value=1e5)
        # 佣金×2 + 印花税 + 滑点0.05%×2 + 冲击≈0
        assert c == pytest.approx(0.0005 + 0.0005 + 0.001 + 0.5 * (1e5 / 6e8) ** 0.5)


# ============================================================
# 买入引擎 B1-B8 (纯函数)
# ============================================================
def _ctx(**kw):
    base = {
        "symbol": "600519",
        "t": "14:40",
        "price": 100.0,
        "pre_close": 98.0,
        "pred_q50": 0.05,
        "atr_pct": 0.02,
        "stop_price": 94.0,
        "adv_20d": 1e9,
        "order_value": 1e5,
        "bar_amount": 2e6,
        "event_mean": 0.06,
        "oos_decay": 0.0,
    }
    return BuyContext(**{**base, **kw})


def _bars(n=30, close=98.0, vol=1e6, vwap=99.5):
    # 默认 close=98: 不触发 B1 (跌穿 VWAP×(1-1%)) 也不触发 B2 (未站上均线)
    return tuple(
        Bar(t=f"14:{i:02d}", close=close, volume=vol, amount=close * vol, vwap=vwap)
        for i in range(n)
    )


class TestBuyEngine:
    def test_veto_chain(self):
        assert not trigger(_ctx(), _bars())["pass"]  # B1 未触发 (close<vwap×0.998?)
        r = trigger(_ctx(price=107.5), _bars())  # B3 涨>7%? 107.5/98-1=9.7% → 否决
        assert "B3" in r["vetoes"]
        r = trigger(_ctx(t="09:30"), _bars())
        assert "B4" in r["vetoes"]
        r = trigger(_ctx(pred_q50=0.001, event_mean=0.001), _bars())
        assert "B5" in r["vetoes"]  # 净收益不足
        r = trigger(_ctx(bar_amount=5e5), _bars())
        assert "B6" in r["vetoes"]
        r = trigger(_ctx(stop_price=99.5), _bars())  # 止损距离 0.5% < 1.2×2%
        assert "B7" in r["vetoes"]
        r = trigger(_ctx(sector_drop_count=2), _bars())
        assert "B8" in r["vetoes"]

    def test_b1_vwap_pullback(self):
        bars = _bars(close=99.6, vwap=99.5)  # 回踩幅度内且站稳
        assert b1_vwap_pullback(bars[-3:])
        assert not b1_vwap_pullback(_bars(close=98.0, vwap=99.5)[-3:])  # 跌穿

    def test_b2_evening_strength(self):
        bars = list(_bars(n=30, close=99.0, vol=1e6))
        bars.append(
            Bar("14:55", close=102.0, volume=3e6, amount=3e8, vwap=99.5)
        )  # 放量3倍+站上均线
        assert b2_evening_strength(tuple(bars), vol_ratio=1.5, ma_window=24)

    def test_full_pass(self):
        bars = list(_bars(n=30, close=99.0, vol=1e6))
        bars.append(Bar("14:55", close=102.0, volume=3e6, amount=3e8, vwap=99.0))
        r = trigger(_ctx(price=102.0, pre_close=100.0), tuple(bars))
        assert r["pass"] and r["positive"] == "B2"

    def test_limit_order_one_tick(self):
        assert limit_order_price(100.0, "buy") > 100.0
        assert limit_order_price(100.0, "sell") < 100.0

    def test_trigger_is_pure(self):
        """铁律 #1 审计: buy/sell 引擎源码无 IO/无随机/无外部调用."""
        import inspect
        import re

        import app.intraday.v51.buy_engine as be
        import app.intraday.v51.sell_engine as se

        forbidden = (
            r"\bopen\(",
            r"\brequests\b",
            r"\brandom\b",
            r"datetime\.now\(",
            r"\bakshare\b",
            r"\bxtquant\b",
        )
        for mod in (be, se):
            src = inspect.getsource(mod)
            for pat in forbidden:
                assert not re.search(pat, src), f"{mod.__name__} 命中 {pat}"


# ============================================================
# 卖出引擎 S1-S8 (优先级链)
# ============================================================
def _pos(**kw):
    base = {
        "symbol": "600519",
        "total_qty": 100,
        "sellable_qty": 100,
        "entry_price": 100.0,
        "entry_date": "2026-07-23",
        "hold_days": 2,
        "stop_price": 95.0,
        "max_price_since_entry": 106.0,
    }
    return Position(**{**base, **kw})


class TestSellEngine:
    def test_t1_gate(self):
        r = sell_trigger(
            SellContext(t="14:40", price=90.0, limit_down_price=88.0),
            _pos(sellable_qty=0),
        )
        assert r["action"] == "HOLD" and "T+1" in r["reason"]

    def test_priority_order(self):
        pos = _pos()
        # S8 跌停 > 一切 (即使 S1 也同时满足)
        r = sell_trigger(SellContext(t="14:40", price=90.0, limit_down_price=90.0), pos)
        assert r["rule"] == "S8" and r["action"] == "AUCTION_SELL"
        # S3 换手异动 > S1
        r = sell_trigger(
            SellContext(
                t="14:40",
                price=94.0,
                limit_down_price=80.0,
                turnover_pct=0.45,
                change_pct=0.09,
            ),
            pos,
        )
        assert r["rule"] == "S3"
        # S6 炸板 > S1
        r = sell_trigger(
            SellContext(
                t="14:40",
                price=94.0,
                limit_down_price=80.0,
                limit_up_price=105.0,
                touched_limit_up=True,
            ),
            pos,
        )
        assert r["rule"] == "S6"
        # S1 动态止损 (stop_price 下发制)
        r = sell_trigger(SellContext(t="11:00", price=94.5, limit_down_price=80.0), pos)
        assert r["rule"] == "S1"
        # S2 移动止盈: 最高 106 浮盈 6% 激活, 回撤带 max(3%, 2%)=3% → 102.8
        r = sell_trigger(
            SellContext(t="11:00", price=102.0, limit_down_price=80.0, atr_pct=0.02),
            _pos(stop_price=90.0),
        )
        assert r["rule"] == "S2"

    def test_s5a_s5b_tail_window(self):
        pos = _pos(stop_price=50.0, max_price_since_entry=100.5)
        # S5a: 满2日涨<1% (尾盘) — price 100.5 < 101 → 触发
        r = sell_trigger(
            SellContext(t="14:40", price=100.5, limit_down_price=80.0), pos
        )
        assert r["rule"] == "S5a"
        # 非尾盘不触发 S5a/S5b
        r = sell_trigger(
            SellContext(t="11:00", price=100.5, limit_down_price=80.0), pos
        )
        assert r["action"] == "HOLD"
        # S5b: 满2日到期 (涨幅≥1% 也触发)
        r = sell_trigger(
            SellContext(t="14:50", price=103.0, limit_down_price=80.0),
            _pos(stop_price=50.0, max_price_since_entry=103.0),
        )
        assert r["rule"] == "S5b"

    def test_s7_s4(self):
        pos = _pos(stop_price=50.0, max_price_since_entry=100.0, hold_days=1)
        r = sell_trigger(
            SellContext(
                t="14:40", price=101.0, limit_down_price=80.0, invalidation=True
            ),
            pos,
        )
        assert r["rule"] == "S7/S4"


# ============================================================
# 资金纪律
# ============================================================
class TestFundManager:
    def test_daily_limits(self):
        fm = FundManager()
        assert fm.can_buy("A", 1.0)[0]
        fm.on_buy("A")
        assert not fm.can_buy("A", 1.0)[0]  # 当日只买一次
        fm.on_buy("B")
        assert not fm.can_buy("C", 1.0)[0]  # 每日 ≤2 笔
        fm.new_day()
        assert fm.can_buy("A", 1.0)[0]  # 次日重置

    def test_cooldown_and_weight(self):
        fm = FundManager()
        fm.on_stop_loss("A")
        assert "冷却" in fm.can_buy("A", 1.0)[1]
        assert "weight=0" in fm.can_buy("B", 0.0)[1]

    def test_fuse_and_halt(self):
        fm = FundManager()
        fm.on_daily_pnl(-0.045)
        assert not fm.can_buy("A", 1.0)[0]  # 保险丝
        fm.new_day()
        fm.on_nav(100.0)
        fm.on_nav(84.0)  # 回撤 16%
        assert fm.halted and not fm.can_buy("A", 1.0)[0]


# ============================================================
# WORM / 红黄绿灯 / 四态流转
# ============================================================
class TestInfra:
    def test_worm_roundtrip(self, tmp_path):
        w = WormLogger(str(tmp_path))
        w.log("2026-07-25", "signal", {"symbol": "600519", "rule": "B2"})
        w.log("2026-07-25", "order", {"symbol": "600519", "side": "buy"})
        w.log("2026-07-25", "manual", {"action": "紧急暂停"})
        recs = w.read_day("2026-07-25")
        assert len(recs) == 3
        assert [r["type"] for r in recs] == ["signal", "order", "manual"]
        with pytest.raises(AssertionError):
            w.log("2026-07-25", "bogus", {})

    def test_traffic_light(self):
        tl = IntradayTrafficLight()
        # μ≈1.024, σ≈0.025 → 1.0 在 μ-1σ (0.999) 之内 → GREEN
        tl.sharpe_history = [1.0, 1.05] * 12 + [1.0]
        assert tl.daily_check(1.0)["light"] == "GREEN"
        assert tl.daily_check(-2.0)["light"] == "RED"  # 跌破 2σ
        tl2 = IntradayTrafficLight()
        tl2.sharpe_history = [1.0] * 20 + [1.3] * 5  # μ≈1.06, σ≈0.11
        for _ in range(3):
            r = tl2.daily_check(0.9)  # 介于 μ-1σ 与 μ-2σ 之间
        assert r["light"] == "YELLOW" and "复检" in r["action"]

    def test_plateau_drift(self):
        r = plateau_drift(
            {"pullback": 0.01, "vol_ratio": 1.5}, {"pullback": 0.013, "vol_ratio": 1.5}
        )
        assert r["need_review"] and "pullback" in r["drifted"]

    def test_state_machine(self):
        sm = ParamStateMachine()
        sm.confirm_by_human()
        good = {
            "excess_return": 0.01,
            "veto_rate": 0.15,
            "data_failures": 0,
            "match_rate": 0.995,
        }
        assert shadow_gate(good)["pass"]
        sm.promote_from_shadow(good)
        assert sm.state == "staging"
        sm.promote_from_staging(abnormal=False)
        assert sm.state == "active"
        sm.demote("红灯下线")
        assert sm.state == "candidate"
        bad = {**good, "match_rate": 0.95}
        assert not shadow_gate(bad)["pass"]
