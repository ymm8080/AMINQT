"""P20.4 V5.1 检查清单专项: 纯函数AST审计 / 熊市接管 / 规则口径 / S8跌停逃生.

覆盖检查清单条目:
  #2  trigger() 纯函数审计 (AST 级: 无IO/无随机/无外部调用/无时钟读取)
  #5a B7 止损距离否决 (无止损价 = 无保护 → 否决)
  #5  B5 双输入取小 (pred_q50 与事件研究均值取小, 再乘样本外衰减)
  #7  S1 stop_price 下发制 (stop_price=0 不触发, 无全局固定 -4%)
  #8  S2 ATR 自适应回撤带 (浮盈 <3% 不激活)
  #10 S8 跌停逃生协议 (最高优先级 / 排队不减仓 / T+1 锁死优先)
  #11 熊市协议接管 (DEFENSE 只卖不买; RECOVERY 早盘关闭 + 仓位上限 30%)
"""

from __future__ import annotations

import ast
import inspect

import pytest

import app.intraday.v51.buy_engine as buy_engine
import app.intraday.v51.sell_engine as sell_engine
from app.intraday.v51.backtest_engine import DayContext, IntradayBacktester
from app.intraday.v51.buy_engine import Bar, BuyContext
from app.intraday.v51.buy_engine import trigger as buy_trigger
from app.intraday.v51.position import Position
from app.intraday.v51.sell_engine import SellContext
from app.intraday.v51.sell_engine import trigger as sell_trigger
from app.intraday.v51.sessions import (
    RECOVERY_POSITION_CAP,
    buy_window_open,
    position_cap,
)


# ============================================================
# #2 纯函数审计 (AST 级, 比正则严格: 抓 import / 调用 / 属性读取)
# ============================================================
class TestTriggerPurityAudit:
    """铁律 #1: buy/sell 引擎 trigger() 纯函数 — AST 静态审计."""

    # 允许的 import (白名单): 纯数据结构 + 同包纯计算模块
    ALLOWED_IMPORTS = {
        "__future__",
        "dataclasses",
        "typing",
        "cost_model",
        "safe_div",
        "position",
        "sessions",
    }
    # 禁止的直接调用
    FORBIDDEN_CALLS = {
        "open",
        "print",
        "input",
        "eval",
        "exec",
        "compile",
        "__import__",
        "globals",
        "locals",
    }
    # 禁止的属性调用 (时钟/随机/网络/磁盘 的典型入口)
    FORBIDDEN_ATTR_CALLS = {
        "now",
        "today",
        "utcnow",
        "random",
        "randint",
        "uniform",
        "sleep",
        "get",
        "post",
        "request",
        "read",
        "write",
    }

    @staticmethod
    def _audit(mod) -> list[str]:
        tree = ast.parse(inspect.getsource(mod))
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    if top not in TestTriggerPurityAudit.ALLOWED_IMPORTS:
                        violations.append(f"{mod.__name__}: import {a.name}")
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[-1]  # 相对导入取末段
                abs_top = (node.module or "").split(".")[0]
                if node.level == 0 and abs_top not in (
                    TestTriggerPurityAudit.ALLOWED_IMPORTS
                ):
                    violations.append(f"{mod.__name__}: from {node.module} import ...")
                elif node.level > 0 and top not in (
                    TestTriggerPurityAudit.ALLOWED_IMPORTS
                ):
                    violations.append(f"{mod.__name__}: from .{node.module} import ...")
            elif isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name) and f.id in (
                    TestTriggerPurityAudit.FORBIDDEN_CALLS
                ):
                    violations.append(f"{mod.__name__}: 调用 {f.id}() (行{f.lineno})")
                elif (
                    isinstance(f, ast.Attribute)
                    and f.attr in TestTriggerPurityAudit.FORBIDDEN_ATTR_CALLS
                ):
                    violations.append(f"{mod.__name__}: 调用 .{f.attr}() (行{f.lineno})")
        return violations

    def test_buy_engine_pure(self):
        assert self._audit(buy_engine) == []

    def test_sell_engine_pure(self):
        assert self._audit(sell_engine) == []

    def test_trigger_deterministic(self):
        """行为级审计: 同输入两次调用结果一致 (无隐藏时钟/随机)."""
        ctx = BuyContext(
            symbol="600519",
            t="14:40",
            price=100.0,
            pre_close=98.0,
            pred_q50=0.05,
            atr_pct=0.02,
            stop_price=94.0,
            adv_20d=1e9,
            order_value=1e5,
            bar_amount=2e6,
        )
        bars = tuple(
            Bar(t="14:40", close=98.0, volume=1e6, amount=9.8e7, vwap=99.5)
            for _ in range(3)
        )
        assert buy_trigger(ctx, bars) == buy_trigger(ctx, bars)
        pos = Position("600519", 100, 100, 100.0, "2026-07-23", stop_price=95.0)
        sctx = SellContext(t="11:00", price=94.0, limit_down_price=80.0)
        assert sell_trigger(sctx, pos) == sell_trigger(sctx, pos)


# ============================================================
# #11 熊市协议接管 (DEFENSE / RECOVERY 行为差异)
# ============================================================
class TestBearTakeover:
    def test_defense_blocks_all_buy_windows(self):
        """DEFENSE = 只卖不买: 所有时段买入窗口全关."""
        for t in ("09:50", "10:00", "14:40", "14:55"):
            assert not buy_window_open(t, bear_state="DEFENSE", signal_grade="A")

    def test_recovery_morning_closed_evening_open(self):
        """RECOVERY = 早盘关闭, 尾盘仍开 (首周恢复纪律)."""
        assert not buy_window_open("09:50", bear_state="RECOVERY", signal_grade="A")
        assert buy_window_open("14:40", bear_state="RECOVERY")

    def test_position_cap(self):
        """RECOVERY 仓位上限 30% (V5.1 §2); DEFENSE=0; NORMAL 不打折."""
        assert position_cap("DEFENSE") == 0.0
        assert position_cap("RECOVERY") == RECOVERY_POSITION_CAP
        assert position_cap("RECOVERY", base_cap=0.75) == pytest.approx(0.30)  # B档→30%
        assert position_cap("NORMAL", base_cap=0.75) == pytest.approx(0.75)

    def test_cap_window_consistency(self):
        """仓位上限与买入窗口同口径: cap=0 ⟺ 窗口全关."""
        for t in ("09:50", "11:00", "14:40"):
            assert buy_window_open(t, bear_state="DEFENSE") == (
                position_cap("DEFENSE") > 0
            )


# ============================================================
# #5/#5a/#7/#8 规则口径核对 (对照 V5.1 §3/§4 正文)
# ============================================================
def _buy_ctx(**kw):
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


def _sell_pos(**kw):
    base = {
        "symbol": "600519",
        "total_qty": 100,
        "sellable_qty": 100,
        "entry_price": 100.0,
        "entry_date": "2026-07-23",
        "hold_days": 2,
        "stop_price": 95.0,
        "max_price_since_entry": 100.0,
    }
    return Position(**{**base, **kw})


class TestRuleCalibration:
    def test_b7_no_stop_price_means_no_protection(self):
        """#5a: stop_price 未下发 (≤0) → B7 否决 (无止损价 = 无保护)."""
        r = buy_trigger(_buy_ctx(stop_price=0.0), ())
        assert "B7" in r["vetoes"]

    def test_b5_uses_min_of_dual_inputs(self):
        """#5: B5 毛利 = min(pred_q50, 事件研究均值) × (1-衰减) — 取小原则."""
        # event_mean 更小: 用 event_mean → 净收益不足否决
        r = buy_trigger(_buy_ctx(pred_q50=0.05, event_mean=0.001), ())
        assert "B5" in r["vetoes"]
        # pred_q50 更小: 用 pred_q50 → 同样否决
        r = buy_trigger(_buy_ctx(pred_q50=0.001, event_mean=0.05), ())
        assert "B5" in r["vetoes"]

    def test_b5_oos_decay_discount(self):
        """#5: 样本外衰减折扣生效 (同一毛利, 高衰减 → 否决)."""
        r = buy_trigger(_buy_ctx(pred_q50=0.05, event_mean=0.05, oos_decay=0.95), ())
        assert "B5" in r["vetoes"]

    def test_s1_no_global_fixed_stop(self):
        """#7: S1 stop_price 下发制 — stop_price=0 时跌破 -4% 也不触发."""
        pos = _sell_pos(stop_price=0.0, hold_days=1)
        ctx = SellContext(t="11:00", price=95.5, limit_down_price=80.0)  # -4.5%
        r = sell_trigger(ctx, pos)
        assert r["rule"] != "S1", "存在全局固定止损 (违反 stop_price 下发制)"

    def test_s2_requires_3pct_profit_to_activate(self):
        """#8: S2 浮盈 <3% 不激活 (最高价 102 → 浮盈 2%, 回撤也不触发)."""
        pos = _sell_pos(stop_price=50.0, max_price_since_entry=102.0, hold_days=1)
        ctx = SellContext(t="11:00", price=98.0, limit_down_price=80.0, atr_pct=0.02)
        r = sell_trigger(ctx, pos)
        assert r["rule"] != "S2", "浮盈不足 3% 即激活移动止盈"


# ============================================================
# #10 S8 跌停逃生协议
# ============================================================
class TestS8EscapeProtocol:
    def test_s8_beats_all_other_rules(self):
        """S8 最高优先级: S3/S6/S1 同时成立也必须走 AUCTION_SELL."""
        pos = _sell_pos(stop_price=95.0, max_price_since_entry=110.0)
        ctx = SellContext(
            t="14:40",
            price=90.0,
            limit_down_price=90.0,  # 跌停
            limit_up_price=110.0,
            touched_limit_up=True,  # S6 也成立
            turnover_pct=0.50,  # S3 也成立
            change_pct=0.09,
        )
        r = sell_trigger(ctx, pos)
        assert r["action"] == "AUCTION_SELL" and r["rule"] == "S8"
        assert r["qty"] == pos.sellable_qty

    def test_s8_only_at_limit_down(self):
        """价格未触及跌停价 → 不触发 S8 (哪怕只差一档)."""
        pos = _sell_pos()
        ctx = SellContext(t="14:40", price=90.01, limit_down_price=90.0)
        r = sell_trigger(ctx, pos)
        assert r["rule"] != "S8"

    def test_s8_respects_t1_lock(self):
        """T+1 锁死优先于 S8: sellable=0 时整条链跳过 (物理约束)."""
        pos = _sell_pos(sellable_qty=0)
        ctx = SellContext(t="14:40", price=90.0, limit_down_price=90.0)
        r = sell_trigger(ctx, pos)
        assert r["action"] == "HOLD" and "T+1" in r["reason"]

    def test_backtest_auction_queue_keeps_position(self):
        """回测口径: 跌停日排队 = 记录 auction_queue, 持仓与现金不变 (未成交)."""
        pos = Position("600519", 1000, 0, 100.0, "2026-07-24", stop_price=99.0)
        bars = tuple(
            Bar(t=f"09:{30 + i:02d}", close=90.0, volume=1e6, amount=9e7, vwap=90.0)
            for i in range(5)
        )
        dc = DayContext(
            symbol="600519",
            date="2026-07-25",
            bars=bars,
            pre_close=100.0,
            pred_q50=0.0,
            atr_pct=0.02,
            stop_price=99.0,
            position_weight=0.0,
            adv_20d=1e9,
            limit_down_price=90.0,
        )
        bt = IntradayBacktester(capital=100_000)
        cash_before = bt.cash
        r = bt.run_day(dc, pos=pos)
        queued = [t for t in r.trades if t["side"] == "auction_queue"]
        assert queued and queued[0]["rule"] == "S8"
        assert bt.cash == cash_before, "排队未成交不得变动现金"
        assert r.end_position is not None
        assert r.end_position.total_qty == 1000, "排队未成交不得减仓"
