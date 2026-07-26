"""P20 V5.1 批2: 决策链回测 / 事件研究 / 平原寻优 / 5min数据层."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.intraday.v51.backtest_engine import DayContext, IntradayBacktester
from app.intraday.v51.buy_engine import Bar
from app.intraday.v51.data_5min import IntradayDataLoader, normalize_5min
from app.intraday.v51.event_study import event_study
from app.intraday.v51.param_optimizer import (
    grid_search,
    oos_decay_check,
    plateau_check,
    three_gate_verdict,
    walk_forward,
)
from app.intraday.v51.position import Position


# ============================================================
# 决策链回测 (T+1 + 满仓锁死 + 扣费)
# ============================================================
def _day_ctx(bars, **kw):
    base = {
        "symbol": "600519",
        "date": "2026-07-25",
        "bars": tuple(bars),
        "pre_close": 100.0,
        "pred_q50": 0.06,
        "atr_pct": 0.02,
        "stop_price": 94.0,
        "position_weight": 1.0,
        "adv_20d": 1e9,
        "event_mean": 0.06,
        "limit_down_price": 90.0,
        "limit_up_price": 110.0,
    }
    return DayContext(**{**base, **kw})


def _make_bars(closes, start_vol=1e6):
    return [
        Bar(
            t=f"{9 + (i * 5) // 60:02d}:{(30 + (i * 5)) % 60:02d}",
            close=c,
            volume=start_vol,
            amount=c * start_vol,
            vwap=c * 0.998,
        )
        for i, c in enumerate(closes)
    ]


class TestIntradayBacktester:
    def test_t1_lock_blocks_same_day_sell(self):
        """满仓锁死: 当日买入后盘中大跌也只能看着 (铁律 #0, 检查清单 #4)."""
        closes = [100.0] * 20 + [101.0] * 8 + [90.0] * 20  # 尾盘走强后暴跌
        bars = _make_bars(closes)
        # 强制尾盘窗口放量走强触发 B2
        for i in range(len(bars) - 5, len(bars) - 2):
            bars[i] = Bar("14:50", 101.0, 3e6, 3e8, 100.0)
        bt = IntradayBacktester(capital=100_000)
        r = bt.run_day(_day_ctx(bars))
        buys = [t for t in r.trades if t["side"] == "buy"]
        sells = [t for t in r.trades if t["side"] == "sell"]
        if buys:  # 若当日买入 → 当日绝不可能卖出 (T+1 锁死)
            assert sells == [], "T+1 锁死被违反: 当日买入当日卖出"
            assert r.end_position is not None
            assert r.end_position.sellable_qty == 0

    def test_overnight_settle_allows_sell(self):
        """隔夜结算后可卖: stop_price 触发 S1."""
        pos = Position("600519", 900, 0, 100.0, "2026-07-24", stop_price=99.0)
        bars = _make_bars([98.0] * 10)  # 开盘即破止损
        bt = IntradayBacktester(capital=100_000)
        r = bt.run_day(_day_ctx(bars, position_weight=0.0), pos=pos)
        sells = [t for t in r.trades if t["side"] == "sell"]
        assert sells and sells[0]["rule"] == "S1"
        assert r.end_position is None

    def test_costs_deducted(self):
        pos = Position("600519", 1000, 0, 100.0, "2026-07-24", stop_price=99.0)
        bars = _make_bars([98.0] * 10)
        bt = IntradayBacktester(capital=200_000)
        bt.run_day(_day_ctx(bars, position_weight=0.0), pos=pos)
        # 卖出 1000×98 = 98000 毛收入; 扣单边成本 (佣金+印花税+滑点+冲击)
        assert bt.cash < 200_000 + 98_000 - 100  # 成本已扣 (< 毛额)
        assert bt.cash > 200_000 + 97_000  # 成本量级合理 (~0.35%)

    def test_recovery_caps_position_weight(self):
        """RECOVERY 仓位上限 30%: position_weight=1.0 被 cap 到 0.30 (本地强制)."""
        closes = [100.0] * 20 + [101.0] * 8 + [102.0] * 20
        bars = _make_bars(closes)
        # 尾盘窗口放量走强触发 B2
        for i in range(len(bars) - 5, len(bars) - 2):
            bars[i] = Bar("14:50", 102.0, 3e6, 3e8, 100.0)
        # RECOVERY + position_weight=1.0 (攻击档默认, 应被 cap 到 0.30)
        bt = IntradayBacktester(capital=100_000)
        r = bt.run_day(_day_ctx(bars, bear_state="RECOVERY", position_weight=1.0))
        buys = [t for t in r.trades if t["side"] == "buy"]
        if buys:
            # 100% 仓位: qty ≈ 100000/100 = 1000 股
            # 30% 仓位: qty ≈ 30000/100 = 300 股
            assert buys[0]["qty"] <= 400, (
                f"RECOVERY cap 失效: 买入 {buys[0]['qty']} 股 (应 ≤400, 30%上限)"
            )


# ============================================================
# 事件研究 (关卡 0/1)
# ============================================================
class TestEventStudy:
    def test_gates(self):
        rng = np.random.default_rng(1)
        good = rng.normal(0.005, 0.02, 1200)  # n≥1000, mean>0
        r = event_study(good)
        assert r["pass_gate0"] and r["pass"]
        few = rng.normal(0.005, 0.02, 500)
        assert not event_study(few)["pass_gate0"]
        noise = rng.normal(0, 0.02, 2000)
        assert not event_study(noise)["pass_gate1"]  # t 不显著


# ============================================================
# 平原检测 / 样本外衰减 / 三关审判 / Walk-Forward
# ============================================================
class TestParamOptimizer:
    def test_plateau(self):
        assert plateau_check([0.8, 0.9, 1.0, 0.7, 0.85], 1.0)["pass"]
        assert not plateau_check([-0.1, 0.2, -0.3, 1.0], 1.0)["pass"]  # 孤峰
        assert not plateau_check([0.1, 0.1, 0.1], 1.0)["pass"]  # 邻域太弱

    def test_oos_decay(self):
        assert oos_decay_check(1.0, 0.6)["pass"]
        assert not oos_decay_check(1.0, 0.3)["pass"]
        assert not oos_decay_check(0.0, 0.5)["pass"]

    def test_three_gates(self):
        r = three_gate_verdict(3.5, [0.8, 0.9, 0.85], 1.0, 0.7)
        assert r["pass"]
        r = three_gate_verdict(2.0, [0.8, 0.9, 0.85], 1.0, 0.7)
        assert not r["pass"]

    def test_grid_and_walk_forward(self):
        grid = {"x": [1, 2, 3]}
        best = grid_search(grid, lambda p: -abs(p["x"] - 2))
        assert best[0]["params"] == {"x": 2}
        windows = [("w1", None, None), ("w2", None, None)]
        results = walk_forward(
            windows,
            lambda _: {
                "params": {"x": 2},
                "score": 1.0,
                "t_stat": 3.5,
                "neighbor_scores": [0.8, 0.9, 0.85],
            },
            lambda p, _: 0.7,
        )
        assert len(results) == 2 and results[0]["verdict"]["pass"]


# ============================================================
# 5min 数据层
# ============================================================
class TestData5Min:
    def test_normalize(self):
        raw = pd.DataFrame(
            {
                "时间": ["2026-07-25 09:35:00"],
                "开盘": [100.0],
                "收盘": [100.5],
                "最高": [100.6],
                "最低": [99.9],
                "成交量": [1e5],
                "成交额": [1e7],
            }
        )
        out = normalize_5min(raw, "600519", "2026-07-25")
        assert out["t"].iloc[0] == "09:35"
        assert out["symbol"].iloc[0] == "600519"

    def test_cache_and_retry(self, tmp_path):
        calls = {"n": 0}

        def flaky(symbol, date):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("network down")
            return pd.DataFrame(
                {
                    "时间": ["2026-07-25 09:35:00"],
                    "开盘": [1.0],
                    "收盘": [1.0],
                    "最高": [1.0],
                    "最低": [1.0],
                    "成交量": [1.0],
                    "成交额": [1.0],
                }
            )

        import app.intraday.v51.data_5min as d5

        d5.RETRY_SLEEP = 0  # 测试不等待
        loader = IntradayDataLoader(str(tmp_path), fetcher=flaky)
        df = loader.load("600519", "2026-07-25")
        assert calls["n"] == 3 and len(df) == 1
        # 第二次走缓存, 不再拉取
        loader.load("600519", "2026-07-25")
        assert calls["n"] == 3

    def test_retry_exhausted_raises(self, tmp_path):
        def always_fail(symbol, date):
            raise ConnectionError("down")

        import app.intraday.v51.data_5min as d5

        d5.RETRY_SLEEP = 0
        loader = IntradayDataLoader(str(tmp_path), fetcher=always_fail)
        with pytest.raises(RuntimeError):
            loader.load("600519", "2026-07-25")
