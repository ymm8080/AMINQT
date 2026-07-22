# -*- coding: utf-8 -*-
"""DualLayerBuyDetector 单元测试 (P10.10, ARCH §5.15)."""

import pandas as pd
import pytest

from app.core.dual_layer_buy_detector import DualLayerBuyDetector

COLS = {
    "pullup": "tech_ths_pullup_flag_decay10",
    "ctrl": "tech_ths_ctrl_ratio",
    "short": "tech_ths_trend_short",
    "mid": "tech_ths_trend_mid",
    "ctrl_low": "tech_ths_ctrl_low",
    "flow": "tech_ths_flow_net",
}


@pytest.fixture
def detector():
    return DualLayerBuyDetector()


def _daily_df(
    n=30,
    pullup_peak_idx=10,
    ctrl_last=0.35,
    trend_violation=False,
    distance_not_min=False,
):
    """构造满足四条件的日线数据 (可通过参数制造失败)."""
    pullup = [0.0] * n
    if pullup_peak_idx is not None:
        pullup[pullup_peak_idx] = 1.0
    short, mid = [], []
    for i in range(n):
        mid.append(100.0)
        if pullup_peak_idx is not None and i >= pullup_peak_idx:
            # 峰值后: 110 → 100.5 线性收敛, 距离最小值在最后一天
            short.append(110.0 - (i - pullup_peak_idx) * 0.5)
        else:
            short.append(95.0)  # 峰前距离 5 (> 末端 0.5)
    if trend_violation:
        short[-3] = 90.0  # 峰后某日红线跌破蓝线
    if distance_not_min:
        short[-5] = 100.1  # 制造更早的最小距离 0.1 < 末端 0.5
    ctrl = [0.2] * (n - 1) + [ctrl_last]
    return pd.DataFrame(
        {
            COLS["pullup"]: pullup,
            COLS["ctrl"]: ctrl,
            COLS["short"]: short,
            COLS["mid"]: mid,
            COLS["ctrl_low"]: [55.0] * (n - 1) + [60.0],
            COLS["flow"]: [100.0] * n,
        }
    )


# ── Layer 1: 日线买点 (四条件 AND) ──────────────────────────────


class TestDailyBuyPoint:
    def test_all_conditions_pass(self, detector):
        df = _daily_df()
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is True
        assert result["failed_conditions"] == []
        assert result["condition_4_trend"] is True

    def test_not_in_pool(self, detector):
        df = _daily_df()
        result = detector.check_daily_buy_point("600000", ["000001"], df)
        assert result["is_daily_buy"] is False
        assert "condition_1_pool" in result["failed_conditions"]

    def test_no_pullup_peak(self, detector):
        df = _daily_df(pullup_peak_idx=None)
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is False
        assert "condition_2_pullup_peak" in result["failed_conditions"]

    def test_low_ctrl_ratio(self, detector):
        df = _daily_df(ctrl_last=0.20)
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is False
        assert "condition_3_ctrl_ratio" in result["failed_conditions"]

    def test_trend_violation_after_peak(self, detector):
        df = _daily_df(trend_violation=True)
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is False
        assert "condition_4_trend" in result["failed_conditions"]

    def test_distance_not_min(self, detector):
        df = _daily_df(distance_not_min=True)
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is False
        assert "condition_4_trend" in result["failed_conditions"]

    def test_missing_columns(self, detector):
        df = pd.DataFrame({"close": [10.0] * 30})
        result = detector.check_daily_buy_point("600000", ["600000"], df)
        assert result["is_daily_buy"] is False


# ── Layer 2: 日内买点 (六条件 AND) ──────────────────────────────


def _intraday_ok(detector, **overrides):
    kwargs = dict(
        is_daily_buy_marked=True,
        current_time="10:30",
        advancing_stocks=3000,
        total_stocks=5000,
        flow_net=100.0,
        ctrl_ratio=0.35,
        ctrl_low_today=60.0,
        ctrl_low_yesterday=55.0,
    )
    kwargs.update(overrides)
    return detector.check_intraday_buy_point(**kwargs)


class TestIntradayBuyPoint:
    def test_all_six_pass(self, detector):
        result = _intraday_ok(detector)
        assert result["is_intraday_buy"] is True
        assert result["failed_conditions"] == []

    def test_daily_not_marked(self, detector):
        result = _intraday_ok(detector, is_daily_buy_marked=False)
        assert result["is_intraday_buy"] is False
        assert "condition_1_daily_mark" in result["failed_conditions"]

    def test_after_deadline(self, detector):
        result = _intraday_ok(detector, current_time="10:41")
        assert result["is_intraday_buy"] is False
        assert "condition_2_time" in result["failed_conditions"]

    def test_breadth_too_low(self, detector):
        result = _intraday_ok(detector, advancing_stocks=2500)
        assert result["is_intraday_buy"] is False
        assert "condition_3_breadth" in result["failed_conditions"]

    def test_flow_net_negative(self, detector):
        result = _intraday_ok(detector, flow_net=-1.0)
        assert result["is_intraday_buy"] is False
        assert "condition_4_flow_net" in result["failed_conditions"]

    def test_ctrl_ratio_low(self, detector):
        result = _intraday_ok(detector, ctrl_ratio=0.25)
        assert result["is_intraday_buy"] is False
        assert "condition_5_ctrl_ratio" in result["failed_conditions"]

    def test_ctrl_low_not_rising(self, detector):
        result = _intraday_ok(detector, ctrl_low_today=54.0, ctrl_low_yesterday=55.0)
        assert result["is_intraday_buy"] is False
        assert "condition_6_ctrl_low" in result["failed_conditions"]

    def test_red_bar_absolute_50_threshold(self, detector):
        """红柱过半 = 绝对阈值 50 (ctrl_low 量纲 0~100).

        ctrl_low 49 > 昨日 40 (红柱升高) 但 < 50 (未过半) → 条件6不通过。
        """
        result = _intraday_ok(detector, ctrl_low_today=49.0, ctrl_low_yesterday=40.0)
        assert result["condition_6_ctrl_low"] is False
        assert result["is_intraday_buy"] is False

    def test_red_bar_exactly_50_fails(self, detector):
        # 50 不算过半 (> 50 才通过)
        result = _intraday_ok(detector, ctrl_low_today=50.0, ctrl_low_yesterday=40.0)
        assert result["condition_6_ctrl_low"] is False


# ── Layer 3: 开盘10分钟买入时机 (互斥场景) ──────────────────────


def _bars(closes, opens=None, lows=None):
    len(closes)
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    if lows is None:
        lows = [min(o, c) - 0.05 for o, c in zip(opens, closes)]
    return pd.DataFrame({"open": opens, "close": closes, "low": lows})


class TestOpening10min:
    def test_scenario_a_rebound_confirmed(self, detector):
        # 开盘 10 → 下行至 9.0 → 回升至 9.4
        df = _bars([10.0, 9.5, 9.0, 9.2, 9.4], opens=[10.0] * 5)
        result = detector.check_opening_10min_buy_timing(df)
        assert result["scenario"] == "A"
        assert result["buy_timing_confirmed"] is True

    def test_scenario_a_trough_at_end_pending(self, detector):
        # 最低点就是最后一根 → 尚未确认回升
        df = _bars([10.0, 9.5, 9.0], opens=[10.0] * 3)
        result = detector.check_opening_10min_buy_timing(df)
        assert result["scenario"] == "A"
        assert result["buy_timing_confirmed"] is False

    def test_scenario_b_second_peak_higher_confirmed(self, detector):
        # 上行: 峰1=10.5, 峰2=10.8 > 峰1, 第二低点 10.4 后回升至 10.6
        df = _bars(
            [10.0, 10.5, 10.2, 10.8, 10.4, 10.6],
            opens=[10.0] * 6,
            lows=[9.95, 10.0, 10.1, 10.2, 10.3, 10.35],
        )
        result = detector.check_opening_10min_buy_timing(df)
        assert result["scenario"] == "B"
        assert result["buy_timing_confirmed"] is True

    def test_scenario_b_second_peak_lower_rejected(self, detector):
        # 上行: 峰1=10.8 > 峰2=10.5 → 不满足第二峰更高
        df = _bars(
            [10.0, 10.8, 10.2, 10.5, 10.4, 10.3],
            opens=[10.0] * 6,
            lows=[9.95, 10.0, 10.1, 10.2, 10.3, 10.2],
        )
        result = detector.check_opening_10min_buy_timing(df)
        assert result["scenario"] == "B"
        assert result["buy_timing_confirmed"] is False

    def test_scenario_b_no_second_trough_pending(self, detector):
        # 两个峰但第二峰后无低点
        df = _bars(
            [10.0, 10.5, 10.2, 10.8, 10.9],
            opens=[10.0] * 5,
            lows=[9.95, 10.0, 10.1, 10.2, 10.75],
        )
        result = detector.check_opening_10min_buy_timing(df)
        assert result["scenario"] == "B"
        assert result["buy_timing_confirmed"] is False

    def test_insufficient_bars(self, detector):
        result = detector.check_opening_10min_buy_timing(_bars([10.0, 10.1]))
        assert result["buy_timing_confirmed"] is False


# ── detect() 三层综合: 必须全部通过 ─────────────────────────────


class TestDetect:
    def _intraday_confirmed(self):
        return _bars([10.0, 9.5, 9.0, 9.2, 9.4], opens=[10.0] * 5)

    def test_all_layers_pass(self, detector):
        df = _daily_df()
        ctx = {"current_time": "10:00", "advancing_stocks": 3000, "total_stocks": 5000}
        result = detector.detect(
            "600000", ["600000"], df, self._intraday_confirmed(), ctx
        )
        assert result["final_buy_signal"] is True
        assert result["failed_layer"] is None

    def test_layer1_failure(self, detector):
        df = _daily_df()
        result = detector.detect("600000", ["000001"], df, self._intraday_confirmed())
        assert result["final_buy_signal"] is False
        assert result["failed_layer"] == 1

    def test_layer2_failure(self, detector):
        df = _daily_df()
        ctx = {
            "current_time": "10:00",
            "advancing_stocks": 1000,
            "total_stocks": 5000,
        }  # 广度 0.2 < 0.6
        result = detector.detect(
            "600000", ["600000"], df, self._intraday_confirmed(), ctx
        )
        assert result["final_buy_signal"] is False
        assert result["failed_layer"] == 2

    def test_layer3_failure(self, detector):
        df = _daily_df()
        ctx = {"current_time": "10:00", "advancing_stocks": 3000, "total_stocks": 5000}
        pending = _bars([10.0, 9.5, 9.0], opens=[10.0] * 3)  # 低点在末端
        result = detector.detect("600000", ["600000"], df, pending, ctx)
        assert result["final_buy_signal"] is False
        assert result["failed_layer"] == 3
