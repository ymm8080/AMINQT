# -*- coding: utf-8 -*-
"""DualLayerSellDetector 单元测试 (P10.9, ARCH §5.14)."""

import pandas as pd
import pytest

from app.core.dual_layer_sell_detector import DualLayerSellDetector


@pytest.fixture
def detector():
    return DualLayerSellDetector()


def _daily(closes):
    return pd.DataFrame({"close": closes})


def _intraday(closes, opens=None):
    len(closes)
    if opens is None:
        opens = [closes[0]] + closes[:-1]
    return pd.DataFrame({"open": opens, "close": closes})


# ── Layer 1: 日线收盘跌破 (两点比较) ─────────────────────────────


class TestDailyCloseBreak:
    def test_break_triggered(self, detector):
        # 今日 9 < 4日前 10 → 触发
        assert detector.check_daily_close_break(_daily([10, 11, 12, 13, 9])) is True

    def test_no_break_when_higher(self, detector):
        assert detector.check_daily_close_break(_daily([10, 11, 12, 13, 12])) is False

    def test_two_point_compare_middle_dip_irrelevant(self, detector):
        # 中间几日暴跌不影响: 只要今日 < 4日前即触发
        assert detector.check_daily_close_break(_daily([10, 5, 5, 5, 9])) is True

    def test_two_point_compare_middle_surge_irrelevant(self, detector):
        # 中间几日暴涨也不影响: 今日 12 > 4日前 10 → 不触发
        assert detector.check_daily_close_break(_daily([10, 15, 15, 15, 12])) is False

    def test_insufficient_data(self, detector):
        assert detector.check_daily_close_break(_daily([10, 11])) is False


# ── 场景 A: 三峰连续下降 ────────────────────────────────────────


class TestThreePeaksDecline:
    def test_three_declining_peaks(self, detector):
        # 局部峰: idx1=12, idx3=11.5, idx5=11 严格递减
        df = _intraday([10, 12, 11, 11.5, 10.5, 11, 10, 9])
        result = detector.check_three_peaks_decline(df)
        assert result["is_signal"] is True
        assert result["sell_after_peak"] == 3
        assert len(result["peaks"]) >= 3

    def test_ascending_peaks_no_signal(self, detector):
        # 峰: 11, 12, 13 递增 → 无信号
        df = _intraday([10, 11, 10.5, 12, 11, 13, 12, 11])
        assert detector.check_three_peaks_decline(df)["is_signal"] is False

    def test_only_two_peaks_no_signal(self, detector):
        df = _intraday([10, 12, 11, 11.5, 10, 9])
        assert detector.check_three_peaks_decline(df)["is_signal"] is False

    def test_insufficient_bars(self, detector):
        assert (
            detector.check_three_peaks_decline(_intraday([10, 11, 10]))["is_signal"]
            is False
        )


# ── 场景 B: 日内急跌 ────────────────────────────────────────────


class TestIntradayCrash:
    def test_crash_triggered(self, detector):
        assert detector.check_intraday_crash(-0.05) is True

    def test_exact_threshold(self, detector):
        assert detector.check_intraday_crash(-0.04) is True

    def test_small_drop_no_trigger(self, detector):
        assert detector.check_intraday_crash(-0.02) is False

    def test_positive_pct_no_trigger(self, detector):
        assert detector.check_intraday_crash(0.03) is False


# ── detect() 双层综合 ───────────────────────────────────────────


class TestDetect:
    def test_crash_priority_over_peaks(self, detector):
        # 日线跌破 (9 < 10), 前收 13; 日内 12.3 → 跌 5.4% → 场景 B
        daily = _daily([10, 11, 12, 13, 9])
        intraday = _intraday([12.8, 12.5, 12.3])
        result = detector.detect(daily, intraday)
        assert result["daily_sell_marked"] is True
        assert result["intraday_signal"] is True
        assert result["scenario"] == "B_crash"
        assert result["action"] == "sell"

    def test_three_peaks_scenario(self, detector):
        # 日线跌破; 日内相对前收仅微跌 (不触发急跌) → 走三峰
        daily = _daily([10, 11, 12, 13, 9])
        intraday = _intraday([12.6, 13.0, 12.7, 12.95, 12.75, 12.9, 12.8, 12.85])
        result = detector.detect(daily, intraday)
        assert result["daily_sell_marked"] is True
        assert result["intraday_signal"] is True
        assert result["scenario"] == "A_three_peaks"
        assert result["action"] == "sell"

    def test_no_daily_mark_no_sell(self, detector):
        # 日线未跌破 → 即使日内急跌也不卖
        daily = _daily([10, 11, 12, 13, 14])
        intraday = _intraday([13.0, 12.5, 12.0])
        result = detector.detect(daily, intraday)
        assert result["daily_sell_marked"] is False
        assert result["intraday_signal"] is False
        assert result["action"] == "hold"

    def test_daily_marked_but_no_intraday_signal(self, detector):
        # 日线跌破, 日内平稳 → 持有等待
        daily = _daily([10, 11, 12, 13, 9])
        intraday = _intraday([12.9, 13.0, 12.95, 13.0, 12.98])
        result = detector.detect(daily, intraday)
        assert result["daily_sell_marked"] is True
        assert result["intraday_signal"] is False
        assert result["action"] == "hold"
