# -*- coding: utf-8 -*-
"""missing_trade_dates 单测 — 缺日自愈的纯判定核心 (2026-10-14 国庆停机事故)."""

import pandas as pd

from app.pipeline1.freshness_guard import missing_trade_dates

# 2026 国庆真实日历: 0930 后开市日为 1008/1009/1012/1013/1014
CAL = pd.DatetimeIndex(
    [
        "2026-09-28",
        "2026-09-29",
        "2026-09-30",
        "2026-10-08",
        "2026-10-09",
        "2026-10-12",
        "2026-10-13",
        "2026-10-14",
    ]
)


def test_holiday_gap_returns_interior_trade_days_excluding_through():
    # 停机 8 天后面板停在 0930, 今日盘前取数日=1013: 应补 1008/1009/1012,
    # 1013 本身不算 (主流程随后拉它)
    got = missing_trade_dates("2026-09-30", "2026-10-13", CAL)
    assert got == list(pd.DatetimeIndex(["2026-10-08", "2026-10-09", "2026-10-12"]))


def test_gap_before_evening_target_includes_day_before_target():
    # 停机后首跑若发生在 1014 盘后 (取数日=1014): 补到 1013 为止
    got = missing_trade_dates("2026-09-30", "2026-10-14", CAL)
    assert got == list(
        pd.DatetimeIndex(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"])
    )


def test_no_gap_when_panel_current():
    assert missing_trade_dates("2026-10-13", "2026-10-14", CAL) == []


def test_panel_reahead_returns_empty():
    # 面板领先于请求日 (历史日幂等重跑) → 不补
    assert missing_trade_dates("2026-10-14", "2026-10-08", CAL) == []


def test_same_day_rerun_returns_empty():
    assert missing_trade_dates("2026-10-14", "2026-10-14", CAL) == []


def test_none_calendar_fails_open():
    assert missing_trade_dates("2026-09-30", "2026-10-14", None) == []


def test_empty_calendar_fails_open():
    assert missing_trade_dates("2026-09-30", "2026-10-14", pd.DatetimeIndex([])) == []


def test_non_trade_days_never_appear():
    # 区间含周末 10-10/10-11, 只认开市日历 → 只返回 10-09/10-12
    got = missing_trade_dates("2026-10-08", "2026-10-13", CAL)
    assert got == list(pd.DatetimeIndex(["2026-10-09", "2026-10-12"]))
