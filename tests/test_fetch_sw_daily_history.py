"""tests for scripts/fetch_sw_daily_history 增量缺口判定 (2026-09-02).

防回归: sw_history 挂入每日自动化链后 (2026-09-02 冻结事故), 周末/节假日/当日
重复运行时现有面板已是最新 — 原逻辑会走到 "No data fetched!" exit 1, 假日误报.
约定: 无缺口 (max_date >= 今天) 静默 exit 0; 真缺口但 fetch 全空仍大声 exit 1.
"""

import scripts.fetch_sw_daily_history as fsw


def test_no_gap_when_max_equals_today():
    """当日已补到今天 (如链内重复触发) → 无缺口, 不得误报 exit 1."""
    assert fsw.incremental_needs_fetch("20260902", "20260902") is False


def test_no_gap_when_max_after_today():
    """max_date 晚于今天 (系统时钟偏差/补跑旧场景) → 也不算缺口."""
    assert fsw.incremental_needs_fetch("20260903", "20260902") is False


def test_gap_when_max_before_today():
    """冻结@07-31 事故场景: max_date 落后今天 → 有缺口, 必须拉取."""
    assert fsw.incremental_needs_fetch("20260731", "20260902") is True


def test_accepts_int_max_date():
    """面板 trade_date 若未来改整型存储, 判定不得崩 (str() 归一化兜底)."""
    assert fsw.incremental_needs_fetch(20260731, "20260902") is True
    assert fsw.incremental_needs_fetch(20260902, "20260902") is False
