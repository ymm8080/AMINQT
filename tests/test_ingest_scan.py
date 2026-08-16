"""V3 入库扫描 (ingest gate) 单元测试 — app/pipeline1/ingest_scan.py.

规则 (2026-08-03): 当日追加前剔 ST/*ST 股 和 上市 < 150 个交易日的新股.
上市天数 = 面板交易日历 (trade_cal) 中 [list_date, trade_date] 的交易日计数.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.ingest_scan import apply_ingest_scan, build_universe

TRADE_DATE = "20260803"
_BASE = pd.Timestamp(TRADE_DATE)
# 模拟交易日历: 2025-01-01 → 20260803 的工作日 (交易日, 忽略节假日)
_CAL = pd.DatetimeIndex(pd.bdate_range("2025-01-01", _BASE)).sort_values()


def _stock_info(specs: dict) -> pd.DataFrame:
    """{symbol: (name, list_date)} → stock_info DataFrame (index=symbol)."""
    df = pd.DataFrame(
        [{"symbol": s, "name": n, "list_date": ld} for s, (n, ld) in specs.items()]
    ).set_index("symbol")
    return df


def _frame(symbols) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": list(symbols), "close": np.arange(len(symbols)) + 1.0}
    )


def test_keeps_old_non_st():
    info = _stock_info({"600519": ("贵州茅台", "20010827")})
    out, dropped = apply_ingest_scan(_frame(["600519"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 0
    assert list(out["symbol"]) == ["600519"]


def test_drops_st_by_name():
    info = _stock_info(
        {"600001": ("ST测试", "20010101"), "600519": ("贵州茅台", "20010827")}
    )
    out, dropped = apply_ingest_scan(
        _frame(["600001", "600519"]), info, TRADE_DATE, 150, _CAL
    )
    assert dropped == 1
    assert list(out["symbol"]) == ["600519"]


def test_drops_star_st():
    info = _stock_info({"600002": ("*ST退市风险", "20010101")})
    out, dropped = apply_ingest_scan(_frame(["600002"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 1
    assert out.empty


def test_drops_young_stock():
    # 上市 149 个交易日 (age 149 < 150) → 剔除
    ld_young = _CAL[-149].strftime("%Y%m%d")
    info = _stock_info({"002001": ("次新股", ld_young)})
    out, dropped = apply_ingest_scan(_frame(["002001"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 1
    assert out.empty


def test_boundary_150_kept_149_dropped():
    ld_150 = _CAL[-150].strftime("%Y%m%d")  # age 150 → 保留
    ld_149 = _CAL[-149].strftime("%Y%m%d")  # age 149 → 剔除
    info = _stock_info({"A150": ("正好150天", ld_150), "B149": ("差1天", ld_149)})
    out, dropped = apply_ingest_scan(
        _frame(["A150", "B149"]), info, TRADE_DATE, 150, _CAL
    )
    assert dropped == 1
    assert list(out["symbol"]) == ["A150"]


def test_old_stock_before_calendar_kept():
    # list_date 早于交易日历起点 → searchsorted left=0 → 天数=整段日历长 (>150) → 保留
    info = _stock_info({"600519": ("贵州茅台", "20010827")})
    out, dropped = apply_ingest_scan(_frame(["600519"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 0
    assert list(out["symbol"]) == ["600519"]


def test_missing_list_date_dropped_conservative():
    # NaT → searchsorted 置于末尾 → 天数 0 → 剔除
    info = _stock_info({"600999": ("无上市日期", np.nan)})
    out, dropped = apply_ingest_scan(_frame(["600999"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 1
    assert out.empty


def test_empty_stock_info_is_noop():
    info = pd.DataFrame(columns=["name", "list_date"])
    out, dropped = apply_ingest_scan(_frame(["600519"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 0
    assert list(out["symbol"]) == ["600519"]


def test_missing_trade_cal_raises():
    info = _stock_info({"600519": ("贵州茅台", "20010827")})
    try:
        apply_ingest_scan(_frame(["600519"]), info, TRADE_DATE, 150, None)
    except ValueError as e:
        assert "trade_cal" in str(e)
    else:
        raise AssertionError("expected ValueError when trade_cal missing")


# ── build_universe (2026-08-16 宇宙解冻) ──

_BASIC = pd.DataFrame(
    {"symbol": ["600001", "600002", "300001"]}, index=["600001", "600002", "300001"]
)


def _panel_dates(pairs):
    """[(symbol, date_str)] → 面板 symbol×date 子集."""
    return pd.DataFrame(
        {
            "symbol": [s for s, _ in pairs],
            "date": [pd.Timestamp(d) for _, d in pairs],
        }
    )


def test_build_universe_full_market_not_frozen():
    # 解冻核心: 面板最新日没有的新股 (600002) 也进宇宙 — 旧口径宇宙冻结会漏掉它.
    panel = _panel_dates(
        [("600001", "2026-08-14"), ("300001", "2026-08-14")] * 2
    )
    universe, kept = build_universe(_BASIC, panel, "20260817")
    assert universe == {"600001", "600002", "300001"}
    assert kept == 0


def test_build_universe_new_day_excludes_delisted():
    # 新交易日 (trade_date > 面板最新日): 宇宙 = stock_basic L 全量,
    # 面板历史里已退市/暂停上市的 symbol 不并入 (当日无其新数据, 并入无意义).
    panel = _panel_dates(
        [
            ("600001", "2026-08-14"),
            ("000999", "2026-08-14"),  # 不在 stock_basic L (已退市)
        ]
    )
    universe, kept = build_universe(_BASIC, panel, "20260817")
    assert universe == {"600001", "600002", "300001"}
    assert kept == 0


def test_build_universe_replace_history_keeps_delisted():
    # 替换历史日 (trade_date <= 面板最新日): 该日已存在但不在 stock_basic 的
    # symbol (退市/暂停上市) 必须并入, 防止其历史行被整行丢弃.
    panel = _panel_dates(
        [
            ("600001", "2026-07-30"),
            ("000999", "2026-07-30"),
            ("600001", "2026-08-14"),
        ]
    )
    universe, kept = build_universe(_BASIC, panel, "20260730")
    assert universe == {"600001", "600002", "300001", "000999"}
    assert kept == 1


def test_build_universe_replace_history_kept_zero_when_all_listed():
    # 替换历史日且该日 symbol 全在 stock_basic → kept=0, 宇宙无额外并入.
    panel = _panel_dates(
        [("600001", "2026-07-30"), ("600001", "2026-08-14")]
    )
    universe, kept = build_universe(_BASIC, panel, "20260730")
    assert universe == {"600001", "600002", "300001"}
    assert kept == 0


def test_build_universe_empty_stock_basic_returns_empty():
    # stock_basic 拉取失败 (空) → 宇宙空集, 调用方 (_daily_fetch) 应 FATAL,
    # 防止宇宙为空时当日行被全部静默丢弃 (面板再冻结).
    panel = _panel_dates([("600001", "2026-08-14")])
    universe, kept = build_universe(pd.DataFrame(), panel, "20260817")
    assert universe == set()
    assert kept == 0
