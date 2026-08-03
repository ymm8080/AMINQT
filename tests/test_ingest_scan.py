"""V3 入库扫描 (ingest gate) 单元测试 — app/pipeline1/ingest_scan.py.

规则 (2026-08-03): 当日追加前剔 ST/*ST 股 和 上市 < 150 个交易日的新股.
上市天数 = 面板交易日历 (trade_cal) 中 [list_date, trade_date] 的交易日计数.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.ingest_scan import apply_ingest_scan

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
    return pd.DataFrame({"symbol": list(symbols), "close": np.arange(len(symbols)) + 1.0})


def test_keeps_old_non_st():
    info = _stock_info({"600519": ("贵州茅台", "20010827")})
    out, dropped = apply_ingest_scan(_frame(["600519"]), info, TRADE_DATE, 150, _CAL)
    assert dropped == 0
    assert list(out["symbol"]) == ["600519"]


def test_drops_st_by_name():
    info = _stock_info(
        {"600001": ("ST测试", "20010101"), "600519": ("贵州茅台", "20010827")}
    )
    out, dropped = apply_ingest_scan(_frame(["600001", "600519"]), info, TRADE_DATE, 150, _CAL)
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
    out, dropped = apply_ingest_scan(_frame(["A150", "B149"]), info, TRADE_DATE, 150, _CAL)
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
