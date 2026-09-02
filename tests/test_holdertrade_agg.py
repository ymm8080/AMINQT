"""非交易日公告丢失修复 — holdertrade_agg.select_unwritten_agg 过滤语义.

002881 2025-11-15 (周六) 增持事件: _daily_fetch 旧过滤 date == TRADE_DATE
对非交易日 ann_date 的聚合行永假, 整条丢失 (raw cache 有, 面板无).
"""

from __future__ import annotations

import pandas as pd

from app.pipeline1.holdertrade_agg import select_unwritten_agg

FRI = "2025-11-14"
SAT = "2025-11-15"
SUN = "2025-11-16"
MON = "2025-11-17"
TUE = "2025-11-18"
WEEKEND_SPAN = [FRI, SAT, SUN, MON]


def _agg(dates):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "symbol": ["000002"] * len(dates),
            "sh_change_amt_total": [100.0] * len(dates),
        }
    )


def _days(out):
    return list(out["date"].dt.strftime("%Y-%m-%d"))


def test_weekend_events_kept_on_next_trading_day():
    # 周一 fetch (prev=周五): 周六/周日/周一公告都落周一的行; 周五已写过, 剔除
    out = select_unwritten_agg(_agg(WEEKEND_SPAN), prev_trade_date=FRI, trade_date=MON)
    assert _days(out) == [SAT, SUN, MON]


def test_no_double_count_on_following_day():
    # 周二 fetch (prev=周一): 周末+周一事件周一已写过, 只保留周二新公告
    out = select_unwritten_agg(
        _agg(WEEKEND_SPAN + [TUE]), prev_trade_date=MON, trade_date=TUE
    )
    assert _days(out) == [TUE]


def test_empty_panel_falls_back_to_today_only():
    # 空面板 (prev=NaT): 退回旧语义, 仅保留当日
    out = select_unwritten_agg(
        _agg(WEEKEND_SPAN), prev_trade_date=pd.NaT, trade_date=MON
    )
    assert _days(out) == [MON]


def test_empty_agg_passthrough():
    out = select_unwritten_agg(_agg([]), prev_trade_date=MON, trade_date=TUE)
    assert out.empty
