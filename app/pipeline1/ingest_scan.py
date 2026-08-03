# -*- coding: utf-8 -*-
"""V3 入库扫描 (ingest gate): 剔 ST/*ST 股 和 上市不足 N 天的新股.

_daily_fetch.py 在追加当日行前调用 apply_ingest_scan, 使 ST 股与次新股
不进入 V3 面板 — universe 在入口处收敛, 而非靠面板列 (is_st/list_days)
在训练/回测阶段反复过滤.
"""

import pandas as pd

from app.core.universe_manager import name_is_st


def apply_ingest_scan(df, stock_info, trade_date, min_list_days, trade_cal):
    """剔 ST/*ST 股 和 上市 < min_list_days 个交易日的新股.

    Args:
        df: 当日 DataFrame, 需含 ``symbol`` 列.
        stock_info: DataFrame, index=symbol, 含 ``name`` / ``list_date`` 列
            (list_date 为 Tushare "YYYYMMDD" 字符串或可解析日期). 为空则跳过扫描.
        trade_date: 交易日 ("YYYYMMDD" 字符串或 Timestamp).
        min_list_days: 最小上市交易日数. list_date 缺失 → 保守剔除 (天数计 0).
        trade_cal: 排序去重后的交易日序列 (如面板唯一 date 列). 上市天数按该
            日历计交易日数 (searchsorted 向量化); 缺失且 stock_info 非空时抛错,
            避免静默回退日历日导致次新股漏筛.

    Returns:
        (filtered_df, dropped_count)
    """
    if stock_info is None or len(stock_info) == 0:
        return df, 0
    if trade_cal is None or len(trade_cal) == 0:
        raise ValueError(
            "trade_cal required when stock_info is non-empty (trading-day age)"
        )
    names = df["symbol"].map(stock_info["name"]).fillna("")
    is_st = names.map(name_is_st)
    ld = pd.to_datetime(
        df["symbol"].map(stock_info["list_date"]), format="%Y%m%d", errors="coerce"
    )
    dts = pd.DatetimeIndex(trade_cal).sort_values()
    # 交易日数 = 首次 >= list_date 的交易日 到 trade_date (含) 的计数.
    # NaT list_date → searchsorted 置于末尾 → 天数为 0 → 保守剔除.
    left = dts.searchsorted(ld, side="left")
    right = dts.searchsorted(pd.Timestamp(trade_date), side="right")
    list_days = right - left
    keep = (~is_st) & (list_days >= min_list_days)
    return df[keep], int((~keep).sum())
