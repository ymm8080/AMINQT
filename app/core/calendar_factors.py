# -*- coding: utf-8 -*-
"""日历因子 (P10.6, ARCH §5.11 — 6 维).

交易日历相关: 距节假日天数/周几/月初月末/季末/长假前缩量预期。
数据: data/calendar/holidays.json + trade_dates.csv。

holidays.json 缺失时降级: 以周末作为"节假日"基准 (weekend-only fallback),
``cal_days_to_holiday`` 退化为"距下一个周末的交易日数"。

防未来函数: 全部因子仅依赖当日日期与静态日历表, 不使用任何未来行情。
"""

import json
import logging
import os
import re
from typing import Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CALENDAR_FACTOR_COLUMNS = [
    "cal_days_to_holiday",  # 距下一个节假日交易日数
    "cal_day_of_week",  # 周几 (0-4)
    "cal_is_month_end",  # 是否月末 (最后 3 交易日)
    "cal_is_month_start",  # 是否月初 (前 3 交易日)
    "cal_is_quarter_end",  # 是否季末
    "cal_pre_holiday_flag",  # 长假前 flag (节前 5 日)
]

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# 月末/月初判定的交易日阈值
_MONTH_EDGE_DAYS = 3
# 节前 flag 窗口 (交易日)
_PRE_HOLIDAY_DAYS = 5


def _collect_date_strings(obj, out: Set[pd.Timestamp]) -> None:
    """递归收集 JSON 结构中所有 ``YYYY-MM-DD`` 日期字符串."""
    if isinstance(obj, str):
        for m in _DATE_RE.findall(obj):
            try:
                out.add(pd.Timestamp(m).normalize())
            except ValueError:
                continue
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_date_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_date_strings(v, out)


def load_holidays(holidays_path: str) -> Set[pd.Timestamp]:
    """加载节假日日期集合.

    支持格式: ["2026-01-01", ...] / {"holidays": [...]} / {"2026": [...]} 等
    嵌套结构, 递归提取所有日期字符串。

    Args:
        holidays_path: 节假日 JSON 路径。

    Returns:
        归一化 ``pd.Timestamp`` 集合; 文件缺失或解析失败返回空集合。
    """
    if not os.path.exists(holidays_path):
        logger.warning("节假日文件缺失: %s — 使用 weekend-only 降级", holidays_path)
        return set()
    try:
        with open(holidays_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        holidays: Set[pd.Timestamp] = set()
        _collect_date_strings(data, holidays)
        logger.info("加载节假日 %d 天 (%s)", len(holidays), holidays_path)
        return holidays
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "节假日文件解析失败: %s (%s) — 使用 weekend-only 降级", holidays_path, exc
        )
        return set()


def _next_holiday_after(
    t: pd.Timestamp, holidays: Set[pd.Timestamp], weekend_fallback: bool
):
    """返回 t 之后最近的节假日 (strictly after t); 无则 None.

    weekend_fallback 模式下, 下一个周六视为"节假日"。
    """
    if weekend_fallback:
        d = t + pd.Timedelta(days=1)
        while d.weekday() < 5:  # 5 = Saturday
            d += pd.Timedelta(days=1)
        return d
    cands = [h for h in holidays if h > t]
    return min(cands) if cands else None


def compute_calendar_factors(
    df: pd.DataFrame, holidays_path: str = "data/calendar/holidays.json"
) -> pd.DataFrame:
    """为日线 DataFrame 追加 6 维日历因子.

    Args:
        df: 含 date 列的日线数据。
        holidays_path: 节假日表路径。

    Returns:
        追加 6 列 cal_* 因子的 DataFrame。
    """
    if "date" not in df.columns:
        raise KeyError("日历因子计算需要 'date' 列")

    df = df.copy()
    dates = pd.to_datetime(df["date"]).dt.normalize()
    df["_cal_date"] = dates

    holidays = load_holidays(holidays_path)
    weekend_fallback = len(holidays) == 0

    # ── 1. 周几 (0=Mon ... 4=Fri) ──
    df["cal_day_of_week"] = dates.dt.weekday.astype(float)

    # ── 2/3. 月初 / 月末 (以 df 内交易日排序, 前/后 3 个交易日) ──
    month_key = dates.dt.to_period("M")
    rank_asc = dates.groupby(month_key).rank(method="first", ascending=True)
    rank_desc = dates.groupby(month_key).rank(method="first", ascending=False)
    df["cal_is_month_start"] = (rank_asc <= _MONTH_EDGE_DAYS).astype(float)
    df["cal_is_month_end"] = (rank_desc <= _MONTH_EDGE_DAYS).astype(float)

    # ── 4. 季末 = 季末月 (3/6/9/12) 的月末 ──
    quarter_end_month = dates.dt.month.isin([3, 6, 9, 12])
    df["cal_is_quarter_end"] = (
        quarter_end_month & (df["cal_is_month_end"] > 0)
    ).astype(float)

    # ── 5. 距下一节假日交易日数 ──
    # 对每个交易日 t: 找到 t 之后的最近节假日 h,
    # days_to_holiday = df 内满足 t < d < h 的交易日数。
    # 若数据视野内无节假日, 取剩余交易日数 (上界估计)。
    trade_dates = pd.DatetimeIndex(sorted(dates.unique()))
    pos = {d: i for i, d in enumerate(trade_dates)}
    n = len(trade_dates)

    def _days_to_holiday(t: pd.Timestamp) -> float:
        h = _next_holiday_after(t, holidays, weekend_fallback)
        i = pos[t]
        if h is None:
            return float(n - 1 - i)
        # h 之前最后一个交易日的索引
        j = int(trade_dates.searchsorted(h, side="left")) - 1
        return float(max(j - i, 0))

    df["cal_days_to_holiday"] = dates.map(_days_to_holiday).astype(float)

    # ── 6. 节前 flag (节前 5 个交易日内, 含节前最后一日 = 0) ──
    df["cal_pre_holiday_flag"] = (
        (df["cal_days_to_holiday"] >= 0)
        & (df["cal_days_to_holiday"] <= _PRE_HOLIDAY_DAYS)
    ).astype(float)

    df = df.drop(columns=["_cal_date"])
    df[CALENDAR_FACTOR_COLUMNS] = np.nan_to_num(
        df[CALENDAR_FACTOR_COLUMNS].to_numpy(dtype=float), nan=0.0
    )
    logger.info(
        "日历因子计算完成: %d 行, weekend_fallback=%s", len(df), weekend_fallback
    )
    return df
