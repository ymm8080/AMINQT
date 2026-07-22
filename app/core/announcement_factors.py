# -*- coding: utf-8 -*-
"""公告事件因子 (P10.6, ARCH §5.11 — 5 维).

个股公告事件: 近 N 日公告数/重大公告 flag/业绩公告/增减持/风险警示。
数据: data/calendar/announcements/{symbol}_ann.parquet (列: date, type)。

公告文件缺失时降级: 全部因子置 0 + 警告日志。

防未来函数: 滚动窗口只回看 (rolling/exp_decay 均为 causal)。
"""

import logging
import os

import numpy as np
import pandas as pd

from app.core.ths_indicators import exp_decay_encode

logger = logging.getLogger(__name__)

ANNOUNCEMENT_FACTOR_COLUMNS = [
    "ann_count_5d",            # 近 5 日公告数
    "ann_major_flag",          # 重大公告 flag (衰减编码)
    "ann_earnings_flag",       # 业绩公告 flag
    "ann_hold_change_flag",    # 增减持 flag
    "ann_risk_warning_flag",   # 风险警示 flag (用于基础过滤剔除)
]

# ── 公告类型归类 (小写匹配, 兼容中英文标注) ─────────────────────────
MAJOR_TYPES = {
    "major", "重大事项", "重组", "并购", "risk_warning", "风险警示",
    "suspend", "停牌",
}
EARNINGS_TYPES = {
    "earnings", "业绩", "业绩预告", "业绩快报", "财报", "年报", "半年报",
    "一季报", "三季报", "financial_report",
}
HOLD_CHANGE_TYPES = {
    "hold_change", "增减持", "增持", "减持", "回购",
}
RISK_WARNING_TYPES = {
    "risk_warning", "风险警示", "st", "退市风险",
}

# 衰减/回看参数
_COUNT_WINDOW = 5        # ann_count_5d 窗口 (交易日)
_DECAY_TAU = 10          # 重大公告指数衰减半衰期 (天)
_RISK_WINDOW = 10        # 风险警示回看窗口 (交易日)


def _zero_frame(df: pd.DataFrame) -> pd.DataFrame:
    """追加 5 列全 0 公告因子."""
    for col in ANNOUNCEMENT_FACTOR_COLUMNS:
        df[col] = 0.0
    return df


def load_announcements(ann_dir: str, symbol: str) -> pd.DataFrame:
    """加载个股公告 parquet; 缺失/损坏返回空 DataFrame.

    Args:
        ann_dir: 公告数据目录。
        symbol: 股票代码。

    Returns:
        含 ``date, type`` 列的 DataFrame (按日期升序), 或空表。
    """
    path = os.path.join(ann_dir, f"{symbol}_ann.parquet")
    if not os.path.exists(path):
        logger.warning("公告文件缺失: %s — 公告因子全部置 0", path)
        return pd.DataFrame(columns=["date", "type"])
    try:
        ann = pd.read_parquet(path)
    except (OSError, ValueError) as exc:
        logger.warning("公告文件读取失败: %s (%s) — 公告因子全部置 0", path, exc)
        return pd.DataFrame(columns=["date", "type"])
    if not {"date", "type"} <= set(ann.columns):
        logger.warning("公告文件缺少 date/type 列: %s — 公告因子全部置 0", path)
        return pd.DataFrame(columns=["date", "type"])
    ann = ann[["date", "type"]].copy()
    ann["date"] = pd.to_datetime(ann["date"]).dt.normalize()
    ann["type"] = ann["type"].astype(str).str.strip().str.lower()
    return ann.sort_values("date").reset_index(drop=True)


def _type_flag_series(dates: pd.Series, ann: pd.DataFrame,
                      type_set: set) -> pd.Series:
    """生成 df 每个交易日上"当日是否有该类公告"的 0/1 序列 (causal)."""
    if ann.empty:
        return pd.Series(0.0, index=dates.index)
    hit_dates = set(ann.loc[ann["type"].isin(type_set), "date"])
    return dates.isin(hit_dates).astype(float)


def compute_announcement_factors(df: pd.DataFrame, symbol: str,
                                 ann_dir: str = "data/calendar/announcements") -> pd.DataFrame:
    """为日线 DataFrame 追加 5 维公告事件因子.

    Args:
        df: 含 date 列的日线数据。
        symbol: 股票代码 (定位公告文件)。
        ann_dir: 公告数据目录。

    Returns:
        追加 5 列 ann_* 因子的 DataFrame。
    """
    if "date" not in df.columns:
        raise KeyError("公告因子计算需要 'date' 列")

    df = df.copy()
    dates = pd.to_datetime(df["date"]).dt.normalize()

    ann = load_announcements(ann_dir, symbol)
    if ann.empty:
        logger.info("无公告数据 (%s) — ann_* 全部置 0", symbol)
        return _zero_frame(df)

    # ── 1. 近 5 日公告数 (含当日, 只看历史) ──
    daily_count = dates.map(
        ann.groupby("date").size()
    ).fillna(0.0).astype(float)
    df["ann_count_5d"] = daily_count.rolling(_COUNT_WINDOW, min_periods=1).sum()

    # ── 2. 重大公告 flag (10 日指数衰减编码) ──
    major_flag = _type_flag_series(dates, ann, MAJOR_TYPES)
    df["ann_major_flag"] = exp_decay_encode(major_flag, tau=_DECAY_TAU)

    # ── 3. 业绩公告 flag (衰减编码) ──
    earnings_flag = _type_flag_series(dates, ann, EARNINGS_TYPES)
    df["ann_earnings_flag"] = exp_decay_encode(earnings_flag, tau=_DECAY_TAU)

    # ── 4. 增减持 flag (衰减编码) ──
    hold_flag = _type_flag_series(dates, ann, HOLD_CHANGE_TYPES)
    df["ann_hold_change_flag"] = exp_decay_encode(hold_flag, tau=_DECAY_TAU)

    # ── 5. 风险警示 flag (近 10 个交易日内出现 → 1, 供基础过滤剔除) ──
    risk_flag = _type_flag_series(dates, ann, RISK_WARNING_TYPES)
    df["ann_risk_warning_flag"] = (
        risk_flag.rolling(_RISK_WINDOW, min_periods=1).max()
    )

    df[ANNOUNCEMENT_FACTOR_COLUMNS] = np.nan_to_num(
        df[ANNOUNCEMENT_FACTOR_COLUMNS].to_numpy(dtype=float), nan=0.0
    )
    logger.info("公告因子计算完成: %s, %d 行, 公告 %d 条",
                symbol, len(df), len(ann))
    return df
