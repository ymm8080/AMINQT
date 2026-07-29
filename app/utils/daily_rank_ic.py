"""
日度 Rank IC 公共工具 (P19.0)
================================
统一封装按 date 分组的横截面 Spearman Rank IC 计算,
供 metrics / leakage_audit / ic_decay / oos_monitor / ic_screener 复用。

设计要点:
- 全部输入为长格式 DataFrame, date 列可配置 (默认 "date")。
- 单截面有效性检查: x 唯一值数 ≥ min_x_unique, y 唯一值数 ≥ min_y_unique。
- 对空/无效截面返回 NaN, 调用方决定 dropna 或填充。
- 不依赖具体业务标签名, x/y 列名由调用方传入。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_sectional_rank_ic(
    x: pd.Series,
    y: pd.Series,
    min_x_unique: int = 5,
    min_y_unique: int = 2,
) -> float:
    """单截面 Spearman Rank IC.

    Args:
        x: 预测得分/因子值序列。
        y: 标签序列 (如次日收益)。
        min_x_unique: x 最小唯一值数, 低于此值返回 NaN。
        min_y_unique: y 最小唯一值数, 低于此值返回 NaN。

    Returns:
        Spearman 相关系数 (statistic); 无法计算时返回 NaN。
    """
    if len(x) < 2 or x.nunique() < min_x_unique or y.nunique() < min_y_unique:
        return float(np.nan)
    # 防御性对齐 (spearmanr 按位置配对, 非按 index; 防止调用方传入了不同 index 的 x/y)
    x, y = x.align(y, join="inner")
    if len(x) < 2 or x.nunique() < min_x_unique or y.nunique() < min_y_unique:
        return float(np.nan)
    try:
        return float(spearmanr(x, y).statistic)
    except (ValueError, TypeError):
        return float(np.nan)


def daily_rank_ic_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
    min_x_unique: int = 5,
    min_y_unique: int = 2,
) -> pd.Series:
    """日度横截面 Rank IC 序列 (index=date, value=IC).

    Args:
        df: 长格式数据, 含 date/x/y 三列。
        x_col: 预测得分/因子列名。
        y_col: 标签列名。
        date_col: 日期列名, 默认 "date"。
        min_x_unique: 单截面 x 最小唯一值数。
        min_y_unique: 单截面 y 最小唯一值数。

    Returns:
        index=date 的 IC 序列; 无效日期为 NaN, 已 dropna。
    """
    cols = [date_col, x_col, y_col]
    sub = df[cols].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _ic(group: pd.DataFrame) -> float:
        return cross_sectional_rank_ic(
            group[x_col],
            group[y_col],
            min_x_unique=min_x_unique,
            min_y_unique=min_y_unique,
        )

    return (
        sub.groupby(date_col, observed=True).apply(_ic, include_groups=False).dropna()
    )


def mean_rank_ic(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
    min_x_unique: int = 5,
    min_y_unique: int = 2,
    abs_mean: bool = False,
) -> float:
    """日度 Rank IC 序列均值.

    Args:
        abs_mean: True 时先对每日 IC 取绝对值再求均值 (方向无关, 用于筛强度)。

    Returns:
        均值 IC; 无有效日期返回 0.0。
    """
    ics = daily_rank_ic_series(
        df,
        x_col,
        y_col,
        date_col=date_col,
        min_x_unique=min_x_unique,
        min_y_unique=min_y_unique,
    )
    if ics.empty:
        return 0.0
    vals = ics.abs().values if abs_mean else ics.values
    return float(np.nanmean(vals))


def icir(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
    min_x_unique: int = 5,
    min_y_unique: int = 2,
    annual_factor: float = 252.0,
) -> float:
    """ICIR = 日度 IC 均值 / 标准差 × √annual_factor.

    Returns:
        ICIR; 样本不足或 std==0 返回 0.0。
    """
    ics = daily_rank_ic_series(
        df,
        x_col,
        y_col,
        date_col=date_col,
        min_x_unique=min_x_unique,
        min_y_unique=min_y_unique,
    )
    if len(ics) < 5 or ics.std() == 0:
        return 0.0
    return float(ics.mean() / ics.std() * np.sqrt(annual_factor))


def random_baseline_check(
    df: pd.DataFrame,
    y_col: str,
    date_col: str = "date",
    n_trials: int = 10,
    threshold: float = 0.01,
    seed: int = 42,
) -> dict:
    """随机因子基准检验 (安全网 #14).

    生成 n_trials 列随机数作为伪因子, 分别计算 IC.
    如果任一随机因子的 |IC| > threshold → 样本/计算方法可能有系统偏差,
    真实因子的 IC 不可直接采信.

    Args:
        df: 含 date / label 的长格式数据。
        y_col: 标签列名。
        date_col: 日期列名。
        n_trials: 随机因子数量 (≥5 建议)。
        threshold: 随机 IC 上限 (A股日频应在 ±0.005 以内, 宽松设 0.01)。
        seed: 随机种子 (可复现)。

    Returns:
        {'pass': bool, 'max_abs_ic': float, 'trials': [float, ...],
         'warning': str | None}
    """
    rng = np.random.default_rng(seed)
    trials = []
    for i in range(n_trials):
        rand_col = f"__rand_baseline_{i}__"
        df_eval = df.copy()
        df_eval[rand_col] = rng.uniform(0, 1, len(df_eval))
        ic = mean_rank_ic(df_eval, rand_col, y_col, date_col=date_col, abs_mean=True)
        trials.append(round(ic, 6))
    max_abs = max(trials)
    passed = max_abs <= threshold
    warning = (
        None
        if passed
        else (
            f"随机因子 |IC|={max_abs:.6f} > {threshold}. "
            f"样本或 IC 计算方法可能有系统偏差, 真实 IC 不可直接采信. "
            f"检查: 1) 是否含停牌/ST/涨跌停样本; 2) 是否未来函数; "
            f"3) 标签计算方向是否正确."
        )
    )
    return {
        "pass": passed,
        "max_abs_ic": max_abs,
        "trials": trials,
        "warning": warning,
    }
