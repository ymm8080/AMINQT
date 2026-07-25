"""
评估指标 (P19.0 W3, PIPELINE1_V3.8 §五 过程指标)
====================================================
日度 Rank IC 序列 / ICIR / 阶段一点火验收门禁:
  Rank IC ≥ 0.03 且 ICIR ≥ 0.3 且 高波动桶 IC ≥ 0.02 且 训练 IC ≤ 0.15 (无泄漏)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 阶段一点火门禁 (P19.0 通过标准)
IGNITION_IC_MIN = 0.03
IGNITION_ICIR_MIN = 0.3
IGNITION_HIGH_VOL_IC_MIN = 0.02
TRAIN_IC_LEAK_MAX = 0.15


def daily_ic_series(df: pd.DataFrame, score_col: str, label_col: str) -> pd.Series:
    """日度横截面 Rank IC 序列 (index=date) — 净收益口径标签由调用方保证."""
    sub = df[["date", score_col, label_col]].dropna()
    return (
        sub.groupby("date")
        .apply(
            lambda g: (
                spearmanr(g[score_col], g[label_col]).statistic
                if g[score_col].nunique() > 5 and g[label_col].nunique() > 1
                else np.nan
            )
        )
        .dropna()
    )


def rank_ic(df: pd.DataFrame, score_col: str, label_col: str) -> float:
    """Rank IC = 日度 IC 序列均值 (带符号, 验收口径)."""
    ics = daily_ic_series(df, score_col, label_col)
    return float(ics.mean()) if len(ics) else 0.0


def icir(df: pd.DataFrame, score_col: str, label_col: str) -> float:
    """ICIR = 日度 IC 均值 / 标准差 × √252 (年化)."""
    ics = daily_ic_series(df, score_col, label_col)
    if len(ics) < 5 or ics.std() == 0:
        return 0.0
    return float(ics.mean() / ics.std() * np.sqrt(252))


def bucket_ic_high_vol(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    atr_col: str = "ATR_pct",
) -> float:
    """E.4 分波动桶 IC: 按 ATR 五桶独立 Rank IC, 取高波动桶 (Q5).

    点火门禁 (P19.0) 高波动桶 IC ≥ 0.02 的直接输入;
    完整五桶报告见 dynamic_engine.DynamicEngine.bucket_ic.
    """
    from .dynamic_engine import DynamicEngine

    return float(
        DynamicEngine.bucket_ic(df, score_col, label_col, atr_col)["high_vol_ic"]
    )


def ignition_gate(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    high_vol_ic: float | None = None,
    train_ic: float = 0.0,
    atr_col: str = "ATR_pct",
) -> dict:
    """阶段一点火验收 (P19.0): 全部满足方可进入阶段二.

    high_vol_ic=None 且面板含 atr_col 时自动按 ATR 五桶计算 (E.4).
    不达标 → 按序排查 未来函数→复权→幸存者偏差; 禁止加特征硬堆 IC.
    """
    ic = rank_ic(df, score_col, label_col)
    ir = icir(df, score_col, label_col)
    if high_vol_ic is None:
        high_vol_ic = (
            bucket_ic_high_vol(df, score_col, label_col, atr_col)
            if atr_col in df.columns
            else 0.0
        )
    checks = {
        "rank_ic": {"value": round(ic, 4), "pass": ic >= IGNITION_IC_MIN},
        "icir": {"value": round(ir, 4), "pass": ir >= IGNITION_ICIR_MIN},
        "high_vol_ic": {
            "value": round(high_vol_ic, 4),
            "pass": high_vol_ic >= IGNITION_HIGH_VOL_IC_MIN,
        },
        "train_ic_no_leak": {
            "value": round(train_ic, 4),
            "pass": train_ic <= TRAIN_IC_LEAK_MAX,
        },
    }
    return {"pass": all(c["pass"] for c in checks.values()), "checks": checks}
