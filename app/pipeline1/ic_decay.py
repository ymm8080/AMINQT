"""
IC 衰减曲线 (P19.0 W4, PIPELINE1_V3.8 §2.3 持仓期与回望窗口 IC 优化)
========================================================================
同一得分对 t+1/t+2/t+3 收益的 Rank IC: 衰减过快 → 信号只适合短持仓;
衰减慢 → 可放宽持仓上限 (momentum=high 上限 5 日的判定依据).
优化器输入为净收益口径 (V3.8 §2.3 沿用).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HORIZONS = (1, 2, 3)  # t+1 / t+2 / t+3


def _label_reference(s: pd.Series, k: int) -> pd.Series:
    """labels 专用前瞻 (合法, 非特征). numpy 切片, 不用 shift(-k)."""
    vals = s.values
    n = len(vals)
    out = np.full(n, np.nan, dtype=float)
    if n > k:
        out[: n - k] = vals[k:]
    return pd.Series(out, index=s.index)


def ic_decay_curve(
    df: pd.DataFrame,
    score_col: str,
    price_col: str = "close_hfq",
    horizons: tuple = HORIZONS,
) -> dict:
    """IC 衰减: score vs forward return (t+k)/t - 1, k ∈ horizons.

    Returns:
        {'ic_t+1': ..., 'ic_t+2': ..., 'ic_t+3': ..., 'decay_ratio_3_1': float,
         'fast_decay': bool (t+3 IC 不足 t+1 一半 → 只适合短持仓)}
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
    g = df.groupby("symbol")[price_col]
    ics = {}
    for k in horizons:
        fwd = g.transform(lambda s, kk=k: _label_reference(s, kk)) / df[price_col] - 1
        sub = pd.DataFrame(
            {"date": df["date"], "score": df[score_col], "ret": fwd}
        ).dropna()
        daily = sub.groupby("date").apply(
            lambda x: (
                spearmanr(x["score"], x["ret"]).statistic
                if x["score"].nunique() > 5 and x["ret"].nunique() > 1
                else np.nan
            )
        )
        ics[f"ic_t+{k}"] = round(float(daily.mean()), 4) if len(daily) else 0.0
    ic1, ic3 = ics.get("ic_t+1", 0.0), ics.get("ic_t+3", 0.0)
    ratio = ic3 / ic1 if abs(ic1) > 1e-9 else 0.0
    return {
        **ics,
        "decay_ratio_3_1": round(ratio, 4),
        "fast_decay": bool(ratio < 0.5),
    }
