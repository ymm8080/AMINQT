"""PIPELINE 并行多系统 打分核心 (2026-08-04).

双头架构:
  - 幅度头 (magnitude head): 池特征截面合成的期望净收益方向分
  - 概率头 (probability head): 池特征一致通过率 → 上涨概率代理

打分路径 (纯特征, 无前瞻): 对每个 (symbol, date), 取特征池各列
每日期截面分位数, 等权平均 → 合成池分. 系统按池分降序取 TOP-N.

双头验收: 对选中 TOP-N 逐视界量 平均净收益(幅度) + 胜率(概率).
纯向量化, 禁 for 循环遍历股票.
"""

from __future__ import annotations

import pandas as pd


def cross_rank(df: pd.DataFrame, col: str) -> pd.Series:
    """每日期截面分位数排名 (升序, 0~1; 值越大排名越高)."""
    return df.groupby("date")[col].rank(pct=True)


def pool_score(
    df: pd.DataFrame, pool: tuple[str, ...], weights: dict[str, float] | None = None
) -> pd.Series:
    """特征池等权(或加权) 截面分位合成分, 索引对齐 df.

    缺列特征自动跳过; 全部缺列 → 抛错 (配置错误, 须大声失败).
    """
    avail = [c for c in pool if c in df.columns]
    if not avail:
        raise ValueError(f"特征池 {pool} 在面板中无可用列")
    if weights is None:
        w = {c: 1.0 / len(avail) for c in avail}
    else:
        w = {c: weights.get(c, 0.0) for c in avail}
        denom = sum(w.values())
        if denom <= 0:
            raise ValueError("权重和必须 > 0")
        w = {c: v / denom for c, v in w.items()}
    score = pd.Series(0.0, index=df.index, dtype=float)
    for c in avail:
        score = score + w[c] * cross_rank(df, c)
    return score


def select_topn(df: pd.DataFrame, score: pd.Series, top_n: int) -> pd.DataFrame:
    """每日期截面按合成池分降序取 TOP-N (返回含 score 的切片)."""
    if top_n <= 0:
        return pd.DataFrame()
    sub = df[["symbol", "date"]].copy()
    sub["score"] = score.values
    sub = sub.dropna(subset=["score"])
    if sub.empty:
        return sub
    top = (
        sub.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", group_keys=False)
        .head(top_n)
    )
    return top


def measure_dual_head(df: pd.DataFrame, label_col: str) -> dict:
    """对选中切片量双头: 幅度=平均净收益, 概率=胜率.

    返回 {mag, winrate, n}; n<5 判定数据不足 (不视为通过).
    """
    v = df[label_col].dropna()
    if len(v) < 5:
        return {"mag": float("nan"), "winrate": float("nan"), "n": int(len(v))}
    return {
        "mag": float(v.mean()),
        "winrate": float((v > 0).mean()),
        "n": int(len(v)),
    }


def dual_head_ok(m: dict, min_winrate: float = 0.55, min_mag: float = 0.0) -> bool:
    """双头通过: 幅度>0 且 胜率>=阈值 (缺任一数据 → False)."""
    if m["n"] < 5:
        return False
    return bool((m["winrate"] >= min_winrate) and (m["mag"] > min_mag))
