"""_diag_chip_trend_check.py — 筹码趋势/蓄势候选特征检验 (2026-08-19).

300911 埋伏形态解剖: chip_entropy 连续 8 周单调下降 (4.457→3.326, -25%),
chip_gini 0.55→0.855 (集中), 8/06-8/12 缩量横盘 (量比<1) 后 8/13 放量试盘,
8/19 放量涨停. 模型 208 特征只有静态 chip_entropy/gini 值, 无趋势特征.

候选特征 (决策日 T 收盘可得, 无前瞻):
  A. entropy_20d_chg  — entropy 20 日变化率 (负=筹码快速集中)
  B. gini_20d_chg     — gini 20 日变化率 (正=集中加速)
  C. entropy_down_20  — 20 日中 entropy 下降天数占比 (持续集中度)
  D. 缩量蓄势         — 近 5 日 vol_ratio<1 且当日 vol_ratio>1.5 (蓄势后放量)
  E. 低位×集中        — dd_250<-40% 与 A/B 交互

检验: dual 池 (30/68) 末 420 交易日, T+1 close 买 / T+11 close 卖, 扣 0.2%.
分层内 vs 基准, 命中率/均值/中位.

用法: python scripts/_diag_chip_trend_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH

COST = 0.0020
SLICE = 420

COLS = [
    "symbol", "date", "close", "open", "high", "low", "volume", "amount",
    "close_hfq", "open_hfq", "high_hfq", "low_hfq",
    "chip_entropy", "chip_gini", "chip_skew_dist", "cost_50pct", "winner_ratio",
]


def main() -> int:
    panel = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    panel = panel[panel["symbol"].str.startswith(("30", "68"))].copy()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["dt"] = pd.to_datetime(panel["date"]).dt.normalize()
    dates = sorted(pd.unique(panel["dt"]))
    cut = dates[-SLICE]
    panel = panel[panel["dt"] >= cut]

    g = panel.groupby("symbol", group_keys=False)

    # --- 候选特征 (全部 T 日收盘可得) ---
    panel["entropy_20d_chg"] = g["chip_entropy"].pct_change(20)
    panel["gini_20d_chg"] = g["chip_gini"].pct_change(20)
    panel["entropy_down_20"] = (
        (g["chip_entropy"].diff(1) < 0).rolling(20, min_periods=15).mean()
    )
    panel["ret_1d"] = g["close_hfq"].pct_change()
    panel["prev_neg"] = g["ret_1d"].shift(1) < 0
    # 停牌日 volume=0 会假"缩量" → 均值排除停牌日, 且缩量/放量判定要求真实有交易
    panel["vol_ma20"] = panel.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    panel["vol_ratio_calc"] = panel["volume"] / panel["vol_ma20"]
    traded = panel["volume"] > 0
    prev_traded5 = (
        panel["volume"].shift(1).rolling(5, min_periods=5).min() > 0
    )  # 前 5 日全部有交易
    panel["vol_shrink5_calc"] = (
        prev_traded5
        & (panel["vol_ratio_calc"].shift(1).rolling(5, min_periods=4).max() < 1.0)
    )  # 前 5 日全部缩量
    panel["vol_break_calc"] = traded & (panel["vol_ratio_calc"] > 1.5)  # 当日放量
    panel["accumulate"] = panel["vol_shrink5_calc"] & panel["vol_break_calc"]

    # --- 位置 dd_250 ---
    pivot = panel.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    max250 = pivot.rolling(250, min_periods=60, axis=1).max()
    panel = panel.merge(
        (pivot / max250 - 1.0).stack().rename("dd_250").reset_index(),
        on=["symbol", "dt"], how="left",
    )

    # --- T+10 净收益 (向量化: buy T+1 / sell T+11) ---
    close_pivot = pivot.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    buy = close_pivot.shift(-1, axis=1)  # T+1 收盘买入
    sell = close_pivot.shift(-11, axis=1)  # T+11 收盘卖出
    t10_long = (sell / buy - 1.0 - COST).stack().rename("t10_ret").reset_index()
    t10_long.columns = ["symbol", "dt", "t10_ret"]
    panel = panel.merge(t10_long, on=["symbol", "dt"], how="left")

    def _stats(sub: pd.DataFrame) -> str:
        r = sub["t10_ret"].dropna()
        return (
            f"n={len(r):>6} 命中={float((r > 0).mean()):>6.1%} 均值={r.mean():+7.2%} "
            f"中位={r.median():+7.2%} ≥10%={float((r >= .10).mean()):>6.1%} ≤-10%={float((r <= -.10).mean()):>6.1%}"
        )

    base_mask = panel["t10_ret"].notna()
    print(f"全池基准:        {_stats(panel[base_mask])}")
    print()

    # --- A/B/C: 筹码趋势分层 ---
    for lo, hi, name in [
        (None, -0.08, "entropy 20d 快速下降 <-8%"),
        (-0.08, -0.03, "entropy 20d 中度 -8~-3%"),
        (-0.03, None, "entropy 20d 平稳/上升 >=-3%"),
    ]:
        if hi is None:
            m = base_mask & (panel["entropy_20d_chg"] >= lo)
        elif lo is None:
            m = base_mask & (panel["entropy_20d_chg"] < hi)
        else:
            m = base_mask & panel["entropy_20d_chg"].between(lo, hi, inclusive="left")
        print(f"{name:<26} {_stats(panel[m])}")

    print()
    for lo, hi, name in [
        (0.05, None, "gini 20d 快速上升 >+5%"),
        (0.0, 0.05, "gini 20d 缓升 0~5%"),
        (None, 0.0, "gini 20d 下降 <0"),
    ]:
        if hi is None:
            m = base_mask & (panel["gini_20d_chg"] >= lo)
        elif lo is None:
            m = base_mask & (panel["gini_20d_chg"] < hi)
        else:
            m = base_mask & panel["gini_20d_chg"].between(lo, hi, inclusive="left")
        print(f"{name:<26} {_stats(panel[m])}")

    print()
    for lo, hi, name in [
        (0.75, None, "entropy 20d 中 >75% 日下降"),
        (0.5, 0.75, "entropy 20d 中 50-75% 日下降"),
        (None, 0.5, "entropy 20d 中 <50% 日下降"),
    ]:
        if hi is None:
            m = base_mask & (panel["entropy_down_20"] >= lo)
        elif lo is None:
            m = base_mask & (panel["entropy_down_20"] < hi)
        else:
            m = base_mask & panel["entropy_down_20"].between(lo, hi, inclusive="left")
        print(f"{name:<26} {_stats(panel[m])}")

    # --- D: 蓄势组合 ---
    print()
    print(f"缩量5日+当日放量: {_stats(panel[base_mask & panel['accumulate']])}")
    print(f"缩量5日(不含放量): {_stats(panel[base_mask & panel['vol_shrink5_calc'] & ~panel['vol_break_calc']])}")

    # --- E: 低位×筹码集中 (300911 形态: dd<-40% + entropy 快速下降) ---
    print()
    low = base_mask & (panel["dd_250"] < -0.40)
    print(f"低位(dd<-40%)基准:        {_stats(panel[low])}")
    print(f"低位+entropy快速下降:     {_stats(panel[low & (panel['entropy_20d_chg'] < -0.08)])}")
    print(f"低位+entropy快降+蓄势:    {_stats(panel[low & (panel['entropy_20d_chg'] < -0.08) & panel['accumulate']])}")
    print(f"低位+entropy快降+蓄势+昨跌: {_stats(panel[low & (panel['entropy_20d_chg'] < -0.08) & panel['accumulate'] & panel['prev_neg']])}")

    # 300911 自身轨迹
    s = panel[panel["symbol"] == "300911"].tail(12)[
        ["dt", "close", "chip_entropy", "entropy_20d_chg", "chip_gini", "gini_20d_chg", "vol_ratio_calc", "vol_shrink5_calc", "vol_break_calc"]
    ]
    print("\n300911 最近 12 日 (决策日形态):")
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
