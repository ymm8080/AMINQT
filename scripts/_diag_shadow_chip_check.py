"""_diag_shadow_chip_check.py — 长上影信号 × 股价位置 × 筹码状态 三层检验 (2026-08-19).

用户观点: "真实低位"不能只看股价, 要看筹码分布 (成本线位置/获利盘/集中度).
检验: dual 池 420 交易日, 形态=长上影(>2x实体)+昨跌; 位置=距250d高点回撤;
筹码=现价偏离持仓成本 (cost_bias 正=大幅获利, 负=深套) + 筹码集中度 (chip_gini).
每层内: 形态 vs 同层基准. T+1 close 买 / T+11 close 卖, 扣 0.2% 成本.

用法: python scripts/_diag_shadow_chip_check.py
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


def main() -> int:
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel = panel[panel["symbol"].str.startswith(("30", "68"))].copy()
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel["dt"] = pd.to_datetime(panel["date"]).dt.normalize()
    dates = sorted(pd.unique(panel["dt"]))
    cut = dates[-SLICE]
    panel = panel[panel["dt"] >= cut]

    panel["ret_1d"] = panel.groupby("symbol")["close_hfq"].pct_change()
    body = (panel["close_hfq"] - panel["open_hfq"]).abs()
    panel["upper_shadow"] = panel["high_hfq"] - panel[["open_hfq", "close_hfq"]].max(
        axis=1
    )
    panel["long_shadow_2x"] = (body > 0) & (panel["upper_shadow"] > 2 * body)
    panel["prev_neg"] = panel.groupby("symbol")["ret_1d"].shift(1) < 0
    panel["sig"] = panel["long_shadow_2x"] & panel["prev_neg"]

    pivot = panel.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    max250 = pivot.rolling(250, min_periods=60, axis=1).max()
    panel = panel.merge(
        (pivot / max250 - 1.0).stack().rename("dd_250").reset_index(),
        on=["symbol", "dt"],
        how="left",
    )

    # 筹码: cost_bias = 现价偏离成本50分位. cost_50pct 是未复权成本 → 必须用真实 close 而非 close_hfq
    # (生产 feature_engine_v35.py:2010 用 close_hfq 是口径 bug, 除权股虚高; 2026-08-19 确认)
    panel["cost_bias"] = (panel["close"] - panel["cost_50pct"]) / panel[
        "cost_50pct"
    ].replace(0, np.nan)

    # T+10 净收益
    close_pivot = pivot.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    i_of = {d: i for i, d in enumerate(close_pivot.columns)}

    def _t10(sym: str, dt) -> float:
        i = i_of[pd.Timestamp(dt)]
        if i + 11 >= len(close_pivot.columns):
            return float("nan")
        pb = close_pivot.at[sym, close_pivot.columns[i + 1]]
        ps = close_pivot.at[sym, close_pivot.columns[i + 11]]
        if not (np.isfinite(pb) and np.isfinite(ps)) or pb <= 0:
            return float("nan")
        return ps / pb - 1.0 - COST

    panel["mask_ok"] = panel["dt"].isin(close_pivot.columns) & (
        panel["dt"].map(i_of) + 11 < len(close_pivot.columns)
    )

    def _stats(sub: pd.DataFrame) -> str:
        r = sub.apply(lambda r: _t10(r["symbol"], r["dt"]), axis=1).dropna()
        return (
            f"n={len(r):>6} 命中={float((r > 0).mean()):>6.1%} 均值={r.mean():+7.2%} "
            f"中位={r.median():+7.2%} ≥10%={float((r >= 0.10).mean()):>6.1%} ≤-10%={float((r <= -0.10).mean()):>6.1%}"
        )

    # ---- 层 1: 股价位置 (低位) × 筹码位置 (cost_bias) ----
    print("== 低位 (dd_250 < -40%) 内, 按筹码位置分层 ==")
    for lo, hi, name in [
        (None, -0.2, "深套 cost_bias<-0.2"),
        (-0.2, 0.0, "微套 -0.2~0"),
        (0.0, 0.5, "微利 0~0.5"),
        (0.5, None, "大幅获利 >0.5"),
    ]:
        low = panel["mask_ok"] & (panel["dd_250"] < -0.40)
        if lo is None:
            chip = low & (panel["cost_bias"] < hi)
        elif hi is None:
            chip = low & (panel["cost_bias"] >= lo)
        else:
            chip = low & panel["cost_bias"].between(lo, hi, inclusive="left")
        sig = chip & panel["sig"]
        base = chip & ~panel["sig"]
        print(f"\n{name:<18}")
        print(f"  形态(长上影+昨跌): {_stats(panel[sig])}")
        print(f"  同层基准:          {_stats(panel[base])}")

    # ---- 层 2: 低位 + 筹码集中度 (chip_gini 截面高位 = 集中) ----
    print("\n== 低位 (dd<-40%) + 获利盘 (cost_bias>0.5) 内, 按筹码集中度分层 ==")
    gini_hi = panel.groupby("dt")["chip_gini"].rank(pct=True) > 0.7
    for use_gini, name in [
        (True, "筹码集中 (gini pct>70%)"),
        (False, "筹码分散 (gini pct<=70%)"),
    ]:
        low_profit = (
            panel["mask_ok"] & (panel["dd_250"] < -0.40) & (panel["cost_bias"] > 0.5)
        )
        chip = low_profit & (gini_hi if use_gini else ~gini_hi)
        sig = chip & panel["sig"]
        base = chip & ~panel["sig"]
        print(f"\n{name:<26}")
        print(f"  形态: {_stats(panel[sig])}")
        print(f"  基准: {_stats(panel[base])}")

    # ---- 层 3: 高位股价 × 筹码也高位 (双高位, 用户说的危险区) ----
    print("\n== 双高位 (dd_250>-0.15 且 cost_bias>0.5) ==")
    both_high = (
        panel["mask_ok"] & (panel["dd_250"] > -0.15) & (panel["cost_bias"] > 0.5)
    )
    sig = both_high & panel["sig"]
    base = both_high & ~panel["sig"]
    print(f"  形态: {_stats(panel[sig])}")
    print(f"  基准: {_stats(panel[base])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
