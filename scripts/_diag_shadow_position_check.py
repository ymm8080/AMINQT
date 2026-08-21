"""_diag_shadow_position_check.py — 长上影信号按股价位置分层检验 (2026-08-19).

用户观点: 长上影+昨日跌在"低位"是洗盘信号 (看涨), 高位才是危险 (看跌).
分层 = 距 250d 最高点回撤 (T 日收盘). 同层内对比: 长上影+昨跌 vs 非长上影基准.
T+1 close 买入 / T+11 close 卖出, 扣 0.2% 成本, 与生产 replay 同口径.

用法: python scripts/_diag_shadow_position_check.py
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

    # 复权价 + 形态
    panel["ret_1d"] = panel.groupby("symbol")["close_hfq"].pct_change()
    body = (panel["close_hfq"] - panel["open_hfq"]).abs()
    panel["upper_shadow"] = panel["high_hfq"] - panel[["open_hfq", "close_hfq"]].max(
        axis=1
    )
    panel["long_shadow_2x"] = (body > 0) & (panel["upper_shadow"] > 2 * body)
    panel["prev_neg"] = panel.groupby("symbol")["ret_1d"].shift(1) < 0

    # 距 250d 高点回撤 (pivot 对齐交易日历, 避免停牌行数污染 rolling)
    pivot = panel.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    max250 = pivot.rolling(250, min_periods=60, axis=1).max()
    drawdown = pivot / max250 - 1.0
    dd_long = drawdown.stack().rename("dd_250").reset_index()
    panel = panel.merge(dd_long, on=["symbol", "dt"], how="left")

    # T+10 净收益 (buy T+1 / sell T+11)
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

    print(f"{'位置层':<12} {'形态':<22} 统计")
    for lo, hi, name in [
        (-1.00, -0.40, "低位 dd<-40%"),
        (-0.40, -0.15, "中低 -40~-15%"),
        (-0.15, 0.00, "中高 -15~0%"),
        (0.00, 1.00, "新高 dd>0"),
    ]:
        in_layer = panel["mask_ok"] & panel["dd_250"].between(lo, hi, inclusive="left")
        sig = in_layer & panel["long_shadow_2x"] & panel["prev_neg"]
        base = in_layer & ~(panel["long_shadow_2x"] & panel["prev_neg"])
        if lo == 0:
            sig = in_layer & panel["long_shadow_2x"] & panel["prev_neg"]
            base = in_layer & ~(panel["long_shadow_2x"] & panel["prev_neg"])
        print(f"{name:<12} {'长上影+昨跌':<22} {_stats(panel[sig])}")
        print(f"{'':<12} {'同层基准(非该形态)':<22} {_stats(panel[base])}")
        print()

    # 300911 自身位置
    s = panel[panel["symbol"] == "300911"].tail(3)[["dt", "close_hfq", "dd_250"]]
    print("300911 最近 3 日位置:")
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
