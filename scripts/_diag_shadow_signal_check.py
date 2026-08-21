"""_diag_shadow_signal_check.py — 检验"长上影(+昨日跌)"形态是否真是上涨信号 (2026-08-19).

用户观点: 300911 长上影+昨日跌 = 洗盘/试盘信号, 安全买点.
数据检验: dual 全池 (GEM+STAR) 末 420 交易日, 决策日 T 收盘看形态 (无前瞻),
T+1 close 买入, T+11 close 卖出, 扣 0.2% 成本 — 与生产 replay 同口径.
细分: 全部长上影 / +昨日跌 / +昨日跌且近期回调.

用法: python scripts/_diag_shadow_signal_check.py
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

    g = panel.groupby("symbol", group_keys=False)
    panel["ret_1d"] = g["close_hfq"].pct_change()
    panel["body"] = (panel["close_hfq"] - panel["open_hfq"]).abs()
    panel["upper_shadow"] = panel["high_hfq"] - panel[["open_hfq", "close_hfq"]].max(
        axis=1
    )
    panel["lower_shadow"] = (
        panel[["open_hfq", "close_hfq"]].min(axis=1) - panel["low_hfq"]
    )
    # 长上影: 上影 > 2×实体 (实体非零) 或上影 > 3×实体 (超长)
    body_pos = panel["body"] > 0
    panel["long_shadow_2x"] = body_pos & (panel["upper_shadow"] > 2 * panel["body"])
    panel["long_shadow_3x"] = body_pos & (panel["upper_shadow"] > 3 * panel["body"])
    # 昨日跌 (T-1 ret < 0) 和近 3 日回调
    panel["prev_ret"] = g["ret_1d"].shift(1)
    panel["prev_neg"] = panel["prev_ret"] < 0

    # T+10 净收益: 买入 T+1 close, 卖出 T+11 close
    close_pivot = panel.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    close_pivot = close_pivot.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
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

    mask_ok = panel["dt"].isin(close_pivot.columns) & (
        panel["dt"].map(i_of) + 11 < len(close_pivot.columns)
    )
    for name, mask in [
        ("全部 长上影(>2x实体)", mask_ok & panel["long_shadow_2x"]),
        ("长上影 + 昨日跌", mask_ok & panel["long_shadow_2x"] & panel["prev_neg"]),
        ("长上影(>3x) + 昨日跌", mask_ok & panel["long_shadow_3x"] & panel["prev_neg"]),
        ("非长上影基准", mask_ok & ~panel["long_shadow_2x"]),
    ]:
        sub = panel[mask]
        if sub.empty:
            print(f"{name}: 无样本")
            continue
        r = sub.apply(lambda r: _t10(r["symbol"], r["dt"]), axis=1)
        r = r.dropna()
        if len(r) == 0:
            print(f"{name}: 无有效实现")
            continue
        print(
            f"{name:<28} n={len(r):>6} 命中={float((r > 0).mean()):>6.1%} "
            f"均值={r.mean():+7.2%} 中位={r.median():+7.2%} "
            f"≥10%={float((r >= 0.10).mean()):>6.1%} ≤-10%={float((r <= -0.10).mean()):>6.1%}"
        )

    # 300911 自身最近形态
    s = panel[panel["symbol"] == "300911"].tail(5)[
        [
            "dt",
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "close_hfq",
            "ret_1d",
            "long_shadow_2x",
            "prev_neg",
        ]
    ]
    print("\n300911 最近 5 日:")
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
