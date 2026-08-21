"""_diag_chip_extreme_burst.py — 筹码集中极值 × 缩量横盘 → 爆发率检验 (2026-08-19).

300911 解剖: 8/12 决策日 chip_entropy 处于自身 250d 0.4% 分位 (历史极低),
chip_gini 处于 100% 分位 (历史极高) — "筹码集中到极值后的变盘临界点".
此前用 20d 变化率检验遗漏该信号 (8/12 已是局部最低点, 变化率趋零).

本检验:
  信号 = entropy_rank250 < lo 且 gini_rank250 > hi (自身滚动分位极值)
        + 可选 缩量横盘 (5日缩量 / 波动压缩)
  结果 = 爆发率 (T+1..T+11 内 close_hfq 最大涨幅 ≥ 10%/15%/20%)
        — 用户关心"爆发"而非小幅上涨, 平均收益会稀释尾部.
  对比 = 同池基准爆发率 + 4 子窗稳定性. 全史 3.5 年.

用法: python scripts/_diag_chip_extreme_burst.py [slice_days=890]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH

COLS = [
    "symbol", "date", "close_hfq", "open_hfq", "high_hfq", "low_hfq",
    "volume", "chip_entropy", "chip_gini",
]


def main() -> int:
    slice_days = int(sys.argv[1]) if len(sys.argv) > 1 else 890
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-slice_days if slice_days < len(dates) else 0]]

    # 自身 250d 滚动分位 (PIT: 当日收盘可得)
    g = p.groupby("symbol", group_keys=False)
    p["ent_rank"] = p.groupby("symbol")["chip_entropy"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    )
    p["gini_rank"] = p.groupby("symbol")["chip_gini"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    )

    # 缩量横盘
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["vol60"] = p.groupby("symbol")["ret_1d"].transform(
        lambda v: v.rolling(60, min_periods=40).std()
    )
    p["vol60_xr"] = p.groupby("dt")["vol60"].rank(pct=True)
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    prev_traded5 = p["volume"].shift(1).rolling(5, min_periods=5).min() > 0
    shrink5 = prev_traded5 & (p["vr"].shift(1).rolling(5, min_periods=4).max() < 1.0)

    # 爆发: T+1 收盘买入, T+1..T+11 内 close_hfq 最大涨幅
    piv = p.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    cp = piv.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    fwd_max = cp.iloc[:, ::-1].rolling(11, min_periods=1, axis=1).max().iloc[:, ::-1]  # 未来 11 天 max (含当日)
    buy = cp.shift(-1, axis=1)  # T+1 买入价
    burst = (fwd_max / buy - 1.0).stack().rename("max_gain").reset_index()
    burst.columns = ["symbol", "dt", "max_gain"]
    p = p.merge(burst, on=["symbol", "dt"], how="left")
    ok = p["max_gain"].notna() & p["ent_rank"].notna()

    def stats(sub: pd.DataFrame, label: str) -> None:
        r = sub["max_gain"].dropna()
        if len(r) < 30:
            print(f"{label:<28} n={len(r):>5} (样本过少)")
            return
        print(
            f"{label:<28} n={len(r):>6} 命中>0={float((r > 0).mean()):>5.1%} "
            f"爆发≥10%={float((r >= .10).mean()):>5.1%} ≥15%={float((r >= .15).mean()):>5.1%} "
            f"≥20%={float((r >= .20).mean()):>5.1%} 中位={r.median():+5.2%}"
        )

    print(f"== 筹码极值 × 爆发率 (全史 {slice_days}d, 截至 {pd.Timestamp(dates[-1]).date()}) ==")
    stats(p[ok], "全池基准")
    piv250 = piv.rolling(250, min_periods=60, axis=1).max()
    dd250 = (cp / piv250.reindex(columns=cp.columns) - 1.0)
    dd250 = dd250.stack().rename("dd250").reset_index()
    dd250.columns = ["symbol", "dt", "dd250"]
    p = p.merge(dd250, on=["symbol", "dt"], how="left")
    stats(p[ok & (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90)], "ent低分位+gini高分位")
    stats(p[ok & (p["ent_rank"] < 0.05) & (p["gini_rank"] > 0.95)], "更严: 5%/95%分位")
    stats(p[ok & (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90) & shrink5], "+ 缩量5日")
    stats(p[ok & (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90) & (p["vol60_xr"] < 0.3)], "+ 波动压缩")
    stats(p[ok & (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90) & (p["ret_1d"] < 0)], "+ 当日下跌")
    stats(p[ok & (p["dd250"] < -0.30)], "低位 alone")
    stats(p[ok & (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90) & (p["dd250"] < -0.30)], "+ 低位<250日高点70%")

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    core = p["ent_rank"] < 0.10
    sig = ok & core & (p["gini_rank"] > 0.90)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        stats(p[sig & (seg == q)], f"极值信号 {q}")

    # 300911 自身轨迹
    s = p[p["symbol"] == "300911"].tail(8)[
        ["dt", "close_hfq", "ent_rank", "gini_rank", "vr", "max_gain"]
    ]
    print("\n300911 最近 8 日:")
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
