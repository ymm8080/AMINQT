"""_diag_breakout_follow.py — 涨停突破事件 + 前期埋伏形态 → 突破后延续性检验 (2026-08-19).

用户问题: "怎么不看今天的 20% 的突破和它过去一两个月的变化"
300911 8/19 放量涨停 +20% (2.2× 量比), 突破前 6-8 月横盘蓄势 (筹码集中/缩量/波动压缩).

本检验以 T 日 = 涨停突破日为事件日 (不同于此前"突破前预测"方向):
  信号 = 20cm 涨停 (ret_1d >= 19.5%)
       + 前期埋伏形态 (T 日之前 1-2 个月): 筹码极值 / 缩量 / 波动压缩 / 低位
  结果 = 突破后:
    A) T 日收盘买入 → T+1..T+11 最大涨幅 (追涨口径)
    B) T+1 开盘买入 → T+11 收盘卖出, 扣 0.2% (次日买入口径, 更实际)
  对比 = 全池基准 vs 涨停 alone vs 涨停+各埋伏形态, 4 子窗稳定性.
  全史 890d dual 池.

用法: python scripts/_diag_breakout_follow.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH

COLS = [
    "symbol", "date", "close", "close_hfq", "open_hfq", "high_hfq", "low_hfq",
    "volume", "chip_entropy", "chip_gini",
]
SLICE = 879  # dual 池全史 (2023-01-03 起)
COST = 0.002


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE if SLICE < len(dates) else 0]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    p["prev_neg"] = g["ret_1d"].shift(1) < 0

    # 突破前形态 (T 日之前可观测, 全部 shift(1) 防泄漏)
    p["ent_rank"] = p.groupby("symbol")["chip_entropy"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    ).shift(1)
    p["gini_rank"] = p.groupby("symbol")["chip_gini"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    ).shift(1)
    p["chip_extreme"] = (p["ent_rank"] < 0.10) & (p["gini_rank"] > 0.90)

    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["shrink20"] = p["vr"].shift(1).rolling(20, min_periods=10).mean() < 0.8
    p["vol60"] = p.groupby("symbol")["ret_1d"].transform(
        lambda v: v.rolling(60, min_periods=40).std()
    ).shift(1)  # T-1 波动率 (涨停日自身会拉高, 必须排除)
    p["vol60_xr"] = p.groupby("dt")["vol60"].rank(pct=True)
    p["low_vol"] = p["vol60_xr"] < 0.3

    # 位置 dd250
    piv = p.pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
    cp = piv.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    max250 = piv.rolling(250, min_periods=60, axis=1).max()
    dd250 = (cp / max250.reindex(columns=cp.columns) - 1.0).stack().rename("dd250").reset_index()
    dd250.columns = ["symbol", "dt", "dd250"]
    p = p.merge(dd250, on=["symbol", "dt"], how="left")

    # 突破后结果
    fwd_max = cp.iloc[:, ::-1].rolling(11, min_periods=1, axis=1).max().iloc[:, ::-1]
    # A: T 日收盘买 → T+1..T+11 最大涨幅
    fm = (fwd_max / cp - 1.0).stack().rename("fut_max").reset_index()
    fm.columns = ["symbol", "dt", "fut_max"]
    p = p.merge(fm, on=["symbol", "dt"], how="left")
    # B: T+1 开盘买 → T+11 收盘卖
    op = p.pivot_table(index="symbol", columns="dt", values="open_hfq", aggfunc="last")
    op = op.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    t10 = (cp.shift(-11, axis=1) / op.shift(-1, axis=1) - 1.0 - COST).stack().rename("t10").reset_index()
    t10.columns = ["symbol", "dt", "t10"]
    p = p.merge(t10, on=["symbol", "dt"], how="left")
    ok = p["ret_1d"].notna()
    limit = ok & (p["ret_1d"] >= 0.195)  # 20cm 涨停

    def stats(sub: pd.DataFrame, label: str) -> None:
        r = sub.dropna(subset=["t10"])
        if len(r) < 30:
            print(f"{label:<32} n={len(r):>5} (样本过少)")
            return
        fm = r["fut_max"]
        print(
            f"{label:<32} n={len(r):>6} 追涨最大≥10%={float((fm >= .10).mean()):>5.1%} "
            f"次日买命中={float((r['t10'] > 0).mean()):>5.1%} "
            f"次日买均值={r['t10'].mean():+6.2%} 中位={r['t10'].median():+6.2%}"
        )

    print(f"== 涨停突破 + 前期埋伏 → 突破后表现 (890d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) ==")
    print("   [追涨最大] = T 日收盘买, T+1..T+11 内最大涨幅 ≥10% 的比例")
    print("   [次日买]   = T+1 开盘买 → T+11 收盘卖, 扣 0.2%\n")
    stats(p[ok], "全池基准")
    stats(p[limit], "涨停 alone")
    stats(p[limit & p["chip_extreme"]], "涨停 + 筹码极值")
    stats(p[limit & p["shrink20"]], "涨停 + 前20日缩量")
    stats(p[limit & p["low_vol"]], "涨停 + 波动压缩(前60日低30%)")
    stats(p[limit & (p["dd250"] < -0.30)], "涨停 + 低位")
    stats(p[limit & p["shrink20"] & p["low_vol"]], "涨停 + 缩量 + 波动压缩")
    stats(p[limit & p["chip_extreme"] & p["low_vol"] & p["shrink20"]], "涨停 + 三形态齐备")

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for name, m in [("涨停 alone", limit), ("涨停+缩量+波动压缩", limit & p["shrink20"] & p["low_vol"])]:
        print(f"{name} 子窗口:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            stats(p[m & (seg == q)], f"  {q}")

    s = p[p["symbol"] == "300911"].tail(10)[
        ["dt", "close_hfq", "ret_1d", "ent_rank", "gini_rank", "shrink20", "low_vol", "dd250", "fut_max", "t10"]
    ]
    print("\n300911 最近 10 日 (8/19 为涨停突破日):")
    print(s.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
