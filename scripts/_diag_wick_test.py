"""_diag_wick_test.py — 影线试盘信号检验 (2026-08-19).

用户: "上影加下影线通常就是试盘线". 300911 8/13 = 放量长上影 (上影率 12.2%, 量比 1.85).

检验:
  1. 涨停股 T-1 影线画像 vs 全池 + 次日涨停逐日截面 Rank IC
  2. 试盘线组合 → 10 天突破率 (联动 _diag_ambush_breakout):
     长上影线: upper_wick >= 3% & vr > 1.5
     双影线:   upper_wick >= 2% & lower_wick >= 2% & vr > 1.5
  3. 4 子窗稳定性. 879d dual 池.

用法: python scripts/_diag_wick_test.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import PANEL_V3_PATH, data_others_path

COLS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
]
SLICE = 879
LIMIT = 0.195


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=COLS)
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE if SLICE < len(dates) else 0]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change()
    body_hi = p[["open", "close"]].max(axis=1)
    body_lo = p[["open", "close"]].min(axis=1)
    p["upper_wick"] = (p["high"] - body_hi) / p["close"]
    p["lower_wick"] = (body_lo - p["low"]) / p["close"]
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["label"] = g["ret_1d"].shift(-1) >= LIMIT

    # 未来 10 天突破率 (联动试盘检验)
    piv_c = p.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    cp = piv_c.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    ret_piv = cp.pct_change(axis=1)
    fut10 = ret_piv.iloc[:, ::-1].rolling(10, min_periods=1, axis=1).max().iloc[:, ::-1]
    fut10 = fut10.stack().rename("fut10").reset_index()
    fut10.columns = ["symbol", "dt", "fut10"]
    p = p.merge(fut10, on=["symbol", "dt"], how="left")

    ok = p["label"].notna() & p["ret_1d"].notna()
    print(f"== 影线试盘信号 (879d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) ==")

    # 1. 画像 + 次日涨停 IC
    lim = p[ok & p["label"]]
    print("\n涨停股 T-1 影线画像 vs 全池:")
    for f in ["upper_wick", "lower_wick"]:
        a, b = lim[f].median(), p[ok][f].median()
        print(f"  {f:<12} 涨停组中位={a:>7.3%} 全池中位={b:>7.3%}")
    print("\n逐日截面 Rank IC (次日涨停):")
    ics = {}
    for f in ["upper_wick", "lower_wick"]:
        vals = []
        for _, sub in p[ok].groupby("dt"):
            s = sub.dropna(subset=[f])
            if len(s) < 50 or s["label"].nunique() < 2:
                continue
            vals.append(spearmanr(s[f], s["label"]).statistic)
        m, sd = float(np.mean(vals)), float(np.std(vals))
        ics[f] = {"ic_mean": round(m, 4), "ic_std": round(sd, 4)}
        print(f"  {f:<12} IC={m:+6.4f} ± {sd:.4f}")

    # 2. 试盘线 → 10 天突破率
    print("\n试盘线组合 → 10 天突破率 (T 日收盘形态, S+1..S+10 max ret):")
    base = p[ok & p["fut10"].notna()]

    def stats(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"  {label:<28} n={len(sub):>6} (样本过少)")
            return
        print(
            f"  {label:<28} n={len(sub):>6} 10天突破≥7%={float((sub['fut10'] >= 0.07).mean()):>6.2%} "
            f"涨停≥19.5%={float((sub['fut10'] >= 0.195).mean()):>6.2%}"
        )

    stats(base, "全池基准")
    stats(base[base["vr"] > 1.5], "放量 alone (vr>1.5)")
    stats(base[(base["upper_wick"] >= 0.03) & (base["vr"] > 1.5)], "长上影≥3%+放量")
    stats(
        base[(base["upper_wick"] >= 0.03) & (base["vr"] > 1.5) & (base["ret_1d"] > 0)],
        "长上影+放量+收涨",
    )
    stats(
        base[
            (base["upper_wick"] >= 0.02)
            & (base["lower_wick"] >= 0.02)
            & (base["vr"] > 1.5)
        ],
        "双影≥2%+放量",
    )
    stats(
        base[
            (base["upper_wick"] >= 0.02)
            & (base["lower_wick"] >= 0.02)
            & (base["vr"] > 1.5)
            & (base["ret_1d"] > 0)
        ],
        "双影+放量+收涨",
    )

    print("\n子窗口 (长上影+放量+收涨):")
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    sig = (
        (p["upper_wick"] >= 0.03)
        & (p["vr"] > 1.5)
        & (p["ret_1d"] > 0)
        & p["fut10"].notna()
    )
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        stats(p[sig & (seg == q)], q)

    s = p[p["symbol"] == "300911"].tail(6)[
        ["dt", "close", "upper_wick", "lower_wick", "vr", "ret_1d", "fut10"]
    ]
    print("\n300911 最近 6 日:")
    print(s.round(4).to_string(index=False))

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "rank_ic": ics,
        "verdict": "",
    }
    out_path = os.path.join(data_others_path("diag"), "wick_test_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
