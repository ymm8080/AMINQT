"""_diag_limitup_case_study.py — 涨停股 case study: 事前共性 vs 全池 (2026-08-19).

用户指示: 对涨停股做 case study 研究共性, 用来增加 PIPELINE 预测度.

事件: 20cm 涨停日 (ret_1d >= 19.5%), dual 池 879 交易日全史.
事前日 T = 涨停前一日, 所有特征 T 日收盘可得 (PIT), 标签 = T+1 是否涨停.

  1. 涨停股 T-1 特征画像 (中位数) vs 全池中位数
  2. 逐日截面 Spearman Rank IC (对次日涨停标签)
  3. top 特征分位分层 → 次日涨停率 (验证区分度形状)

用法: python scripts/_diag_limitup_case_study.py
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
    "close_hfq",
    "low_hfq",
    "volume",
    "chip_entropy",
    "chip_gini",
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
    p["ret_5d"] = p["close_hfq"] / g["close_hfq"].shift(5) - 1.0
    p["ret_10d"] = p["close_hfq"] / g["close_hfq"].shift(10) - 1.0
    p["ret_20d"] = p["close_hfq"] / g["close_hfq"].shift(20) - 1.0

    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["vol20"] = p.groupby("symbol")["ret_1d"].transform(
        lambda v: v.rolling(20, min_periods=12).std()
    )
    p["vol20_xr"] = p.groupby("dt")["vol20"].rank(pct=True)
    p["shrink10"] = p["vr"].shift(1).rolling(10, min_periods=5).mean() < 0.85
    p["probe5"] = (
        p.groupby("symbol")["vr"]
        .transform(lambda v: (v > 1.5).rolling(5, min_periods=5).sum())
        .shift(1)
    )  # 过去 5 日放量天数 (排除 T 日自身)

    p["ent_rank"] = p.groupby("symbol")["chip_entropy"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    )
    p["gini_rank"] = p.groupby("symbol")["chip_gini"].transform(
        lambda v: v.rolling(250, min_periods=120).rank(pct=True)
    )
    p["ent_chg40"] = p["chip_entropy"] / g["chip_entropy"].shift(40) - 1.0

    piv_c = p.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    cp = piv_c.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    max250 = piv_c.rolling(250, min_periods=60, axis=1).max()
    dd250 = (
        (cp / max250.reindex(columns=cp.columns) - 1.0)
        .stack()
        .rename("dd250")
        .reset_index()
    )
    dd250.columns = ["symbol", "dt", "dd250"]
    p = p.merge(dd250, on=["symbol", "dt"], how="left")

    # 标签: T+1 涨停 (ret_1d.shift(-1))
    p["label"] = p.groupby("symbol")["ret_1d"].shift(-1) >= LIMIT

    ok = p["label"].notna() & p["ret_1d"].notna()
    p = p[ok].copy()

    feats = [
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vr",
        "vol20_xr",
        "ent_rank",
        "gini_rank",
        "ent_chg40",
        "shrink10",
        "probe5",
        "dd250",
    ]
    n_lim = int(p["label"].sum())
    print(
        f"== 涨停股 case study (879d dual 池, 涨停样本 n={n_lim}, 基准率 {n_lim / len(p):.3%}) =="
    )
    print(f"截至 {pd.Timestamp(dates[-1]).date()}\n")

    # 1. 画像对比
    lim = p[p["label"]]
    print("特征画像 (涨停股 T-1 中位数 vs 全池):")
    print(f"{'特征':<12}{'涨停组':>10}{'全池':>10}  方向")
    profile = {}
    for f in feats:
        a, b = lim[f].median(), p[f].median()
        profile[f] = {
            "limitup_median": round(float(a), 4),
            "pool_median": round(float(b), 4),
        }
        arrow = (
            "↑↑"
            if abs(a - b) > 0.5 * abs(b)
            else ("↑" if a > b else ("↓" if a < b else "="))
        )
        print(f"{f:<12}{a:>10.4f}{b:>10.4f}  {arrow}")
    print()

    # 2. 逐日截面 Rank IC (对次日涨停标签)
    print("逐日截面 Spearman Rank IC (mean ± std):")
    ic_rows = {}
    for f in feats:
        ics = []
        for _, sub in p.groupby("dt"):
            s = sub.dropna(subset=[f])
            if len(s) < 50 or s["label"].nunique() < 2:
                continue
            ics.append(spearmanr(s[f], s["label"]).statistic)
        mean, std = float(np.mean(ics)), float(np.std(ics))
        ic_rows[f] = {"ic_mean": round(mean, 4), "ic_std": round(std, 4)}
        print(f"{f:<12} IC={mean:+6.4f} ± {std:.4f}  (t={mean / std:+.2f})")
    print()

    # 3. top 特征分层涨停率
    top = sorted(ic_rows, key=lambda k: ic_rows[k]["ic_mean"], reverse=True)[:3]
    for f in top:
        print(f"分层 ({f}, IC={ic_rows[f]['ic_mean']:+.4f}):")
        sub = p.dropna(subset=[f])
        try:
            bins = pd.qcut(
                sub[f], 5, labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
            )
        except ValueError:
            continue
        rates = sub.groupby(bins, observed=True)["label"].agg(["count", "mean"])
        for q, (cnt, r) in rates.iterrows():
            bar = "#" * int(round(r / rates["mean"].max() * 20))
            print(f"  {q:<8} n={int(cnt):>7} 涨停率={r:>6.3%} {bar}")
        print()

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "limitup_samples": int(n_lim),
        "base_rate": round(float(n_lim / len(p)), 4),
        "profile": profile,
        "rank_ic": ic_rows,
        "verdict": "",
    }
    out_path = os.path.join(
        data_others_path("diag"), "limitup_case_study_20260819.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
