"""_diag_stall_regime.py — 入选+滞涨 × 市场状态: 2025 vs 2026 差异 (2026-08-19).

用户线索: 300911 连续入选 (模型已识别) + 横盘洗盘 (10 日涨幅≈0) → 8/19 涨停突破.
上一轮 (_diag_top_stall_check) 发现入选+滞涨+近20日入选≥3 全窗 65.3%/+6.61%,
但子窗 2/4 — Q1/Q2 (2025) 29.1%/-3.07% 与 45.3%/+0.94% 失效, Q3/Q4 (2026) 77.0%/
+9.87% 与 64.3%/+6.11% 有效. 本脚本回答: 差异来自市场状态还是组合本身.

市场状态 = 自现有数据推导 (用户: 不用交叉外部市场信息, 自己找):
  - mkt_realized : 当日候选池 T+10 已实现净收益均值 (市场已实现强弱, 事后分组)
  - mkt_base     : 当日 base_prod 均值 (池达到概率基线, PIT 当日可判)
  - strong / strong_pt: 各自中位数以上 = 强市日
分 2025 vs 2026 × 强/弱市 四格 + 300911 个股案例 (试盘→横盘→突破全轨迹).

用法: python scripts/_diag_stall_regime.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

REPLAY_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_prob_head_replay_20260816_220029.csv"
STALL_RET = 0.02
MIN_SEL = 3


def _stats(sub: pd.DataFrame, label: str) -> dict:
    r = sub["realized_net"].dropna()
    if len(r) < 30:
        print(f"  {label:<30} n={len(r):>5} (样本过少)")
        return {"label": label, "n": int(len(r)), "insufficient": True}
    out = {
        "label": label,
        "n": int(len(r)),
        "days": int(sub["date"].nunique()),
        "hit": float((r > 0).mean()),
        "mean": float(r.mean()),
        "med": float(r.median()),
        "ge5": float((r >= 0.05).mean()),
        "ge10": float((r >= 0.10).mean()),
    }
    print(
        f"  {label:<30} n={len(r):>6} {sub['date'].nunique():>4}d "
        f"命中={out['hit']:>5.1%} 实得={out['mean']:+6.2%} "
        f"≥5%={out['ge5']:>5.1%} ≥10%={out['ge10']:>5.1%}"
    )
    return out


def main() -> int:
    df = pd.read_csv(REPLAY_CSV)
    df = df[(df["board"] == "dual") & (~df["pain_excluded"])].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[replay] dual 候选 {len(df):,} 行 {df['date'].nunique()} 日", flush=True)

    p = pd.read_parquet(str(PANEL_V3_PATH), columns=["symbol", "date", "close_hfq"])
    p["symbol"] = p["symbol"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    g = p.groupby("symbol")
    p["ret_10d"] = p["close_hfq"] / g["close_hfq"].shift(10) - 1.0
    df = df.merge(p[["symbol", "date", "ret_10d"]], on=["symbol", "date"], how="left")

    ok = df["realized_net"].notna() & df["pred_prob_new"].notna()
    d = df[ok].copy()
    d["selected"] = d["pred_prob_new"] > d["base_prod"].fillna(-1)
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    d["sel_20d"] = (
        d.groupby("symbol")["selected"].transform(
            lambda v: v.rolling(20, min_periods=5).sum()
        )
        - d["selected"]
    )

    # 市场状态: 逐日候选池已实现/基线的均值, 自数据推导
    day = (
        d.groupby("date")
        .agg(
            mkt_realized=("realized_net", "mean"),
            mkt_base=("base_prod", "mean"),
            sel_rate=("selected", "mean"),
        )
        .reset_index()
    )
    day["year"] = day["date"].dt.year
    day["strong"] = day["mkt_realized"] >= day["mkt_realized"].median()
    day["strong_pt"] = day["mkt_base"] >= day["mkt_base"].median()
    d = d.merge(
        day[["date", "mkt_realized", "mkt_base", "strong", "strong_pt", "sel_rate"]],
        on="date",
        how="left",
    )
    d["year"] = d["date"].dt.year

    print("\n== 市场层面: 2025 vs 2026 (逐日候选池, 自数据) ==")
    for y, sub in day.groupby("year"):
        print(
            f"  {y}: {len(sub):>3}日  池T+10实得 {sub['mkt_realized'].mean():+.2%} "
            f"(中位 {sub['mkt_realized'].median():+.2%})  "
            f"base_prod {sub['mkt_base'].mean():+.1%}  入选率 {sub['sel_rate'].mean():.1%}"
        )

    sig = d["selected"] & (d["ret_10d"] < STALL_RET) & (d["sel_20d"] >= MIN_SEL)
    print(f"\n== 入选+滞涨+近20日入选≥{MIN_SEL} → T+10 (c2c 净) ==")
    rows = []
    rows.append(_stats(d[sig], "全窗"))
    rows.append(_stats(d[sig & (d["year"] == 2025)], "2025"))
    rows.append(_stats(d[sig & (d["year"] == 2026)], "2026"))
    rows.append(_stats(d[sig & d["strong"]], "强市日 (池实得中位以上)"))
    rows.append(_stats(d[sig & ~d["strong"]], "弱市日"))
    rows.append(_stats(d[sig & d["strong_pt"]], "高基线日 (base_prod 中位以上)"))
    rows.append(_stats(d[sig & ~d["strong_pt"]], "低基线日"))
    print("  4 格 (年份 × 强/弱市):")
    rows.append(_stats(d[sig & (d["year"] == 2025) & d["strong"]], "2025×强市"))
    rows.append(_stats(d[sig & (d["year"] == 2025) & ~d["strong"]], "2025×弱市"))
    rows.append(_stats(d[sig & (d["year"] == 2026) & d["strong"]], "2026×强市"))
    rows.append(_stats(d[sig & (d["year"] == 2026) & ~d["strong"]], "2026×弱市"))

    # 对照: 入选+已涨 (滞涨的反面)
    print("\n== 对照: 入选+已涨 (ret_10d≥2%) ==")
    risen = d["selected"] & (d["ret_10d"] >= STALL_RET)
    rows.append(_stats(d[risen], "入选+已涨 全窗"))
    rows.append(_stats(d[risen & (d["year"] == 2025)], "入选+已涨 2025"))
    rows.append(_stats(d[risen & (d["year"] == 2026)], "入选+已涨 2026"))

    # 300911 案例: 试盘→横盘→突破全轨迹 + 当时市场状态
    s = d[d["symbol"] == "300911"].sort_values("date")
    p911 = p[(p["symbol"] == "300911") & (p["date"] >= "2026-04-01")].sort_values(
        "date"
    )
    print("\n== 300911 个股案例 (2026-04.. 价格轨迹 + 入选 + 市场) ==")
    day_idx = day.set_index("date")
    for _, r in p911.iterrows():
        sel = ""
        srow = s[(s["date"] == r["date"])]
        if len(srow):
            x = srow.iloc[0]
            sel = "入选" if x["selected"] else ""
            if x["selected"] and x["ret_10d"] < STALL_RET:
                sel = "入选+滞涨"
        mkt = (
            day_idx.loc[r["date"], "mkt_realized"]
            if r["date"] in day_idx.index
            else np.nan
        )
        ret = r["ret_10d"] if not pd.isna(r["ret_10d"]) else np.nan
        print(
            f"  {r['date'].date()} close={r['close_hfq']:8.2f} "
            f"ret10={ret:+7.2%}  {sel:<10} 市场池实得={mkt:+6.2%}"
        )

    out = {
        "as_of": "2026-08-19",
        "source": "legacy_prob_head_replay_20260816_220029 (250d dual)",
        "defs": {
            "selected": "pred_prob_new > base_prod",
            "stall": f"ret_10d < {STALL_RET:.0%}",
            "freq": f"sel_20d >= {MIN_SEL}",
            "market_realized": "当日候选池 T+10 已实现均值 (事后)",
            "market_base": "当日 base_prod 均值 (PIT)",
        },
        "groups": rows,
    }
    out_path = os.path.join(data_others_path("diag"), "stall_regime_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
