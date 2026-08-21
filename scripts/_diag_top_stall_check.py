"""_diag_top_stall_check.py — 入选 TOP + 滞涨洗盘 → 未来爆发检验 (2026-08-19).

用户线索: 300911 08-03/08-06/08-07/08-12 连续入选 TOP 短名单 (score 0.78-0.88),
但价格横盘 10 天 (21.2-21.9, 近 10 日涨幅 ≈ 0%) 一直洗盘 → 8/19 涨停.
"模型已识别 (高分) + 价格未兑现 (滞涨) → 终将爆发".

数据源: WORM replay CSV (250d dual 候选行, 含 walk-forward pred_prob_new + base_prod
+ realized_net 标签), 近 10/20 日涨幅从面板现算.

检验:
  入选 = pred_prob_new > base_prod (生产概率闸) 或 截面 top-20
  滞涨 = 近 10 日涨幅 < 阈值 (2%/5%), 近 20 日 < 阈值
  结果 = realized_net (T+10 c2c 净, 扣 0.2%)
  对比: 入选+滞涨 vs 入选+已涨 vs 全候选, 4 子窗.

用法: python scripts/_diag_top_stall_check.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path

REPLAY_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_prob_head_replay_20260816_220029.csv"


def main() -> int:
    df = pd.read_csv(REPLAY_CSV)
    df = df[(df["board"] == "dual") & (~df["pain_excluded"])].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    print(f"[replay] dual 候选 {len(df):,} 行 {df['date'].nunique()} 日")

    # 近 10/20 日涨幅 (T 日收盘可得, PIT)
    p = pd.read_parquet(str(PANEL_V3_PATH), columns=["symbol", "date", "close_hfq"])
    p["symbol"] = p["symbol"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    g = p.groupby("symbol")
    p["ret_10d"] = p["close_hfq"] / g["close_hfq"].shift(10) - 1.0
    p["ret_20d"] = p["close_hfq"] / g["close_hfq"].shift(20) - 1.0
    df = df.merge(
        p[["symbol", "date", "ret_10d", "ret_20d"]],
        on=["symbol", "date"],
        how="left",
    )
    print(f"[merge] ret_10d 覆盖 {df['ret_10d'].notna().mean():.1%}")

    ok = df["realized_net"].notna() & df["pred_prob_new"].notna()
    d = df[ok].copy()
    d["selected"] = d["pred_prob_new"] > d["base_prod"].fillna(-1)
    # 截面 top-20 (按日)
    d["rank"] = d.groupby("date")["pred_prob_new"].rank(ascending=False)
    d["top20"] = d["rank"] <= 20
    # 过去 20 个决策日入选天数 (按 symbol 时序 rolling)
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    d["sel_20d"] = (
        d.groupby("symbol")["selected"].transform(
            lambda v: v.rolling(20, min_periods=5).sum()
        )
        - d["selected"]
    )  # 不含当日

    def stats(sub: pd.DataFrame, label: str) -> None:
        r = sub["realized_net"].dropna()
        if len(r) < 30:
            print(f"{label:<34} n={len(r):>5} (样本过少)")
            return
        print(
            f"{label:<34} n={len(r):>6} 命中={float((r > 0).mean()):>5.1%} "
            f"均值={r.mean():+6.2%} 中位={r.median():+6.2%} ≥10%={float((r >= 0.10).mean()):>5.1%}"
        )

    print("\n== 入选 TOP + 滞涨洗盘 → T+10 表现 (250d dual, 候选口径) ==")
    stats(d, "全候选基准")
    stats(d[d["selected"]], "入选 (prob>base_prod)")
    stats(d[d["top20"]], "截面 top-20")
    print()
    stats(d[d["selected"] & (d["ret_10d"] < 0.02)], "入选 + 近10日滞涨(<2%)")
    stats(d[d["selected"] & (d["ret_10d"] < 0.05)], "入选 + 近10日滞涨(<5%)")
    stats(d[d["selected"] & (d["ret_20d"] < 0.05)], "入选 + 近20日滞涨(<5%)")
    stats(
        d[d["selected"] & (d["ret_10d"] < 0.02) & (d["ret_20d"] < 0.05)],
        "入选 + 10日&20日双滞涨",
    )
    print()
    stats(d[d["selected"] & (d["ret_10d"] >= 0.02)], "入选 + 已涨(≥2%) [对照]")
    stats(d[d["top20"] & (d["ret_10d"] < 0.02)], "top20 + 滞涨(<2%)")
    stats(d[d["top20"] & (d["ret_10d"] < 0.05)], "top20 + 滞涨(<5%)")
    print()
    stats(
        d[
            d["selected"]
            & (d["ret_10d"] < 0.02)
            & (d["ret_20d"] < 0.05)
            & (d["ret_10d"] > -0.15)
        ],
        "双滞涨+非暴跌",
    )
    print()
    stats(
        d[d["selected"] & (d["ret_10d"] < 0.02) & (d["sel_20d"] >= 2)],
        "入选+滞涨+近20日入选≥2次",
    )
    stats(
        d[d["selected"] & (d["ret_10d"] < 0.02) & (d["sel_20d"] >= 3)],
        "入选+滞涨+近20日入选≥3次",
    )
    stats(
        d[d["selected"] & (d["ret_10d"] < 0.02) & (d["sel_20d"] >= 4)],
        "入选+滞涨+近20日入选≥4次",
    )

    print("\n子窗口 (入选+近10日滞涨<2%):")
    seg = pd.cut(d["date"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    sig = d["selected"] & (d["ret_10d"] < 0.02)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        stats(d[sig & (seg == q)], q)
    print("\n子窗口 (入选+滞涨+近20日入选≥3次):")
    sig3 = d["selected"] & (d["ret_10d"] < 0.02) & (d["sel_20d"] >= 3)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        stats(d[sig3 & (seg == q)], q)
    print("\n子窗口 (入选+滞涨+近20日入选≥4次):")
    sig4 = d["selected"] & (d["ret_10d"] < 0.02) & (d["sel_20d"] >= 4)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        stats(d[sig4 & (seg == q)], q)

    s = d[(d["symbol"] == "300911")].sort_values("date")[
        ["date", "pred_prob_new", "base_prod", "selected", "ret_10d", "realized_net"]
    ]
    print("\n300911 候选期逐日:")
    print(s.round(3).to_string(index=False))

    out = {
        "as_of": "2026-08-19",
        "source": "legacy_prob_head_replay_20260816_220029 (250d dual)",
        "defs": {
            "selected": "pred_prob_new > base_prod",
            "stall": "ret_10d < 2%",
        },
    }
    out_path = os.path.join(data_others_path("diag"), "top_stall_check_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
