"""_diag_base_rate_layers.py — 预测质量 × 当日 base_rate 分层 (2026-08-19).

回答: point2 (参与度提示) 的区分维度下, 模型预测是否真的更好?
第五轮 (_diag_stall_regime) 只验证了"入选+滞涨+高频"组合的分层
(低基线日 82.9%/+13.28% vs 高基线日 -4.40%), 未验证纯预测/清单口径.

本脚本对 250d dual replay (与 stall_regime 同源) 按当日 base_rate 分层
(阈值 = STALL_MARKER.base_rate_max=0.732, 与生产参与度提示一致):
  口径1 全池 (~pain_excluded)
  口径2 入选股 (pred_prob_new > base_prod, 交付候选口径)
  口径3 入选 Top5 (每日 prob 前 5, 短名单口径)
输出 命中/实得/≥5%/≥10% + 4 子窗稳定性.

用法: python scripts/_diag_base_rate_layers.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import STALL_MARKER, data_others_path

REPLAY_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_prob_head_replay_20260816_220029.csv"
BASE_MAX = STALL_MARKER["base_rate_max"]


def _stats(sub: pd.DataFrame, label: str) -> dict:
    r = sub["realized_net"].dropna()
    if len(r) < 30:
        print(f"  {label:<34} n={len(r):>5} (样本过少)")
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
        f"  {label:<34} n={len(r):>6} {sub['date'].nunique():>4}d "
        f"命中={out['hit']:>5.1%} 实得={out['mean']:+6.2%} "
        f"≥5%={out['ge5']:>5.1%} ≥10%={out['ge10']:>5.1%}"
    )
    return out


def main() -> int:
    df = pd.read_csv(REPLAY_CSV)
    df = df[(df["board"] == "dual") & (~df["pain_excluded"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    print(f"[replay] dual 候选 {len(df):,} 行 {df['date'].nunique()} 日", flush=True)

    # 逐日 base_rate (当日恒定): 用 base_prod 唯一值
    day_base = (
        df.groupby("date")["base_prod"]
        .first()
        .reset_index()
        .rename(columns={"base_prod": "day_base"})
    )
    df = df.merge(day_base, on="date", how="left")
    low = df["day_base"] < BASE_MAX
    print(f"  低基线日 (base<{BASE_MAX}): {low.sum():,} 行 / 高基线日: {(~low).sum():,} 行")

    df["selected"] = df["pred_prob_new"] > df["base_prod"].fillna(-1)
    top5 = (
        df[df["selected"]]
        .sort_values(["date", "pred_prob_new"], ascending=[True, False])
        .groupby("date")
        .head(5)
    )
    ok = df["realized_net"].notna()

    print("\n== 预测质量 × 当日 base_rate 分层 (250d dual, T+10 c2c 净) ==")
    rows = []
    for name, m in [("全池 (~pain)", ok), ("入选股 (prob>base)", ok & df["selected"])]:
        rows.append(_stats(df[m], f"{name} 全窗"))
        rows.append(_stats(df[m & low], f"  {name} 低基线日"))
        rows.append(_stats(df[m & ~low], f"  {name} 高基线日"))
    t = top5[ok]
    rows.append(_stats(t, "入选 Top5 全窗"))
    rows.append(_stats(t[t["day_base"] < BASE_MAX], "  入选 Top5 低基线日"))
    rows.append(_stats(t[t["day_base"] >= BASE_MAX], "  入选 Top5 高基线日"))

    print("\n== 子窗稳定性 (入选 Top5, 低/高基线日) ==")
    seg = pd.cut(df["date"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        rows.append(_stats(t[seg.loc[t.index] == q], f"  Top5 Q{q} 全"))
        rows.append(_stats(t[(seg.loc[t.index] == q) & (t["day_base"] < BASE_MAX)], f"  Top5 Q{q} 低基线"))
        rows.append(_stats(t[(seg.loc[t.index] == q) & (t["day_base"] >= BASE_MAX)], f"  Top5 Q{q} 高基线"))

    out = {
        "as_of": "2026-08-19",
        "source": REPLAY_CSV,
        "base_threshold": BASE_MAX,
        "defs": {
            "low_base": f"当日 base_prod < {BASE_MAX} (生产参与度提示同阈值)",
            "selected": "pred_prob_new > base_prod",
            "top5": "入选股按 pred_prob_new 每日前 5",
        },
        "groups": rows,
    }
    out_path = os.path.join(data_others_path("diag"), "base_rate_layers_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
