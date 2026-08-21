"""_diag_wick_wf_ab.py — wick_probe 特征 A/B: baseline 208 特征 vs +wick_probe (2026-08-19).

输入 (全部已有):
  WORM replay CSV          = 候选池 + realized_net + base_prod (250d dual)
  baseline wf 检查点       = data/_diag_legacy_wf_pred_dual_e250.parquet (208 特征)
  wick wf 检查点           = data/_diag_wick_wf_pred_dual_e250.parquet (209 特征)

对比 (同一候选池, 唯一差异 = 特征集):
  A. baseline: 排名键 = pred_ret_10d × pred_prob_baseline
  B. +wick:    排名键 = pred_ret_10d × pred_prob_wick
  口径: top-5/日, 命中 (T+10 c2c>0), 实得 (扣0.2%), ≥5%/≥10%, 4 子窗.
  判据: 总赢 + ≥3/4 子窗 + 过闸 (命中≥0.50 / main 实得≥1% / dual≥1.5%).

用法: python scripts/_diag_wick_wf_ab.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, data_others_path

REPLAY_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_prob_head_replay_20260816_220029.csv"
BASE_WF = DATA_DIR / "_diag_legacy_wf_pred_dual_e250.parquet"
WICK_WF = DATA_DIR / "_diag_wick_wf_pred_dual_e250.parquet"
COST = 0.002


def _stats(sub: pd.DataFrame) -> dict:
    r = sub["realized_net"].dropna()
    days = sorted(sub["date"].unique())
    n_sub = 4
    step = len(days) // n_sub
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = r[sub["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i+1}/{n_sub}",
                "hit10": float((seg > 0).mean()) if len(seg) else float("nan"),
                "mean10": float(seg.mean()) if len(seg) else float("nan"),
            }
        )
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": round(float(len(sub) / max(1, sub["date"].nunique())), 1),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
        "sub_windows": subs,
    }


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(REPLAY_CSV)
    df = df[(df["board"] == "dual") & (~df["pain_excluded"])].copy()
    df["date"] = pd.to_datetime(df["date"])
    print(f"[replay] dual 过闸候选 {len(df):,} 行 {df['date'].nunique()} 日", flush=True)

    base = pd.read_parquet(str(BASE_WF))
    wick = pd.read_parquet(str(WICK_WF))
    base["date"] = pd.to_datetime(base["date"])
    wick["date"] = pd.to_datetime(wick["date"])
    base["symbol"] = base["symbol"].astype(str)
    wick["symbol"] = wick["symbol"].astype(str)
    df["symbol"] = df["symbol"].astype(str)

    df = df.merge(base[["symbol", "date", "pred"]], on=["symbol", "date"], how="left")
    df = df.rename(columns={"pred": "pred_prob_base"})
    df = df.merge(wick[["symbol", "date", "pred"]], on=["symbol", "date"], how="left")
    df = df.rename(columns={"pred": "pred_prob_wick"})
    n_base = int(df["pred_prob_base"].notna().sum())
    n_wick = int(df["pred_prob_wick"].notna().sum())
    print(f"[merge] base {n_base:,} / wick {n_wick:,} 行有预测", flush=True)

    df["blend_base"] = df["pred_ret_10d"] * df["pred_prob_base"]
    df["blend_wick"] = df["pred_ret_10d"] * df["pred_prob_wick"]

    def top5(sub: pd.DataFrame, rank_col: str) -> pd.DataFrame:
        return (
            sub.sort_values(["date", rank_col], ascending=[True, False])
            .groupby("date", sort=False)
            .head(5)
        )

    def report(name: str, v: pd.DataFrame, rank_col: str) -> dict:
        s = _stats(top5(v, rank_col))
        sub_s = "  ".join(
            f"{w['win']}:{w['hit10']:.0%}/{w['mean10']:+.2%}" for w in s["sub_windows"]
        )
        print(
            f"  {name:<22}{s['picks']:>5}{s['avg_picks']:>6} {s['hit']:>7.1%} "
            f"{s['mean']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}  {sub_s}",
            flush=True,
        )
        return s

    print("\n===== dual | 250d 同候选池 A/B (top-5/日, T+10 c2c 净) =====", flush=True)
    print(f"  {'变体':<22}{'出票':>5}{'票/日':>6} {'命中':>7} {'实得':>8} "
          f"{'≥5%':>7} {'≥10%':>7}  子窗 hit/实得", flush=True)
    ok_rows = df.dropna(subset=["pred_prob_base", "pred_prob_wick"])
    a = report("baseline (208 特征)", ok_rows, "blend_base")
    b = report("+wick_probe (209 特征)", ok_rows, "blend_wick")

    # 判定
    wins = 0
    subs = []
    for wa, wb in zip(a["sub_windows"], b["sub_windows"]):
        w = wb["mean10"] > wa["mean10"] or (wb["mean10"] == wa["mean10"] and wb["hit10"] > wa["hit10"])
        wins += int(w)
        subs.append({"win": wa["win"], "base": wa, "wick": wb, "wick_wins": bool(w)})
    gate = (b["hit"] >= 0.50) and (b["mean"] >= 0.015)
    verdict = "通过 → 可落地" if (b["mean"] > a["mean"] and wins >= 3 and gate) else (
        "不通过" if not (b["mean"] > a["mean"] and wins >= 3) else "通过但不过闸"
    )
    print(f"\n[判定] 实得 base {a['mean']:+.2%} vs wick {b['mean']:+.2%} | "
          f"子窗 wick 赢 {wins}/4 | 过闸(命中≥50%, 实得≥1.5%): {gate}")
    print(f"[判定] {verdict}")

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "ts": ts,
        "eval_days": a["n_days"],
        "cost": COST,
        "note": "同候选池 (WORM 20260816_220029), 唯一差异 = walk-forward 特征集",
        "base": a,
        "wick": b,
        "sub_windows": subs,
        "verdict": verdict,
    }
    out_path = os.path.join(data_others_path("diag"), f"wick_wf_ab_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
