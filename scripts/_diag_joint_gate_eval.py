"""联合门 (T+2/T+3) 选股效果评估 — 用户问: 联合门在选股 & 入选股涨幅预测/概率上会不会更好 (2026-08-07).

Part A (30日主裁决): 复用 t3_decision raw.parquet (legacy 全池 30 日预测, ~885只/日),
每日按生产 score_w 取 top-N 作为短名单候选, 分别套 旧门(T+3>0) vs 联合门(T+3>0 或 T+2>1%且T+3>-1%),
对比两组已实现收益/上涨率/组内预测 IC/概率排序力. 救回股=联合门多保留的股单独一行.

Part B (真实并行短名单实例): 遍历交付的 parallel_shortlist_*.csv (08-04/05/06, 含 301326),
把已交付(联合门产出)拆成 旧门也会留(pred_mag_3d>0) vs 联合门新增(pred_mag_3d∈(-1%,0]且pred_mag_2d>1%),
对比已实现表现.

结果 WORM → BACKTEST_RESULT_DIR/t3_joint_gate_eval_<ts>/
用法: python scripts/_diag_joint_gate_eval.py [src_dir]
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH, SHORTLIST_SCORE

SRC_DIR = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else (BACKTEST_RESULT_DIR / "t3_decision_20260807_002008")
)
CLS = 0.005  # 上涨率阈值 +0.5%
T2_MIN = SHORTLIST_SCORE["select_gate"]["t2_min"]
T3_FLOOR = SHORTLIST_SCORE["select_gate"]["t3_floor"]
HW = SHORTLIST_SCORE["horizon_w"]
GW, PW = SHORTLIST_SCORE["gain_w"], SHORTLIST_SCORE["prob_w"]


def old_gate(df: pd.DataFrame) -> pd.Series:
    return df["pred_ret_3d"] > 0.0


def joint_gate(df: pd.DataFrame) -> pd.Series:
    return (df["pred_ret_3d"] > 0.0) | (
        (df["pred_ret_2d"] > T2_MIN) & (df["pred_ret_3d"] > T3_FLOOR)
    )


def realized_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["symbol", "date", "close_hfq"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")["close_hfq"]
    for h in (1, 2, 3, 5):
        out[f"ret_{h}d"] = g.shift(-h) / out["close_hfq"] - 1
    return out[["symbol", "date", "ret_1d", "ret_2d", "ret_3d", "ret_5d"]]


def score_w(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in ("2d", "3d", "5d"):
        gh, ph = f"norm_g_{h}", f"norm_p_{h}"
        g, p = df[f"pred_ret_{h}"], df[f"prob_up_{h}"]
        glo, ghi = g.min(), g.max()
        plo, phi = p.min(), p.max()
        out[gh] = ((g - glo) / (ghi - glo)).fillna(0.0) if ghi > glo else 0.0
        out[ph] = ((p - plo) / (phi - plo)).fillna(0.0) if phi > plo else 0.0
    out["score_w"] = sum(
        HW[h] * (GW * out[f"norm_g_{h}"] + PW * out[f"norm_p_{h}"])
        for h in ("2d", "3d", "5d")
    )
    return out


def subset_metrics(sub: pd.DataFrame, tag: str, day) -> dict:
    sub = sub.dropna(subset=["ret_3d", "ret_2d"])
    row = {"day": day, "group": tag, "n": len(sub)}
    if len(sub) < 3:
        row.update(
            {
                "ret3d": np.nan,
                "ret2d": np.nan,
                "win3d": np.nan,
                "win3d5": np.nan,
                "ic3d": np.nan,
                "prob_ric3d": np.nan,
            }
        )
        return row
    row["ret3d"] = float(sub["ret_3d"].mean())
    row["ret2d"] = float(sub["ret_2d"].mean())
    row["win3d"] = float((sub["ret_3d"] > 0).mean())
    row["win3d5"] = float((sub["ret_3d"] > CLS).mean())
    if sub["ret_3d"].nunique() > 1:
        r = spearmanr(sub["pred_ret_3d"], sub["ret_3d"])
        row["ic3d"] = float(r.statistic) if r.statistic == r.statistic else np.nan
        r2 = spearmanr(sub["prob_up_3d"], sub["ret_3d"])
        row["prob_ric3d"] = (
            float(r2.statistic) if r2.statistic == r2.statistic else np.nan
        )
    else:
        row["ic3d"], row["prob_ric3d"] = np.nan, np.nan
    return row


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"t3_joint_gate_eval_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= all_dates[-300]].reset_index(drop=True)
    realized = realized_from_panel(panel)
    summary: dict = {"t2_min": T2_MIN, "t3_floor": T3_FLOOR}

    # ── Part A: 30 日 legacy 全池 top-N 门对比 ──
    raw = pd.read_parquet(SRC_DIR / "raw.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.merge(realized, on=["symbol", "date"], how="left")
    print("========== Part A: 30 日全池 top-N 旧门 vs 联合门 ==========")
    partA = []
    rescued_all: list[str] = []
    for N in (10, 14, 20):
        rows = []
        for _d, g in raw.groupby("date"):
            g = score_w(g).dropna(subset=["score_w"])
            if len(g) < max(N, 8):
                continue
            top = g.nlargest(N, "score_w")
            old = top[old_gate(top)]
            joint = top[joint_gate(top)]
            rescued = joint[~joint.index.isin(old.index)]
            rescued_all += rescued["symbol"].tolist()
            for df2, tag in ((old, "old"), (joint, "joint")):
                rows.append(subset_metrics(df2, tag, _d))
            if len(rescued):
                rows.append(subset_metrics(rescued, "rescued", _d))
        A = pd.DataFrame(rows)
        partA.append((N, A))
        for period, tag_n in (("月", None), ("周", 5)):
            sub = A
            if tag_n:
                days = sorted(A["day"].unique())[-tag_n:]
                sub = A[A["day"].isin(days)]
            print(f"\n[top-{N} / {period}]")
            for grp in ("old", "joint", "rescued"):
                g = sub[sub["group"] == grp]
                if not len(g):
                    continue
                print(
                    f"  {grp:8s} n={g['n'].mean():4.1f}/日 | "
                    f"T+3 {g['ret3d'].mean():+.4f} | T+2 {g['ret2d'].mean():+.4f} | "
                    f"上涨率 {g['win3d'].mean():.1%} | 组内3dIC {g['ic3d'].mean():+.4f} | "
                    f"概率秩IC {g['prob_ric3d'].mean():+.4f}"
                )
    # 落盘 Part A
    for N, A in partA:
        A.to_csv(out_dir / f"partA_top{N}_daily.csv", index=False)
        for period, tag_n in (("month", None), ("week", 5)):
            sub = A
            if tag_n:
                days = sorted(A["day"].unique())[-tag_n:]
                sub = A[A["day"].isin(days)]
            agg = sub.groupby("group").agg(
                n=("n", "mean"),
                ret3d=("ret3d", "mean"),
                ret2d=("ret2d", "mean"),
                win3d=("win3d", "mean"),
                win3d5=("win3d5", "mean"),
                ic3d=("ic3d", "mean"),
                prob_ric3d=("prob_ric3d", "mean"),
            )
            agg.to_csv(out_dir / f"partA_top{N}_{period}.csv")
            summary[f"partA_top{N}_{period}"] = agg.round(4).to_dict("index")
    print(
        f"\n[PartA] 联合门 30 日内累计救回股次: {len(rescued_all)} "
        f"(唯一 {len(set(rescued_all))} 只)"
    )

    # ── Part A2: 救回带实得 (全池逐日) — 联合门多留的股到底值不值 ──
    # 救回带 = pred_ret_3d ∈ (t3_floor, 0] 且 pred_ret_2d > t2_min; 对照 = 旧门保留 (pred_ret_3d > 0).
    print(
        "\n========== Part A2: 救回带实得 (全池逐日, 联合门多留的股 vs 旧门保留的股) =========="
    )
    rb_rows = []
    for _d, g in raw.groupby("date"):
        g2 = g.dropna(subset=["ret_3d", "ret_2d", "pred_ret_3d", "pred_ret_2d"])
        old = g2[g2["pred_ret_3d"] > 0]
        rescued = g2[
            (g2["pred_ret_3d"] > T3_FLOOR)
            & (g2["pred_ret_3d"] <= 0)
            & (g2["pred_ret_2d"] > T2_MIN)
        ]
        for df2, tag in ((old, "old_keep"), (rescued, "rescue_band")):
            if len(df2) < 3:
                continue
            rb_rows.append(
                {
                    "day": _d,
                    "group": tag,
                    "n": len(df2),
                    "ret3d": float(df2["ret_3d"].mean()),
                    "ret2d": float(df2["ret_2d"].mean()),
                    "ret5d": float(df2["ret_5d"].mean()),
                    "win3d": float((df2["ret_3d"] > 0).mean()),
                    "win3d5": float((df2["ret_3d"] > CLS).mean()),
                }
            )
    RB = pd.DataFrame(rb_rows)
    RB.to_csv(out_dir / "partA2_rescueband_daily.csv", index=False)
    for period, tag_n in (("月", None), ("周", 5)):
        sub = RB
        if tag_n:
            days = sorted(RB["day"].unique())[-tag_n:]
            sub = RB[RB["day"].isin(days)]
        print(f"\n[救回带 / {period}]")
        for grp in ("old_keep", "rescue_band"):
            g = sub[sub["group"] == grp]
            if not len(g):
                print(f"  {grp:12s} (无数据)")
                continue
            print(
                f"  {grp:12s} n={g['n'].mean():5.1f}/日 | "
                f"T+3 {g['ret3d'].mean():+.4f} | T+2 {g['ret2d'].mean():+.4f} | T+5 {g['ret5d'].mean():+.4f} | "
                f"上涨率 {g['win3d'].mean():.1%} ({g['win3d5'].mean():.1%} >+0.5%)"
            )
        agg = sub.groupby("group").agg(
            n=("n", "mean"),
            ret3d=("ret3d", "mean"),
            ret2d=("ret2d", "mean"),
            ret5d=("ret5d", "mean"),
            win3d=("win3d", "mean"),
            win3d5=("win3d5", "mean"),
        )
        agg.to_csv(out_dir / f"partA2_rescueband_{period}.csv")
        summary[f"partA2_{period}"] = agg.round(4).to_dict("index")

    # ── Part B: 真实交付并行短名单 (联合门产出) 拆旧/新 ──
    print("\n========== Part B: 交付并行短名单 — 旧门也会留 vs 联合门新增 ==========")
    fs = sorted(
        glob.glob(r"D:/AMINQT/DAILY OPERATION/STOCK LIST/parallel_shortlist_2026*.csv")
    )
    partB = []
    for f in fs:
        df = pd.read_csv(f, dtype={"symbol": str})
        if "date" not in df.columns or "pred_mag_3d" not in df.columns:
            continue
        dc = df["date"].astype(str)
        dts = pd.to_datetime(dc, format="%Y%m%d", errors="coerce")
        if dts.notna().mean() < 0.5:
            dts = pd.to_datetime(dc, errors="coerce")
        df["date"] = dts
        df = df.dropna(subset=["date"])
        df = df.merge(realized, on=["symbol", "date"], how="left")
        old = df[df["pred_mag_3d"] > 0]
        added = df[
            (df["pred_mag_3d"] > T3_FLOOR)
            & (df["pred_mag_3d"] <= 0)
            & (df["pred_mag_2d"] > T2_MIN)
        ]
        for df2, tag in ((old, "old_keep"), (added, "joint_added")):
            if len(df2) == 0:
                continue
            r3 = df2["ret_3d"].dropna()
            r2 = df2["ret_2d"].dropna()
            print(
                f"  {f.split('/')[-1]:46s} {tag:12s} n={len(df2):2d} "
                f"T+3 {r3.mean():+.2%}({len(r3)}有数据) T+2 {r2.mean():+.2%}({len(r2)}有数据) "
                f"标的: {','.join(df2['symbol'])}"
            )
            partB.append(
                {
                    "file": f.split("/")[-1],
                    "tag": tag,
                    "n": len(df2),
                    "ret3d": float(r3.mean()) if len(r3) else np.nan,
                    "ret2d": float(r2.mean()) if len(r2) else np.nan,
                    "syms": ",".join(df2["symbol"]),
                }
            )
    pdB = pd.DataFrame(partB)
    if len(pdB):
        pdB.to_csv(out_dir / "partB_delivered.csv", index=False)
    summary["partB"] = partB

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
