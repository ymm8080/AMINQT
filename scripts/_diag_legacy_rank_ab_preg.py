"""诊断: legacy p_reg 排名 A/B — mag vs mag×p_reg vs 纯 p_reg (2026-08-24)
==================================================================================
数据源: _diag_legacy_hitrate_topn.py 的 125d 回放明细 CSV (当前 20260823r3 bundle),
       每票行含 pred_ret_10d (mag) / prob (p_reg) / base_rate / pain_excluded /
       realized_net (T+10 c2c 净, 已含 0.2% 交易成本).

排名键 (同一生产 E7 池内, pain_excluded=False):
  mag   = pred_ret_10d          (当前生产排名)
  blend = pred_ret_10d × prob   (mag × p_reg 乘积)
  prob  = prob                  (纯 p_reg)

评估: TOP-5/10/15 × realized_net; 窗口 = 全窗/末63d/末31d + 4 子窗 (稳定性).
诊断: 每日 rank IC(键, realized) 判别力 + prob 分布分辨率 + mag-prob 相关性
      (blend 若无增益则说明两键几乎同序, 这是"为什么"的一部分).

用法: python scripts/_diag_legacy_rank_ab_preg.py [--csv <path>]
输出: DATA OTHERS/diag/legacy_rank_ab_preg_<ts>.csv/.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import data_others_path

DEFAULT_CSV = (
    r"D:\AMINQT\DATA OTHERS\diag\legacy_hitrate_topn_20260824_113908.csv"
)
KEYS = ["mag", "blend", "prob"]
COL = {"mag": "pred_ret_10d", "blend": "blend", "prob": "prob"}
DEPTHS = [5, 10, 15]
WINDOWS = [("full", None), ("63d", 63), ("31d", 31)]
SUBWINDOWS = 4


def _stats(topn: pd.DataFrame) -> dict:
    r = topn["realized_net"].dropna()
    n_days = int(topn["date"].nunique())
    return {
        "n_days": n_days,
        "picks": int(len(topn)),
        "avg_picks": float(len(topn) / max(n_days, 1)),
        "hit": float((r > 0).mean()) if len(r) else np.nan,
        "mean": float(r.mean()) if len(r) else np.nan,
        "med": float(r.median()) if len(r) else np.nan,
        "ge5": float((r >= 0.05).mean()) if len(r) else np.nan,
        "ge10": float((r >= 0.10).mean()) if len(r) else np.nan,
    }


def _rank_ic(sub: pd.DataFrame, key: str, real: str = "realized_net") -> float:
    vals = []
    for _d, g in sub.dropna(subset=[key, real]).groupby("date"):
        if len(g) >= 5 and g[real].nunique() > 1:
            r = spearmanr(g[key], g[real])
            if r.statistic == r.statistic:
                vals.append(r.statistic)
    return float(np.mean(vals)) if vals else float("nan")


def _topn(sub: pd.DataFrame, key: str, depth: int) -> pd.DataFrame:
    return (
        sub.sort_values(["date", key], ascending=[True, False])
        .groupby("date", sort=False)
        .head(depth)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV, help="125d 回放明细 CSV 路径")
    args = ap.parse_args()
    t0 = time.time()

    df = pd.read_csv(args.csv, parse_dates=["date"])
    df["prob"] = df["prob"].astype(float)
    print(
        f"[load] {len(df):,}r days={df['date'].nunique()} "
        f"{df['date'].min().date()}..{df['date'].max().date()} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)

    rows: list[dict] = []
    diag: dict = {}
    verdict: list[dict] = []

    for board in ("main", "dual"):
        pool = df[(df["board"] == board) & (~df["pain_excluded"])].copy()
        pool["blend"] = pool["pred_ret_10d"] * pool["prob"]
        dates = sorted(pool["date"].unique())
        n_days = len(dates)
        print(
            f"\n===== {board} | 生产 E7 池 {n_days} 已实现日 ===== ({time.time() - t0:.0f}s)",
            flush=True,
        )
        print(
            f"  {'键×深度':<14}{'窗口':<6}{'票/日':>6} {'命中':>7} {'实得':>8} "
            f"{'中位':>8} {'≥5%':>7} {'≥10%':>7}",
            flush=True,
        )

        # 诊断: 每日 rank IC + prob 分辨率 + mag-prob 相关
        ic = {k: _rank_ic(pool, COL[k]) for k in KEYS}
        corr = float(pool["pred_ret_10d"].corr(pool["prob"]))
        diag[board] = {
            "n_days": n_days,
            "pool_rows": int(len(pool)),
            "rank_ic": ic,
            "prob_min": float(pool["prob"].min()),
            "prob_max": float(pool["prob"].max()),
            "prob_std": float(pool["prob"].std()),
            "mag_prob_corr": corr,
        }
        print(
            f"  [diag] rank_IC mag={ic['mag']:+.4f} blend={ic['blend']:+.4f} "
            f"prob={ic['prob']:+.4f} | prob std={pool['prob'].std():.4f} "
            f"| corr(mag,prob)={corr:+.3f}",
            flush=True,
        )

        for key in KEYS:
            for depth in DEPTHS:
                for wname, w in WINDOWS:
                    dd = dates[-w:] if w else dates
                    sub = pool[pool["date"].isin(dd)]
                    s = _stats(_topn(sub, COL[key], depth))
                    s["board"] = board
                    s["key"] = key
                    s["depth"] = depth
                    s["window"] = wname
                    rows.append(s)
                    if wname == "full":
                        print(
                            f"  {key:<9} top-{depth:<3}{wname:<6}{s['avg_picks']:>6.1f} "
                            f"{s['hit']:>7.1%} {s['mean']:>+8.2%} {s['med']:>+8.2%} "
                            f"{s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                            flush=True,
                        )

        # 4 子窗稳定性 (top-10)
        chunks = np.array_split(np.asarray(dates), SUBWINDOWS)
        sub_rows: list[dict] = []
        for key in KEYS:
            for ci, ch in enumerate(chunks):
                sub = pool[pool["date"].isin(list(ch))]
                s = _stats(_topn(sub, COL[key], 10))
                s.update(board=board, key=key, subwindow=f"q{ci + 1}")
                sub_rows.append(s)
                print(
                    f"  [sub] {key:<9} top-10 q{ci + 1} n={s['n_days']:>3} "
                    f"hit={s['hit']:>7.1%} mean={s['mean']:>+8.2%}",
                    flush=True,
                )
        subd = pd.DataFrame(sub_rows)
        subd.to_csv(out_dir / f"legacy_rank_ab_preg_sub_{board}_{ts}.csv", index=False)

        # 判词: 全窗 top-10 每键均值 → 胜者 + 子窗符号稳定性
        full10 = pd.DataFrame(
            [_s for _s in rows if _s["board"] == board and _s["depth"] == 10 and _s["window"] == "full"]
        ).set_index("key")
        best = full10["mean"].idxmax()
        sub_mean = (
            subd.groupby("key")["mean"].apply(lambda x: (x > 0).sum())
        )
        verdict.append(
            {
                "board": board,
                "best_key": str(best),
                "mag_top10": float(full10.loc["mag", "mean"]),
                "blend_top10": float(full10.loc["blend", "mean"]),
                "prob_top10": float(full10.loc["prob", "mean"]),
                "delta_blend_vs_mag": float(full10.loc["blend", "mean"] - full10.loc["mag", "mean"]),
                "subwin_positive_count": {k: int(v) for k, v in sub_mean.items()},
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"legacy_rank_ab_preg_{ts}.csv", index=False)
    report = {
        "ts": ts,
        "source_csv": args.csv,
        "diag": diag,
        "verdict": verdict,
    }
    (out_dir / f"legacy_rank_ab_preg_{ts}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(f"\n[saved] {out_dir}/legacy_rank_ab_preg_{ts}.csv/.json", flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
