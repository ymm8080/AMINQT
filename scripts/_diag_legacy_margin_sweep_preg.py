"""诊断: legacy E7 prob_margin 闸重扫 (final p_reg, 2026-08-24)
==================================================================================
背景: p_reg 是 pred_ret_10d 的单调变换 (corr 0.99), 排名键无法改变 Top-N (A/B 已证).
     p_reg 的价值只能通过 E7 闸 `prob > base_rate + margin` 兑现: 收紧 margin →
     池子更小但更纯. 本脚本在 final p_reg 的 125d 回放明细上扫 margin 网格,
     找 top-10 实得最高且子窗稳定的工作点.

数据源: 同 _diag_legacy_rank_ab_preg.py (08-24 125d 回放明细, 当前 bundle).
     注意: 明细只含通过生产闸 (main margin 0.0 / dual 0.08) 的行,
     故 main 可扫 margin ≥ 0, dual 只能扫 ≥ 0.08 (收紧方向, 正是要测的方向).

评估: margin 网格 × TOP-5/10/15 × 全窗/末63d/末31d + 4 子窗 top-10.
     额外记录 avg_picks (池缩水程度) + 非空日数, 区分"更少但更纯" vs "池子不够填".

用法: python scripts/_diag_legacy_margin_sweep_preg.py
输出: DATA OTHERS/diag/legacy_margin_sweep_preg_<ts>.csv/.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import data_others_path

DEFAULT_CSV = r"D:\AMINQT\DATA OTHERS\diag\legacy_hitrate_topn_20260824_113908.csv"
PROD_MARGIN = {"main": 0.0, "dual": 0.08}
MARGINS_MAIN = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
MARGINS_DUAL = [0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
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


def main() -> int:
    t0 = time.time()
    df = pd.read_csv(DEFAULT_CSV, parse_dates=["date"])
    df["prob"] = df["prob"].astype(float)
    print(
        f"[load] {len(df):,}r days={df['date'].nunique()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)

    rows: list[dict] = []
    verdict: list[dict] = []

    for board in ("main", "dual"):
        base = df[(df["board"] == board) & (~df["pain_excluded"])].copy()
        dates = sorted(base["date"].unique())
        margins = MARGINS_MAIN if board == "main" else MARGINS_DUAL
        print(
            f"\n===== {board} | 生产 margin={PROD_MARGIN[board]} | "
            f"{len(dates)} 已实现日 ===== ({time.time() - t0:.0f}s)",
            flush=True,
        )
        print(
            f"  {'margin':<7}{'深度':<5}{'窗口':<6}{'票/日':>6} {'命中':>7} {'实得':>8} "
            f"{'中位':>8} {'≥5%':>7} {'≥10%':>7}",
            flush=True,
        )

        # 池大小随 margin 缩水 (全窗口径, 仅看能过闸的行数)
        pool_sizes = {}
        for m in margins:
            v = base[base["prob"] > base["base_rate"] + m]
            pool_sizes[m] = {
                "rows": int(len(v)),
                "avg_day": float(len(v) / len(dates)) if dates else 0.0,
                "n_nonempty_days": int(v["date"].nunique()),
            }
        print(
            "  [pool] "
            + "  ".join(f"{m:.2f}→{pool_sizes[m]['avg_day']:.1f}/日" for m in margins),
            flush=True,
        )

        for m in margins:
            v = base[base["prob"] > base["base_rate"] + m]
            for depth in DEPTHS:
                for wname, w in WINDOWS:
                    dd = dates[-w:] if w else dates
                    sub = v[v["date"].isin(dd)]
                    topn = (
                        sub.sort_values(
                            ["date", "pred_ret_10d"], ascending=[True, False]
                        )
                        .groupby("date", sort=False)
                        .head(depth)
                    )
                    s = _stats(topn)
                    s.update(board=board, margin=m, depth=depth, window=wname)
                    rows.append(s)
                    if wname == "full":
                        print(
                            f"  {m:<7.2f} top-{depth:<3}{wname:<6}{s['avg_picks']:>6.1f} "
                            f"{s['hit']:>7.1%} {s['mean']:>+8.2%} {s['med']:>+8.2%} "
                            f"{s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                            flush=True,
                        )

        # 4 子窗 top-10 (稳定性)
        chunks = np.array_split(np.asarray(dates), SUBWINDOWS)
        sub_rows: list[dict] = []
        for m in margins:
            v = base[base["prob"] > base["base_rate"] + m]
            for ci, ch in enumerate(chunks):
                sub = v[v["date"].isin(list(ch))]
                topn = (
                    sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(10)
                )
                s = _stats(topn)
                s.update(board=board, margin=m, subwindow=f"q{ci + 1}")
                sub_rows.append(s)
        subd = pd.DataFrame(sub_rows)
        subd.to_csv(
            out_dir / f"legacy_margin_sweep_preg_sub_{board}_{ts}.csv", index=False
        )

        # 判词: 全窗 top-10 实得最高 margin, 以子窗正数数优先 (稳定>峰值)
        full10 = pd.DataFrame(
            [
                _s
                for _s in rows
                if _s["board"] == board and _s["depth"] == 10 and _s["window"] == "full"
            ]
        )
        pos_count = {m: int((subd[subd["margin"] == m]["mean"] > 0).sum()) for m in margins}
        full10 = full10.assign(positive_sub=full10["margin"].map(pos_count))
        best = full10.sort_values(
            ["positive_sub", "mean"], ascending=[False, False]
        ).iloc[0]
        verdict.append(
            {
                "board": board,
                "prod_margin": PROD_MARGIN[board],
                "best_margin": float(best["margin"]),
                "best_mean_top10": float(best["mean"]),
                "best_hit_top10": float(best["hit"]),
                "best_avg_picks": float(best["avg_picks"]),
                "best_positive_sub": int(best["positive_sub"]),
                "prod_top10_mean": float(
                    full10.loc[full10["margin"] == PROD_MARGIN[board], "mean"].iloc[0]
                ),
                "delta": float(best["mean"])
                - float(
                    full10.loc[full10["margin"] == PROD_MARGIN[board], "mean"].iloc[0]
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"legacy_margin_sweep_preg_{ts}.csv", index=False)
    (out_dir / f"legacy_margin_sweep_preg_{ts}.json").write_text(
        json.dumps(
            {"ts": ts, "source_csv": DEFAULT_CSV, "verdict": verdict},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n[saved] {out_dir}/legacy_margin_sweep_preg_{ts}.csv/.json", flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
