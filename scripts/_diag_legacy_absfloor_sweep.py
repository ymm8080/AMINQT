"""诊断: legacy E7 绝对概率地板 (regime-adaptive cash control, 2026-08-24)
================================================================================
背景: p_reg 是 pred_ret_10d 的单调变换 (corr 0.99) → 排名/仓位/混合通道封闭.
     唯一剩余杠杆是"校准让绝对概率可作门槛" (旧 p_cls 常数 ~0.58, 绝对地板退化).
     相对闸 `prob > base_rate + margin` 在坏 regime 日反而放宽 (base_rate 跌 → 门槛跌);
     绝对地板 `prob >= FLOOR` 反过来: 市场平均概率低的日子强制缩清单 → 现金/选择性.

本脚本在 final p_reg 的 125d 回放明细上, 于当前生产闸 (main +0.08 / dual +0.10)
之上叠加绝对地板扫网格: 看 top-10 实得是否提高、坏日子是否变少、容量是否被破坏.

数据源: 明细只含通过旧生产闸 (main 0.0 / dual 0.08) 的行 → 新 margin 是收紧子集,
     绝对地板 ≥ 0.50 同样收紧 → 子集关系成立, 可安全叠加.

评估: 地板网格 × TOP-10 × 全窗 + 4 子窗 + 容量 (票/日, <10 票天数).
用法: python scripts/_diag_legacy_absfloor_sweep.py
输出: DATA OTHERS/diag/legacy_absfloor_sweep_<ts>.csv/.json
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

DEFAULT_CSV = (
    r"D:\AMINQT\DATA OTHERS\diag\legacy_hitrate_topn_20260824_113908.csv"
)
CUR_MARGIN = {"main": 0.08, "dual": 0.10}  # 2026-08-24 落地生产
ABS_FLOORS = [0.00, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65]
SUBWINDOWS = 4


def _stats(topn: pd.DataFrame) -> dict:
    r = topn["realized_net"].dropna()
    n_days = int(topn["date"].nunique())
    return {
        "n_days": n_days,
        "picks": int(len(topn)),
        "avg_picks": float(len(topn) / max(n_days, 1)),
        "n_days_lt10": int(
            topn.groupby("date").size().lt(10).sum()
        ),
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
        m = CUR_MARGIN[board]
        # 先套当前生产 margin, 再叠加绝对地板
        gated = base[base["prob"] > base["base_rate"] + m]
        print(
            f"\n===== {board} | margin={m} | {len(dates)} 已实现日 ===== "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        print(f"  prob 分位: " + " ".join(
            f"{q:.2f}" for q in gated["prob"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        ), flush=True)
        print(
            f"  {'地板':<6}{'票/日':>7}{'<10票日':>8}{'命中':>7}{'实得':>8}"
            f"{'中位':>8}{'≥5%':>7}{'≥10%':>7}",
            flush=True,
        )

        chunks = np.array_split(np.asarray(dates), SUBWINDOWS)
        sub_rows: list[dict] = []
        for fl in ABS_FLOORS:
            v = gated[gated["prob"] >= fl] if fl > 0 else gated
            topn = (
                v.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                .groupby("date", sort=False)
                .head(10)
            )
            s = _stats(topn)
            s.update(board=board, floor=fl)
            rows.append(s)
            print(
                f"  {fl:<6.2f}{s['avg_picks']:>7.1f}{s['n_days_lt10']:>8}"
                f"{s['hit']:>7.1%}{s['mean']:>+8.2%}{s['med']:>+8.2%}"
                f"{s['ge5']:>7.1%}{s['ge10']:>7.1%}",
                flush=True,
            )
            for ci, ch in enumerate(chunks):
                sub = v[v["date"].isin(list(ch))]
                st = (
                    sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(10)
                )
                ss = _stats(st)
                ss.update(board=board, floor=fl, subwindow=f"q{ci + 1}")
                sub_rows.append(ss)
        subd = pd.DataFrame(sub_rows)
        subd.to_csv(
            out_dir / f"legacy_absfloor_sweep_sub_{board}_{ts}.csv", index=False
        )

        # 判词: 全窗 top-10 实得最高地板, 以子窗正数数优先; 容量破坏 (<10票) 否决
        full10 = pd.DataFrame(
            [_s for _s in rows if _s["board"] == board]
        ).assign(
            positive_sub=full10_positive(subd, board),
        )
        cand = full10[full10["n_days_lt10"] <= full10["n_days_lt10"].iloc[0]]
        best = cand.sort_values(
            ["positive_sub", "mean"], ascending=[False, False]
        ).iloc[0]
        verdict.append(
            {
                "board": board,
                "margin": m,
                "best_floor": float(best["floor"]),
                "best_mean_top10": float(best["mean"]),
                "best_hit_top10": float(best["hit"]),
                "best_avg_picks": float(best["avg_picks"]),
                "best_n_days_lt10": int(best["n_days_lt10"]),
                "best_positive_sub": int(best["positive_sub"]),
                "base_mean_top10": float(full10.loc[full10["floor"] == 0.0, "mean"].iloc[0]),
                "delta": float(best["mean"]) - float(
                    full10.loc[full10["floor"] == 0.0, "mean"].iloc[0]
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"legacy_absfloor_sweep_{ts}.csv", index=False)
    (out_dir / f"legacy_absfloor_sweep_{ts}.json").write_text(
        json.dumps({"ts": ts, "source_csv": DEFAULT_CSV, "verdict": verdict},
                   ensure_ascii=False, indent=2)
    )
    print(f"\n[saved] {out_dir}/legacy_absfloor_sweep_{ts}.csv/.json", flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)
    return 0


def full10_positive(subd: pd.DataFrame, board: str) -> list[int]:
    return [
        int((subd[(subd["floor"] == fl) & (subd["board"] == board)]["mean"] > 0).sum())
        for fl in ABS_FLOORS
    ]


if __name__ == "__main__":
    raise SystemExit(main())
