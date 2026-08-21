"""_diag_dual_feature_count_compare.py — dual 特征数扫描结果同窗对比 (2026-08-19).

输入: legacy_hitrate_topn_*.csv (detail 逐日逐票), 按 name=path 键值对传入.
对每候选: dual 板生产闸行 (非 pain_excluded), 按 pred_ret_10d 降序取 top-N,
统计 命中/实得/中位/≥5%/≥10% + 4 个子窗 (按评估日四等分) 的 top10 实得.
判定口径与 phase2_decide 一致: 总赢 + ≥3/4 子窗 + 过闸 (hit>=0.50) 才换 pin.

用法:
  python scripts/_diag_dual_feature_count_compare.py --cand neg200=path_a.csv --cand cand300=path_b.csv --cand candALL=path_c.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import data_others_path


def _topn_stats(sub: pd.DataFrame, n: int) -> dict:
    topn = (
        sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(n)
    )
    r = topn["realized_net"].dropna()
    return {
        "n_days": int(topn["date"].nunique()),
        "picks": int(len(topn)),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
    }


def _subwin_top10(sub: pd.DataFrame, k: int) -> dict:
    """评估日按出现序四等分, 每窗 top10 实得 (与 phase2_decide sub_wins 同口径)."""
    days = pd.to_datetime(sub["date"]).unique()
    n = len(days)
    lo, hi = k * n // 4, (k + 1) * n // 4
    window_days = set(days[lo:hi])
    wsub = sub[pd.to_datetime(sub["date"]).isin(window_days)]
    s = _topn_stats(wsub, 10)
    return {"days": f"{lo}-{hi}", "win_mean": s["mean"], "win_hit": s["hit"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", action="append", required=True, help="name=path 键值对")
    args = ap.parse_args()

    cands = [dict(name=c.split("=")[0], path=c.split("=", 1)[1]) for c in args.cand]
    results = {}
    for c in cands:
        df = pd.read_csv(c["path"], dtype={"symbol": str})
        sub = df[(df["board"] == "dual") & (~df["pain_excluded"].astype(bool))]
        results[c["name"]] = {
            "top5": _topn_stats(sub, 5),
            "top10": _topn_stats(sub, 10),
            "subwin_top10": [_subwin_top10(sub, k) for k in range(4)],
        }

    print(
        f"\n{'cand':<14}{'top10_hit':>10}{'top10_mean':>11}{'top10_med':>10}"
        f"{'ge5':>7}{'ge10':>7}  subwin_means(4q)"
    )
    for name, r in results.items():
        t = r["top10"]
        sw = r["subwin_top10"]
        swm = " ".join(f"{s['win_mean']:+.1%}" for s in sw)
        print(
            f"{name:<14}{t['hit']:>9.1%}{t['mean']:>+10.2%}{t['med']:>+9.2%}"
            f"{t['ge5']:>6.1%}{t['ge10']:>6.1%}  {swm}"
        )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"dual_feature_count_compare_{ts}.json"
    out.write_text(
        json.dumps(
            {"ts": ts, "cands": cands, "results": results},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
