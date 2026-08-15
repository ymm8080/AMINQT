"""_diag_legacy_rank_ab.py — legacy 排名键 A/B: mag vs prob vs mag×prob (2026-08-15).

问题: 并行 PR 1aaa380e 排名键 blend=pred_mag_10d×pred_prob 250d A/B 全窗赢 →
      legacy 是否跟进 (pred_ret_10d → pred_ret_10d × prob)?
本脚本在 legacy 250d 重放 CSV (DATA_OTHERS/diag/legacy_hitrate_topn_*.csv,
与 _diag_legacy_hitrate_topn / _diag_legacy_combo_gates 同源, 生产 bundle 重放) 上,
按生产顺序 (E7 边际闸 dual +0.08 → pain≤0.4 → 排名) 比较 4 个排名键:
  mag      = pred_ret_10d 降序 (生产现状)
  prob     = prob 降序 (参照: 并行纯 prob = Platt 式死法)
  blend    = pred_ret_10d × prob 降序 (并行定案)
  blend_ex = pred_ret_10d × (prob − base_rate) 降序 (超额概率)
指标: TOP-5/10/15 × full/126d/63d 命中/实得/中位/≥5%/≥10% + 4 等分逐票子窗.
WORM: DATA_OTHERS/diag/legacy_rank_ab_<ts>.csv/.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config.settings import data_others_path

PROB_MARGIN = 0.08  # landed dual 边际 (de8790fa, main 分支生产)
PAIN_MAX = 0.4  # landed dual pain 上限 (de8790fa)
DEPTHS = (5, 10, 15)  # 15 = legacy 生产清单深度 TOP_N
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
N_SUB = 4  # 等分子窗 (稳定性, 同并行 A/B)


def _stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "n_days": 0,
            "picks": 0,
            "avg_picks": 0.0,
            "hit": float("nan"),
            "mean": float("nan"),
            "med": float("nan"),
            "ge5": float("nan"),
            "ge10": float("nan"),
        }
    r = sub["realized_net"].dropna()
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
    }


def _sub_windows(top: pd.DataFrame) -> list[dict]:
    dates = np.sort(top["date"].unique())
    step = max(1, len(dates) // N_SUB)
    subs = []
    for i in range(N_SUB):
        s0, s1 = i * step, len(dates) if i == N_SUB - 1 else (i + 1) * step
        seg = top[top["date"].isin(dates[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{N_SUB}",
                "rows": int(len(seg)),
                "hit": float((seg["realized_net"] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean": float(seg["realized_net"].mean())
                if len(seg)
                else float("nan"),
            }
        )
    return subs


def main() -> int:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(
            glob.glob(str(data_others_path("diag") / "legacy_hitrate_topn_*.csv"))
        )[-1]
    )
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    b = df[df["board"] == "dual"].copy()
    dates_all = np.sort(b["date"].unique())

    # ---- 生产顺序: E7 边际闸 +0.08 → pain≤0.4 (landed) ----
    pool = b[
        (b["prob"] - b["base_rate"] > PROB_MARGIN)
        & (b["pain_prob"].fillna(0) <= PAIN_MAX)
    ].copy()
    print(
        f"[pool] {len(pool)} 票 / {pool['date'].nunique()} 出票日 "
        f"(源 CSV: {os.path.basename(path)}, {len(b)} 票 / {b['date'].nunique()} 日)",
        flush=True,
    )
    n_uniq = int(pool["prob"].nunique())
    print(
        f"[prob] 池内唯一概率值 {n_uniq} 个, 众数占比 "
        f"{pool['prob'].value_counts().iloc[0] / len(pool):.0%}",
        flush=True,
    )
    pool["rank_mag"] = pool["pred_ret_10d"]
    pool["rank_prob"] = pool["prob"]
    pool["rank_blend"] = pool["pred_ret_10d"] * pool["prob"]
    pool["rank_blend_ex"] = pool["pred_ret_10d"] * (pool["prob"] - pool["base_rate"])

    rows: list[dict] = []
    for rkname, rkcol in (
        ("mag", "rank_mag"),
        ("prob", "rank_prob"),
        ("blend", "rank_blend"),
        ("blend_ex", "rank_blend_ex"),
    ):
        for depth in DEPTHS:
            for wname, wdays in WINDOWS.items():
                cutoff = dates_all[0] if wdays >= len(dates_all) else dates_all[-wdays]
                w = pool[pool["date"].values >= cutoff]
                top = (
                    w.sort_values(["date", rkcol], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(depth)
                )
                s = _stats(top)
                rows.append(
                    {"rank": rkname, "depth": depth, "window": wname, **s}
                )
                subs = _sub_windows(top)
                sub_s = "  ".join(
                    f"{x['win']}:{x['hit']:.0%}/{x['mean']:+.2%}" for x in subs
                )
                print(
                    f"[{rkname:>9}/top{depth:>2}/{wname:>4}] "
                    f"日{s['n_days']:>3} 票/日{s['avg_picks']:>5.2f} "
                    f"命中{s['hit']:>6.1%} 实得{s['mean']:>+8.2%} "
                    f"中位{s['med']:>+8.2%} ≥5%{s['ge5']:>6.1%} ≥10%{s['ge10']:>6.1%}",
                    flush=True,
                )
                print(f"    sub: {sub_s}", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    pd.DataFrame(rows).to_csv(out_dir / f"legacy_rank_ab_{ts}.csv", index=False)
    (out_dir / f"legacy_rank_ab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "source": os.path.basename(path),
                "gate": f"dual pob>+{PROB_MARGIN:.2f} & pain<={PAIN_MAX} (生产 landed)",
                "rank_keys": (
                    "mag=pred_ret_10d(生产); prob; blend=mag×prob; "
                    "blend_ex=mag×(prob−base_rate)"
                ),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_rank_ab_{ts}.csv/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
