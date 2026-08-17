"""_diag_legacy_rank_ab_newprob.py — legacy 排名键 A/B: 新 GBM 概率头 (2026-08-16).

背景: 08-15 旧 cls 概率头 (22 唯一值/60% 同值) 使 blend=mag×prob 无区分度被否决
(legacy-blend-rank-verdict). 08-16 新并行式 GBM 概率头已训练+接线 (41eff001),
本脚本用 _diag_legacy_prob_head_replay.py 的 walk-forward 250d 预测 (WORM CSV,
含 pred_prob_new + base_prod 富化列) 重跑排名键矩阵:

  mag      = pred_ret_10d 降序 (生产现状)
  blend_new= pred_ret_10d × pred_prob_new (并行定案同款, 待验证)
  blend_old= pred_ret_10d × prob_up (旧头, 已证伪参照)
  prob_new / prob_old = 纯概率排名 (参照: 并行 Platt 式死法)
  blend_ex = pred_ret_10d × (pred_prob_new − base_prod)

池:
  P1 = E7 闸池 (entry_filter 通过, pain_excluded=False)
  P2 = 生产闸池 = P1 + 新头概率闸 pred_prob_new > base_prod + 0.08 (fail-open NaN)
       (41eff001 生产链同款)
指标: TOP-5/10/15 × full/126d/63d 命中/实得/中位/≥5%/≥10% + 4 子窗.
区分度: 池内旧 vs 新概率 IQR/唯一值/众数占比 + 逐日 Spearman (旧头≈0.98 无区分).
WORM: DATA_OTHERS/diag/legacy_rank_ab_newprob_<ts>.csv/.json

用法:
  python scripts/_diag_legacy_rank_ab_newprob.py [replay_csv_path]
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

PROB_MARGIN = 0.08  # LEGACY_PROB_GATE margin (landed 41eff001)
DEPTHS = (5, 10, 15)
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
N_SUB = 4
KEYS = (
    ("mag", "pred_ret_10d"),
    ("blend_new", "blend_new"),
    ("blend_old", "blend_old"),
    ("prob_new", "pred_prob_new"),
    ("prob_old", "prob"),
    ("blend_ex", "blend_ex"),
)


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
                "mean": float(seg["realized_net"].mean()) if len(seg) else float("nan"),
            }
        )
    return subs


def _prob_diagnostics(pool: pd.DataFrame) -> dict:
    out = {}
    for tag, col in (("old", "prob"), ("new", "pred_prob_new")):
        v = pool[col].dropna()
        if v.empty:
            out[tag] = {"n": 0}
            continue
        iqr = float(v.quantile(0.75) - v.quantile(0.25))
        mode_share = float(v.value_counts().iloc[0] / len(v))
        out[tag] = {
            "n": int(len(v)),
            "n_unique": int(v.nunique()),
            "mode_share": mode_share,
            "q25": float(v.quantile(0.25)),
            "q50": float(v.quantile(0.50)),
            "q75": float(v.quantile(0.75)),
            "iqr": iqr,
            "mean": float(v.mean()),
            "std": float(v.std()),
        }
    # 逐日 Spearman: 旧头无区分度的机制指标 (08-15 ≈0.98)
    spears = []
    for _, day in pool.groupby("date"):
        sub = day[["prob", "pred_prob_new"]].dropna()
        if (
            len(sub) >= 5
            and sub["prob"].nunique() > 1
            and sub["pred_prob_new"].nunique() > 1
        ):
            spears.append(sub.corr(method="spearman").iloc[0, 1])
    out["spearman_daily_mean"] = float(np.mean(spears)) if spears else float("nan")
    out["spearman_daily_n"] = len(spears)
    return out


def main() -> int:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(
            glob.glob(str(data_others_path("diag") / "legacy_prob_head_replay_*.csv"))
        )[-1]
    )
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    print(
        f"[load] {os.path.basename(path)}: {len(df):,} 票 / "
        f"{df['date'].nunique()} 日 (boards: {sorted(df['board'].unique())})",
        flush=True,
    )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    rows: list[dict] = []
    diag: dict = {}

    for board in ("main", "dual"):
        b = df[df["board"] == board].copy()
        p1 = b[~b["pain_excluded"].fillna(False)].copy()
        p1["blend_old"] = p1["pred_ret_10d"] * p1["prob"]
        p1["blend_new"] = p1["pred_ret_10d"] * p1["pred_prob_new"]
        p1["blend_ex"] = p1["pred_ret_10d"] * (p1["pred_prob_new"] - p1["base_prod"])
        keep_new = (
            (p1["pred_prob_new"] > p1["base_prod"] + PROB_MARGIN)
            | p1["pred_prob_new"].isna()
            | p1["base_prod"].isna()
        )
        p2 = p1[keep_new].copy()
        diag[board] = {
            "n_days_e7": int(p1["date"].nunique()),
            "n_days_prod": int(p2["date"].nunique()),
            "picks_e7": int(len(p1)),
            "picks_prod": int(len(p2)),
            "prob_e7": _prob_diagnostics(p1),
            "prob_prod": _prob_diagnostics(p2),
        }
        print(
            f"\n===== {board}: E7 池 {len(p1)} 票/{p1['date'].nunique()} 日 | "
            f"生产闸池 {len(p2)} 票/{p2['date'].nunique()} 日 =====",
            flush=True,
        )
        old = diag[board]["prob_e7"].get("old", {})
        new = diag[board]["prob_e7"].get("new", {})
        if new:
            print(
                f"[区分度 E7 池] 旧头: 唯一值 {old.get('n_unique')} / 众数 {old.get('mode_share', 0):.0%} / "
                f"IQR {old.get('iqr', 0):.3f} | 新头: 唯一值 {new.get('n_unique')} / 众数 {new.get('mode_share', 0):.0%} / "
                f"IQR {new.get('iqr', 0):.3f} | 逐日 Spearman {diag[board]['prob_e7'].get('spearman_daily_mean', float('nan')):.3f}",
                flush=True,
            )

        for pool_name, pool in (("E7池", p1), ("生产闸池", p2)):
            dates_all = np.sort(pool["date"].unique())
            for rkname, rkcol in KEYS:
                for depth in DEPTHS:
                    for wname, wdays in WINDOWS.items():
                        cutoff = (
                            dates_all[0]
                            if wdays >= len(dates_all)
                            else dates_all[-wdays]
                        )
                        w = pool[pool["date"].values >= cutoff]
                        top = (
                            w.sort_values(["date", rkcol], ascending=[True, False])
                            .groupby("date", sort=False)
                            .head(depth)
                        )
                        s = _stats(top)
                        rows.append(
                            {
                                "board": board,
                                "pool": pool_name,
                                "rank": rkname,
                                "depth": depth,
                                "window": wname,
                                **s,
                            }
                        )
                        sub_s = "  ".join(
                            f"{x['win']}:{x['hit']:.0%}/{x['mean']:+.2%}"
                            for x in _sub_windows(top)
                        )
                        print(
                            f"[{rkname:>9}/top{depth:>2}/{wname:>4}] "
                            f"日{s['n_days']:>3} 票/日{s['avg_picks']:>5.2f} "
                            f"命中{s['hit']:>6.1%} 实得{s['mean']:>+8.2%} "
                            f"中位{s['med']:>+8.2%} ≥5%{s['ge5']:>6.1%} ≥10%{s['ge10']:>6.1%}",
                            flush=True,
                        )
                        print(f"    sub: {sub_s}", flush=True)

    pd.DataFrame(rows).to_csv(out_dir / f"legacy_rank_ab_newprob_{ts}.csv", index=False)
    (out_dir / f"legacy_rank_ab_newprob_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "source": os.path.basename(path),
                "gate": (
                    "E7池=entry_filter 通过; 生产闸池=E7 + 新头 prob>base_prod+0.08 "
                    "(fail-open NaN, 41eff001 生产链)"
                ),
                "rank_keys": (
                    "mag=pred_ret_10d(生产现状); blend_new=mag×pred_prob_new; "
                    "blend_old=mag×prob_up(旧头, 已证伪); prob_new/prob_old=纯概率; "
                    "blend_ex=mag×(pred_prob_new−base_prod)"
                ),
                "prob_diagnostics": diag,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_rank_ab_newprob_{ts}.csv/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
