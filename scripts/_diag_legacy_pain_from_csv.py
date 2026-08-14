"""_diag_legacy_pain_from_csv.py — 离线补算 legacy 全量诊断的 pain 变体 + 排名分解 (2026-08-14).

背景: 052011 全量 run (旧版脚本, 无 pain 变体) 的 CSV 已落盘, 逐行带 pain_prob
(基准闸已拦 pain>0.5, 故只能验"更严"方向; 关闸方向待新版脚本下次全量跑)。

补算:
  1) pain 更严档 top-5: pain≤0.4 / ≤0.3 (与生产 ≤0.5 对比) — 命中/实得/≥5%/≥10%.
  2) pred_ret_10d 排名分解: top-5 vs 6-15 段 (top15 命中>top5 的异常是否来自 6-15 段).
  3) 首 20 个评估日 vs 其余 (base_rate 预热是否含在本次 run 内的旁证).

WORM: DATA_OTHERS/diag/legacy_pain_from_csv_<ts>.json

用法: python scripts/_diag_legacy_pain_from_csv.py [csv路径]
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


def _stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {"n_days": 0, "picks": 0, "avg_picks": 0.0, "hit": float("nan"),
                "mean": float("nan"), "med": float("nan"),
                "ge5": float("nan"), "ge10": float("nan")}
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


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(
        glob.glob(str(data_others_path("diag") / "legacy_hitrate_topn_*.csv"))
    )[-1]
    print(f"[load] {path}", flush=True)
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    print(f"[rows] {len(df):,} | dates {df['date'].nunique()} | boards {sorted(df['board'].unique())}", flush=True)

    rows: list[dict] = []
    # 排名键变体: 过闸集内换排序键 (rk6_15 > rk1_5 提示 pred_ret_10d 排序无效, 08-14)
    RANK_KEYS = {
        "ret10": ("pred_ret_10d", False),        # 生产现状
        "prob": ("prob", False),
        "prob_over_base": ("prob_over_base", False),
        "pain": ("pain_prob", True),             # 低痛苦优先
        "pain_then_prob": ("prob", False),       # 先在 pain≤0.4 内按 prob 排 (组合)
    }
    WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
    for board in ("main", "dual"):
        b = df[df["board"] == board].copy()
        if b.empty:
            rows.append({"board": board, "variant": "empty", **{k: float("nan") for k in
                        ("n_days", "picks", "avg_picks", "hit", "mean", "med", "ge5", "ge10")}})
            continue
        b["prob_over_base"] = b["prob"] - b["base_rate"]
        dates_all = np.sort(b["date"].unique())
        # 1) pain 更严档 top-5 (基准行已 pain≤0.5)
        for pt in (0.5, 0.4, 0.3):
            v = b[b["pain_prob"].fillna(0) <= pt]
            top5 = (v.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                     .groupby("date", sort=False).head(5))
            s = _stats(top5)
            rows.append({"board": board, "variant": f"pain_le_{pt:.1f}_top5", **s})
        # 2) pred_ret_10d 排名分解: top-5 vs 6-15 段 (每个日期内按 pred_ret_10d 降序)
        ranked = b.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
        ranked["rk"] = ranked.groupby("date", sort=False).cumcount() + 1
        for seg, mask in (("rk1_5", ranked["rk"] <= 5), ("rk6_15", (ranked["rk"] > 5) & (ranked["rk"] <= 15))):
            s = _stats(ranked[mask])
            rows.append({"board": board, "variant": seg, **s})
        # 3) 排名键变体 × 窗口 (全窗 / 近126d / 近63d, top-5 + top-10)
        for kname, (kcol, asc) in RANK_KEYS.items():
            src = b
            if kname == "pain_then_prob":
                src = b[b["pain_prob"].fillna(0) <= 0.4]
            for depth in (5, 10):
                for wname, wdays in WINDOWS.items():
                    cutoff = dates_all[0] if wdays >= len(dates_all) else dates_all[-wdays]
                    w = src[src["date"].values >= cutoff]
                    topn = (w.sort_values(["date", kcol], ascending=[True, asc])
                             .groupby("date", sort=False).head(depth))
                    s = _stats(topn)
                    rows.append({"board": board, "variant": f"{kname}_top{depth}_{wname}", **s})
        # 4) 首 20 评估日 vs 其余 (base_rate 预热旁证)
        ds = sorted(b["date"].unique())
        for seg, sub in (("first20d", b[b["date"].isin(ds[:20])]), ("rest", b[b["date"].isin(ds[20:])])):
            top5 = (sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                     .groupby("date", sort=False).head(5))
            s = _stats(top5)
            rows.append({"board": board, "variant": seg, **s})

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    (out_dir / f"legacy_pain_from_csv_{ts}.json").write_text(
        json.dumps({"ts": ts, "source": os.path.basename(path), "rows": rows},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    def _sf(v, fmt):
        return fmt.format(v) if not (isinstance(v, float) and np.isnan(v)) else "-" * len(fmt.format(0.0))

    print(f"\n{'板':<4}{'变体':<18}{'日':>4}{'票':>5}{'票/日':>6} {'命中':>7}{'实得':>8}{'中位':>8}{'≥5%':>7}{'≥10%':>7}", flush=True)
    for r in rows:
        nd = int(r["n_days"]) if r["n_days"] == r["n_days"] else 0
        pk = int(r["picks"]) if r["picks"] == r["picks"] else 0
        ap = r["avg_picks"] if r["avg_picks"] == r["avg_picks"] else 0.0
        print(f"{r['board']:<4}{r['variant']:<18}{nd:>4}{pk:>5}{ap:>6.1f} "
              f"{_sf(r['hit'], '{:.1%}')}{_sf(r['mean'], '{:+.2%}')}{_sf(r['med'], '{:+.2%}')}"
              f"{_sf(r['ge5'], '{:.1%}')}{_sf(r['ge10'], '{:.1%}')}", flush=True)
    print(f"\n[saved] {out_dir}/legacy_pain_from_csv_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
