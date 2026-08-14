"""_diag_legacy_combo_gates.py — legacy 双创 组合闸离线补算 (2026-08-14).

052011 全量 run 的 CSV 行 = 旧基准闸 (prob>base_rate, 无边际). 另一会话已落
LEGACY_ENTRY_GATE.prob_margin dual=0.08. 本脚本在 CSV 上补算组合:
  A) margin 0.08 单独 (复现另一会话 66.3% 结论)
  B) pain≤0.4 单独
  C) margin 0.08 + pain≤0.4 叠加
  D) C + 排名键 prob 降序 (top-5 / top-10)
  每变体 × full/126d/63d 子窗 (稳定性), 出票日/票量权衡.

WORM: DATA_OTHERS/diag/legacy_combo_gates_<ts>.json
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

WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}


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
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    b = df[df["board"] == "dual"].copy()
    b["pob"] = b["prob"] - b["base_rate"]
    dates_all = np.sort(b["date"].unique())

    gates = {
        "A_m008": b["pob"] > 0.08,
        "B_pain04": b["pain_prob"].fillna(0) <= 0.4,
        "C_m008_pain04": (b["pob"] > 0.08) & (b["pain_prob"].fillna(0) <= 0.4),
    }
    rows: list[dict] = []
    for gname, mask in gates.items():
        src = b[mask]
        for rkname, rkcol, asc in (("ret10", "pred_ret_10d", False), ("prob", "prob", False)):
            for depth in (5, 10):
                for wname, wdays in WINDOWS.items():
                    cutoff = dates_all[0] if wdays >= len(dates_all) else dates_all[-wdays]
                    w = src[src["date"].values >= cutoff]
                    topn = (w.sort_values(["date", rkcol], ascending=[True, asc])
                             .groupby("date", sort=False).head(depth))
                    s = _stats(topn)
                    rows.append({"gate": gname, "rank": rkname, "depth": depth,
                                 "window": wname, **s})
                    print(f"[{gname}/{rkname}/top{depth}/{wname}] "
                          f"日{s['n_days']} 票{s['picks']} 命中{s['hit']:.1%} 实得{s['mean']:+.2%}", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    (out_dir / f"legacy_combo_gates_{ts}.json").write_text(
        json.dumps({"ts": ts, "source": os.path.basename(path), "rows": rows},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[saved] {out_dir}/legacy_combo_gates_{ts}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
