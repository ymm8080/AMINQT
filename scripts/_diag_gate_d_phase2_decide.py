"""_diag_gate_d_phase2_decide.py — Phase 2 四候选 250d OOS 决策表 (2026-08-18).

输入: _diag_gate_d_phase2_replay_runall.py 的 WORM CSV (逐日逐票):
  legacy_hitrate_topn_{pin50,candA,candB,neg200}_*.csv
  (生产闸行 pain_excluded=False; 排序键=pred_ret_10d 降序, 同生产排名)

输出 (WORM):
  DATA_OTHERS/diag/phase2_decide_<ts>.json — 全指标
  DATA_OTHERS/diag/phase2_decide_<ts>.csv  — 可读决策表 (候选 × topN × 窗口)

决策规则 (预注册, 08-18 协议):
  过闸 = top-10 全窗 dual 胜率≥0.50 且 实得≥1.5%
  换 pin ⇔ 候选 总 OOS 实得赢 pin50 + ≥3/4 子窗实得赢 + 过闸; 否则保留 pin.

用法: python scripts/_diag_gate_d_phase2_decide.py [csv1 csv2 ...] (缺省=最新每候选)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config.settings import data_others_path

DIAG = data_others_path("diag")
CANDS = ["pin50", "candA", "candB", "neg200"]
TOPN = [5, 10, 15]
PRIMARY_TOPN = 10
GATE = {"hit": 0.50, "mean": 0.015}
SUBWINDOWS = 4


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


def _resolve_inputs() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for cand in CANDS:
        hits = sorted(glob.glob(str(DIAG / f"legacy_hitrate_topn_{cand}_*.csv")))
        if not hits:
            print(f"[warn] {cand} 无 WORM CSV, 跳过", flush=True)
            continue
        paths[cand] = Path(hits[-1])
        print(f"[load:{cand}] {hits[-1]}", flush=True)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="*", help="显式 CSV 路径 (缺省=最新每候选)")
    ap.add_argument(
        "--base",
        default="pin50",
        help="基线候选名 (默认 pin50; >200 轮传 neg200)",
    )
    args = ap.parse_args()
    base_cand = args.base

    inputs: dict[str, Path] = {}
    if args.csvs:
        import re

        for p in args.csvs:
            stem = Path(p).name
            cand = next((c for c in CANDS if f"_{c}_" in stem), None)
            if cand is None:
                m = re.search(r"legacy_hitrate_topn_(.+?)_\d{8}_\d{6}\.csv", stem)
                cand = m.group(1) if m else Path(p).stem
            inputs[cand] = Path(p)
    else:
        inputs = _resolve_inputs()
    if not inputs:
        print("[FATAL] 无输入 CSV", flush=True)
        return 2

    frames: dict[str, pd.DataFrame] = {}
    for cand, p in inputs.items():
        df = pd.read_csv(p, dtype={"symbol": str})
        df["date"] = pd.to_datetime(df["date"])
        # 生产闸行 (pain_excluded=False); 排序键 pred_ret_10d 降序
        base = df[df["pain_excluded"] == False].copy()  # noqa: E712
        frames[cand] = base.sort_values(
            ["date", "pred_ret_10d"], ascending=[True, False]
        )
        print(
            f"[rows:{cand}] {len(df):,} | 生产闸行 {len(base):,} | "
            f"dates {base['date'].nunique()}",
            flush=True,
        )

    windows: dict[str, np.ndarray] = {}
    ref_dates = frames["pin50"]["date"].unique()
    parts = np.array_split(np.sort(ref_dates), SUBWINDOWS)
    for i, part in enumerate(parts):
        windows[f"w{i + 1}"] = part
    windows["full"] = np.sort(ref_dates)
    print(
        f"[windows] full={len(windows['full'])}d "
        + " ".join(f"{k}={len(v)}d" for k, v in windows.items() if k != "full"),
        flush=True,
    )

    rows: list[dict] = []
    for cand, df in frames.items():
        for topn in TOPN:
            cut = df.groupby("date", sort=False).head(topn)
            for wname, wdates in windows.items():
                s = _stats(cut[cut["date"].isin(wdates)])
                rows.append(
                    {"cand": cand, "topn": topn, "window": wname, **s}
                )

    out = pd.DataFrame(rows)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    DIAG.mkdir(parents=True, exist_ok=True)
    jpath = DIAG / f"phase2_decide_{ts}.json"
    cpath = DIAG / f"phase2_decide_{ts}.csv"
    jpath.write_text(
        json.dumps(
            {
                "ts": ts,
                "inputs": {k: str(v) for k, v in inputs.items()},
                "gate": GATE,
                "decision_rule": "换 pin ⇔ 总OOS实得赢 + ≥3/4子窗实得赢 + 过闸",
                "rows": out.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    out.to_csv(cpath, index=False, encoding="utf-8-sig")
    print(f"[worm] {jpath}", flush=True)
    print(f"[worm] {cpath}", flush=True)

    # ---- 决策表 (top-10 主判) ----
    t10 = out[(out["topn"] == PRIMARY_TOPN)].pivot_table(
        index="cand", columns="window", values=["hit", "mean", "avg_picks"]
    )
    print(f"\n===== 决策表 top-{PRIMARY_TOPN} (实得 mean) =====", flush=True)
    print(
        t10["mean"]
        .reindex(CANDS)
        .round(4)
        .to_string(),
        flush=True,
    )
    print(f"\n===== 命中率 (hit>0) =====", flush=True)
    print(t10["hit"].reindex(CANDS).round(4).to_string(), flush=True)

    base = t10["mean"].loc[base_cand, "full"]
    verdicts = []
    for cand in CANDS:
        if cand == base_cand:
            verdicts.append({"cand": cand, "change": "baseline"})
            continue
        if cand not in frames:
            continue
        full_win = (
            not np.isnan(t10["mean"].loc[cand, "full"])
            and t10["mean"].loc[cand, "full"] > base
        )
        sub_wins = sum(
            t10["mean"].loc[cand, f"w{i}"] > t10["mean"].loc[base_cand, f"w{i}"]
            for i in range(1, SUBWINDOWS + 1)
        )
        hit_ok = t10["hit"].loc[cand, "full"] >= GATE["hit"]
        mean_ok = t10["mean"].loc[cand, "full"] >= GATE["mean"]
        gate_ok = hit_ok and mean_ok
        change = full_win and sub_wins >= 3 and gate_ok
        full_mean = round(float(t10["mean"].loc[cand, "full"]), 4)
        verdicts.append(
            {
                "cand": cand,
                "change": "换 pin" if change else "保留 pin",
                "full_mean": full_mean,
                "base_mean": round(float(base), 4),
                "full_win": bool(full_win),
                "sub_wins": int(sub_wins),
                "gate_ok": bool(gate_ok),
                "hit": round(float(t10["hit"].loc[cand, "full"]), 4),
            }
        )
        print(
            f"[verdict {cand}] full_mean={full_mean:.4f} vs {base_cand} {base:.4f} | "
            f"子窗 {sub_wins}/{SUBWINDOWS} | 过闸({GATE['hit']}/{GATE['mean']}) "
            f"hit={hit_ok} mean={mean_ok} → {verdicts[-1]['change']}",
            flush=True,
        )
    vpath = DIAG / f"phase2_decide_verdict_{ts}.json"
    vpath.write_text(
        json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[worm] {vpath}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
