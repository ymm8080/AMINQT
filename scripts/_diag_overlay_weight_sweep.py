"""_diag_overlay_weight_sweep.py — OVERLAY_WEIGHTS 单因子扫描 (2026-08-08).

复用 _diag_overlay_orthogonality 的装载/推理/量测原语 (非新 harness), 只把
fixed 0.5/0.5 换成 w_pool 扫描: combined = w_pool*pool_score + (1-w_pool)*prob.

回答 config.OVERLAY_WEIGHTS (main 0.2/0.8, dual 0.5/0.5) 是否仍是最优:
  - 每板块 OOS 6m (末窗), LEGACY prob 一次推理
  - w_pool ∈ {0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0} 扫 sniper TOP-5 / fusion TOP-10
  - 量 C2C 双头 (winrate/mag) 四视界; 附每视界加权汇总 (LABEL_WEIGHTS 口径)
输出 (WORM): data/_diag_overlay_weight_sweep_<ts>.json
用法: python scripts/_diag_overlay_weight_sweep.py [--board main] [--out ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _diag_overlay_orthogonality import (  # noqa: E402
    SYSTEMS,
    load_board,
    load_probs,
    measure,
)

from app.pipeline1.label_engine import LABEL_WEIGHTS  # noqa: E402
from app.pipeline_parallel.backtest import run_system  # noqa: E402
from app.pipeline_parallel.config import (  # noqa: E402
    BOARD_THRESHOLDS,
    C2C_LABELS,
    OOS_WINDOWS,
    PANEL,
)
from app.pipeline_parallel.scoring import pool_score  # noqa: E402

MODEL_DIR = os.path.join("models", "pipeline1")
DEFAULT_W_POOL = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)


def _fmt(v, pct=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:+.1%}" if pct else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="OVERLAY_WEIGHTS 单因子扫描")
    ap.add_argument("--board", default=None, help="main/dual, 默认两者")
    ap.add_argument("--window", default="6m", choices=list(OOS_WINDOWS))
    ap.add_argument("--w-pool", default=",".join(map(str, DEFAULT_W_POOL)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from app.pipeline1.predict_runner import resolve_current_bundles
    from app.pipeline1.predictor import V35Predictor

    bundles = resolve_current_bundles(MODEL_DIR)
    if not bundles:
        print(f"无模型包: {MODEL_DIR}")
        return 1
    predictor = V35Predictor(bundles)
    print("模型包:", {b: os.path.basename(p) for b, p in bundles.items()})

    w_pool_list = [float(x) for x in args.w_pool.split(",") if x.strip()]
    d = OOS_WINDOWS[args.window]
    out: dict = {
        "ts": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "overlay_weight_sweep",
        "window": args.window,
        "trading_days": d,
        "w_pool_grid": w_pool_list,
        "criteria": {b: t for b, t in BOARD_THRESHOLDS.items()},
        "boards": {},
    }
    for board in ["main", "dual"] if not args.board else [args.board]:
        if board not in bundles:
            print(f"[{board}] 无模型包, 跳过")
            continue
        print(f"\n========== 板块 [{board}] ==========", flush=True)
        sub = load_board(board)
        # load_board 只补 MFE 标签; C2C 标签 (label_pm_*_net, 验收口径) 需单独合并
        ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
        lab = pd.read_parquet(ckpt, columns=["symbol", "date"] + list(C2C_LABELS))
        sub = sub.merge(lab, on=["symbol", "date"], how="left")
        del lab
        gc.collect()
        dates = np.sort(sub["date"].unique())
        oos_start = dates[len(dates) - d]
        oos_end = dates[-1]
        mask = (sub["date"].values >= oos_start) & (sub["date"].values <= oos_end)
        sub = sub[mask].copy()
        bcrit = (
            BOARD_THRESHOLDS[board]["min_winrate"],
            BOARD_THRESHOLDS[board]["min_mag"],
        )
        print(
            f"OOS {pd.Timestamp(oos_start).date()} → {pd.Timestamp(oos_end).date()} "
            f"({d} 交易日) | 行 {len(sub):,} | wr>={bcrit[0]} mag>{bcrit[1]}",
            flush=True,
        )

        prob, pc = load_probs(board, predictor, oos_start)
        sub = sub.merge(prob, on=["date", "symbol"], how="left")
        print(f"LEGACY prob 列 = {pc} | OOS 覆盖 {sub['prob_up'].notna().mean():.1%}", flush=True)
        del prob
        gc.collect()

        board_out: dict = {"prob_col": pc, "latest": str(sub["date"].max()), "systems": {}}
        for name, spec in SYSTEMS.items():
            print(f"\n── 系统 [{name}] TOP-{spec.top_n} ──")
            s = sub.copy()
            s["pool_score"] = pool_score(s, spec.pool)
            prob_f = s["prob_up"].fillna(0.0)
            ref = run_system(sub, spec, spec.top_n, None, bcrit)
            rows = {}
            for wp in w_pool_list:
                wprob = 1.0 - wp
                s["combined"] = wp * s["pool_score"] + wprob * prob_f
                m = measure(s, s["combined"], spec, bcrit)
                per = m["per_horizon"]
                rows[f"{wp:.2f}"] = {
                    "w_pool": round(wp, 2),
                    "w_prob": round(wprob, 2),
                    "n_picks": m["n_picks"],
                    "per_horizon": {
                        h: {
                            "winrate": per.get(h, {}).get("winrate"),
                            "mag": per.get(h, {}).get("mag"),
                            "ok": bool(per.get(h, {}).get("ok", False)),
                        }
                        for h in ("3d", "5d", "10d", "2d")
                    },
                }
                # 主视界 3d + 全视界加权 (C2C 口径, LABEL_WEIGHTS)
                w3 = rows[f"{wp:.2f}"]["per_horizon"]["3d"]
                lab_w = {3: LABEL_WEIGHTS[3], 5: LABEL_WEIGHTS[5], 10: LABEL_WEIGHTS[10]}
                wsum, wtot = 0.0, 0.0
                for h, w in lab_w.items():
                    mag = rows[f"{wp:.2f}"]["per_horizon"][f"{h}d"]["mag"]
                    if np.isfinite(mag):
                        wsum += w * mag
                        wtot += w
                rows[f"{wp:.2f}"]["w_mag"] = round(wsum / wtot, 5) if wtot else None
                print(
                    f"  w_pool={wp:.2f} w_prob={wprob:.2f} | 3d wr {_fmt(w3.get('winrate'), True)} "
                    f"mag {_fmt(w3.get('mag'), True)} ok={'Y' if w3.get('ok') else 'n'} "
                    f"| 3d/5d/10d w_mag {_fmt(rows[f'{wp:.2f}']['w_mag'], True)}",
                    flush=True,
                )
            # 自检: w_pool=1.0 (纯池分) 的 3d mag 应等于 run_system 池分 3d mag
            _m = rows["1.00"]["per_horizon"].get("3d", {}).get("mag")
            _r = ref["per_horizon"].get("3d", {}).get("mag")
            ok = _m is not None and _r is not None and abs(_m - _r) < 1e-9
            print(f"  自检 w_pool=1.0 mag_3d==run_system: {'✓' if ok else '✗'}", flush=True)
            board_out["systems"][name] = {
                "top_n": spec.top_n,
                "selfcheck": bool(ok),
                "weights": rows,
            }
            del s
            gc.collect()
        out["boards"][board] = board_out
        del sub
        gc.collect()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
        print(f"\nWORM 落盘: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
