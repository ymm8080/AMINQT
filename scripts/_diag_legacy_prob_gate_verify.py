"""_diag_legacy_prob_gate_verify.py — legacy 并行式概率闸端到端验证 (2026-08-17 接线落地).

背景: LEGACY_PROB_GATE 代码已 landed (prob_head.py / list_generator / daily_pipeline),
bundle 已训练 (data/prob_head_legacy/). 本脚本验证闸在生产路径上真在工作:
  1. load_panel_v3 预过滤加载面板 (同训练脚本口径)
  2. CleaningPipeline.run_inference → main_df/dual_df (生产推理端口径)
  3. FeatureEngineV35.build 现场构建特征 (inference_cols, dual csr=True)
  4. DailySelectionPipeline._prob_gate_inputs 组装 {feats, tail, panel_dates} (生产函数)
  5. 对最近候选文件 (data/lists/candidates_*.parquet 最新) 直接调
     prob_head.apply_prob_gate + ListGenerator.emit(prob_gate=...) 全链路
验证点 (任务要求): a) load_latest 能读到 bundle; b) pred_prob 截面区分度
(唯一值数 > 20, std > 0.02); c) 剔除率 ∈ (0, 95%) 不全杀不全放.
WORM: DATA_OTHERS/diag/legacy_prob_gate_verify_<ts>.json (不覆盖).

用法: python scripts/_diag_legacy_prob_gate_verify.py [--days N]
  --days N: 面板只取最近 N 个交易日 (特征窗口 ≤125d, 最新日截面值几乎一致,
            省 ~3.5x 构建时间 + 峰值内存; 2026-08-17 全量版因并发内存压力
            退化到 thrash 被终止, 故加此开关).
"""

from __future__ import annotations

import gc
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import LEGACY_PROB_GATE, PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
# 任务验收线: b) 截面区分度 / c) 闸生效且不全杀
MIN_UNIQUE = 20
MIN_STD = 0.02
MAX_DROP_RATE = 0.95


def _gate_stats(cands: pd.DataFrame, out: pd.DataFrame, board_group: str) -> dict:
    """闸前候选 vs 闸后输出 → 该 board 组 (main/GEM/STAR→dual) 剔除率 + pred_prob 统计."""
    n_in = int(cands["board"].astype(str).map(prob_head._BOARD_GROUP).eq(board_group).sum())
    s = out["board"].astype(str).map(prob_head._BOARD_GROUP).eq(board_group)
    n_out = int(s.sum())
    p = out.loc[s, "pred_prob"].dropna()
    return {
        "n_candidates": n_in,
        "n_survivors": n_out,
        "drop_rate": 1.0 - n_out / n_in if n_in else float("nan"),
        "n_with_prob": int(len(p)),
        "coverage": float(len(p) / n_in) if n_in else float("nan"),
        "pred_prob": {
            "n_unique": int(p.nunique()),
            "std": float(p.std()) if len(p) > 1 else float("nan"),
            "min": float(p.min()) if len(p) else None,
            "median": float(p.median()) if len(p) else None,
            "max": float(p.max()) if len(p) else None,
        },
    }


def main() -> int:
    t0 = time.time()
    cand_path = max(
        glob.glob("data/lists/candidates_*.parquet"),
        key=lambda p: p.split("candidates_")[1].split(".")[0],
    )
    cands = pd.read_parquet(cand_path)
    print(f"[cands] {cand_path}: {len(cands):,}r", flush=True)

    days_arg = None
    if "--days" in sys.argv:
        days_arg = int(sys.argv[sys.argv.index("--days") + 1])

    panel = load_panel_v3(path=PANEL_V3_PATH)
    panel_max = pd.Timestamp(panel["date"].max())
    if days_arg is not None:
        uniq = np.unique(pd.to_datetime(panel["date"].to_numpy()))
        cutoff = uniq[-days_arg]
        panel = panel[pd.to_datetime(panel["date"].to_numpy()) >= cutoff]
        print(
            f"[panel] --days {days_arg} (从 {pd.Timestamp(cutoff).date()}): {len(panel):,}r "
            f"max={panel_max.date()} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    else:
        print(
            f"[panel] {len(panel):,}r max={panel_max.date()} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    main_df, dual_df, valve = CleaningPipeline().run_inference(panel)
    print(
        f"[clean] valve={valve} main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    predictor = V35Predictor(BUNDLES)
    features = FeatureEngineV35()
    gate_feats: dict[str, pd.DataFrame] = {}
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        if dfb.empty:
            print(f"[feat:{board}] 清洗帧为空 -> skip", flush=True)
            continue
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        gate_feats[board] = feat[feat["date"] == feat["date"].max()].copy()
        del feat, dfb
        gc.collect()
        print(
            f"[feat:{board}] 截面 {len(gate_feats[board]):,}r / "
            f"{len(gate_feats[board].columns)}c ({time.time() - t0:.0f}s)",
            flush=True,
        )

    prob_gate = DailySelectionPipeline._prob_gate_inputs(panel, main_df, dual_df, gate_feats)
    del panel, main_df, dual_df
    gc.collect()
    print(
        f"[prob_gate] 输入组装: feats={ {k: len(v) for k, v in prob_gate['feats'].items()} } "
        f"tail={len(prob_gate['tail']):,} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    panel_dates = np.unique(pd.to_datetime(prob_gate["panel_dates"]).to_numpy())
    bundles = {}
    for board in ("main", "dual"):
        b = prob_head.load_latest(board)
        age = None if b is None else prob_head.bundle_age_trading_days(
            panel_dates, str(b["trained_through"])
        )
        bundles[board] = {
            "bundle": None if b is None else str(b["trained_through"]),
            "age_trading_days": age,
            "n_feats": None if b is None else len(b["feat_cols"]),
            "usable": b is not None and age is not None
            and age <= LEGACY_PROB_GATE["max_stale_days"],
        }
        print(
            f"[bundle:{board}] {bundles[board]} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ---- 5a) 直接闸 (候选文件全量) ----
    out = prob_head.apply_prob_gate(cands.copy(), prob_gate["feats"], prob_gate["tail"], panel_dates)
    gate = {board: _gate_stats(cands, out, board) for board in ("main", "dual")}
    for board, g in gate.items():
        print(
            f"[gate:{board}] 候选 {g['n_candidates']} -> 剩 {g['n_survivors']} "
            f"(剔 {g['drop_rate']:.1%}) | pred_prob unique={g['pred_prob']['n_unique']} "
            f"std={g['pred_prob']['std']:.4f} cov={g['coverage']:.1%}",
            flush=True,
        )

    # ---- 5b) 生产全链路 emit(prob_gate=...) ----
    emit_res = None
    try:
        lister = ListGenerator()
        emit_res = lister.emit(
            cands.copy(),
            env=None,
            market_state="range",
            ref_date=str(panel_max.date()),
            prob_gate=prob_gate,
        )
        lst = emit_res.get("list", pd.DataFrame())
        print(
            f"[emit] mode={emit_res.get('mode')} list={len(lst):,} "
            f"empty={emit_res.get('empty')} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    except Exception as exc:  # emit 失败 → 大声报告, 不吞
        print(f"[emit] FAIL: {type(exc).__name__}: {exc}", flush=True)
        emit_res = {"error": f"{type(exc).__name__}: {exc}"}

    # ---- 判定 ----
    checks = {}
    ok = True
    for board in ("main", "dual"):
        b_ok = bundles[board]["usable"]
        g = gate[board]
        disc = g["pred_prob"]["n_unique"] > MIN_UNIQUE and g["pred_prob"]["std"] > MIN_STD
        engaged = g["drop_rate"] > 0
        not_wipe = g["drop_rate"] < MAX_DROP_RATE
        checks[board] = {
            "bundle_usable": b_ok,
            "discrimination": disc,
            "engaged": engaged,
            "not_wipeout": not_wipe,
            "pass": b_ok and disc and engaged and not_wipe,
        }
        if not checks[board]["pass"]:
            ok = False
            print(
                f"[VERDICT:{board}] FAIL "
                f"bundle={b_ok} disc={disc} engaged={engaged} not_wipe={not_wipe}",
                flush=True,
            )
        else:
            print(f"[VERDICT:{board}] PASS", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "ts": ts,
        "candidates_file": cand_path,
        "panel_max": str(panel_max.date()),
        "panel_days": days_arg,
        "bundles": bundles,
        "gate": gate,
        "checks": checks,
        "emit": emit_res,
        "pass_all": ok,
    }
    out_path = data_others_path("diag") / f"legacy_prob_gate_verify_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(
        f"[saved] {out_path} | pass_all={ok} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
