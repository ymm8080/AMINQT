"""E6% (bottom_amount_pct) 重校 sweep — N=800 定案后.

08-20 topn_pool_sweep 定案 N=800 (top10 10d +25.53%, wIC 0.2172, 全赢 500/600).
旧 E6=10% 是 N=200 时扫参定案 (08-19 重扫: 10% 命中54.5%/实得+4.28% 胜 20%).
N 从 200 扩到 800 后, 池尾流动性结构已变 — 需重扫 E6% 找 N=800 最优.

设计 (对齐 _bt_sweep_topn_20260820.py):
  - 固定 N=800 (liquidity_top_n=800, main=0)
  - 扫 E6 ∈ {0%, 5%, 10%, 15%, 20%} (dual 板块; main 保持 0%)
  - 评估: top-10 实得 10d + wIC + 子窗稳定性 (250d OOS)
  - 生产 bundle: dual_20260819 (208 特征)

用法: python scripts/_sweep_e6_for_N800.py [--eval-days=250] [--pct=0,5,10,15,20]
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import (
    CleaningConfig,
    CleaningPipeline,
    load_panel_v3,
)
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from config.settings import BACKTEST_RESULT_DIR

MODEL_DIR = "models/pipeline1"
BUNDLES = ("dual_20260819.pkl",)
E6_PCTS = (0.0, 0.05, 0.10, 0.15, 0.20)  # 扫描的 E6 百分比
LIQUIDITY_TOP_N = 800  # 定案 N
FEATURE_WARMUP_DAYS = 270
EVAL_DAYS = 250
N_SUB = 4
TOPN = (5, 10, 20)
HORIZONS = (3, 5, 10)


def topn_metrics(test: pd.DataFrame, models: dict, cols: list[str]) -> dict:
    reg10, _ = models["10d_reg"]
    X = np.nan_to_num(test[cols].values, nan=0.0)
    test = test.copy()
    test["_pred_10d"] = reg10.predict(X)
    out: dict = {}
    for h in HORIZONS:
        lab = f"label_pm_{h}d_net"
        sub = test.dropna(subset=[lab]).copy()
        for n in TOPN:
            top = sub.sort_values("_pred_10d", ascending=False).groupby("date").head(n)
            out[f"{h}d_n{n}"] = {
                "mean_ret": float(top[lab].mean()),
                "hit": float((top[lab] > 0).mean()),
                "rows": int(len(top)),
            }
    return out


def eval_bundle(
    bundle: dict, cols: list[str], test: pd.DataFrame, n_sub: int = N_SUB
) -> dict:
    trained = {"segs": {"test": test}, "feature_cols": cols, "models": bundle["models"]}
    oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)
    sub_dates = sorted(test["date"].unique())
    step = len(sub_dates) // n_sub
    sub_ics: list[float] = []
    for i in range(n_sub):
        s0, s1 = i * step, len(sub_dates) if i == n_sub - 1 else (i + 1) * step
        sub_df = test[test["date"].isin(sub_dates[s0:s1])]
        tsub = {
            "segs": {"test": sub_df},
            "feature_cols": cols,
            "models": bundle["models"],
        }
        sub_ics.append(
            DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(tsub)["weighted_ic"]
        )
    return {
        "weighted_ic": oos["weighted_ic"],
        "ics": oos["ics"],
        "sub_window_ic": sub_ics,
        "sub_window_ic_mean": float(np.mean(sub_ics)),
        "sub_window_ic_std": float(np.std(sub_ics)),
        "topn": topn_metrics(test, bundle["models"], cols),
        "rows": int(len(test)),
    }


def main() -> int:
    import sys as _sys

    _eval_days = EVAL_DAYS
    _pcts = E6_PCTS
    _args = [a for a in _sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    _pct_args = [a for a in _sys.argv[1:] if a.startswith("--pct=")]
    if _pct_args:
        _pcts = tuple(float(x) for x in _pct_args[-1].split("=", 1)[1].split(","))
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days
    n_sub = max(2, _eval_days // 60)

    t0 = time.time()
    union_cols: list[str] = []
    bundle_cols: dict[str, list[str]] = {}
    for fname in BUNDLES:
        tag = fname[len("dual_") : -len(".pkl")]
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        bundle_cols[tag] = list(b["feature_cols"])
        union_cols = list(dict.fromkeys(union_cols + bundle_cols[tag]))
        del b
        gc.collect()
    for tag, cols in bundle_cols.items():
        print(f"[cols] {tag} n_feats={len(cols)}", flush=True)

    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-warmup_days]]
    print(
        f"[panel] sliced last {warmup_days} trading days (warmup=270+eval={_eval_days}) "
        f"-> {len(panel):,}r",
        flush=True,
    )

    features = FeatureEngineV35()
    results: dict = {}

    for pct in _pcts:
        tpct = time.time()
        # N=800 定案 + 当前扫描的 E6%
        cleaner = CleaningPipeline(
            CleaningConfig(
                liquidity_top_n=LIQUIDITY_TOP_N,
                bottom_amount_pct=pct,
                bottom_amount_pct_main=0.0,
            )
        )
        main_b, dual_b, state = cleaner.run_inference(panel)
        del main_b
        gc.collect()
        if state == "empty":
            print(f"[E6={pct:.0%}] FATAL valve empty → abort", flush=True)
            return 3
        df = features.build(
            dual_b, None, inference_cols=union_cols, cross_sectional_rank=True
        )
        del dual_b
        gc.collect()
        df = LabelEngine.build_path_labels(df)
        df = LabelEngine.build_labels(df)
        df = LabelEngine.mask_suspension(df)
        df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)

        ddates = sorted(df["date"].unique())
        eval_start = ddates[-_eval_days]
        eval_df = df[df["date"] >= eval_start].copy()
        del df
        gc.collect()
        print(
            f"[E6={pct:.0%}] 特征帧 eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
            f"{len(eval_df):,}r",
            flush=True,
        )

        results[pct] = {}
        for tag, cols in bundle_cols.items():
            bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"dual_{tag}.pkl"))
            labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
            keep = [c for c in (cols + labels) if c in eval_df.columns] + [
                "date",
                "symbol",
            ]
            missing = [c for c in cols if c not in eval_df.columns]
            if missing:
                print(
                    f"[FATAL] E6={pct:.0%} {tag} 缺失 {missing[:5]} → abort", flush=True
                )
                return 3
            test = eval_df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
            results[pct][tag] = eval_bundle(bundle, cols, test, n_sub)
            r = results[pct][tag]
            t10 = r["topn"]["10d_n10"]
            print(
                f"  [{tag}] wIC={r['weighted_ic']:.4f} sub={[round(x, 4) for x in r['sub_window_ic']]} "
                f"top10 10d={t10['mean_ret']:+.3f} hit={t10['hit']:.3f}",
                flush=True,
            )
            del bundle, test
            gc.collect()
        print(f"[E6={pct:.0%}] done ({time.time() - tpct:.0f}s)", flush=True)

    # ---- 判定: 每 bundle 选 top-10 10d 实得最高 + 子窗稳定 ----
    verdict: dict = {}
    for tag in bundle_cols:
        best = max(_pcts, key=lambda p: results[p][tag]["topn"]["10d_n10"]["mean_ret"])
        verdict[tag] = {
            "best_E6_top10": best,
            "top10_10d_by_E6": {
                p: results[p][tag]["topn"]["10d_n10"]["mean_ret"] for p in _pcts
            },
            "wic_by_E6": {p: results[p][tag]["weighted_ic"] for p in _pcts},
        }
    results["_verdict"] = verdict
    results["_meta"] = {
        "N": LIQUIDITY_TOP_N,
        "eval_days": _eval_days,
        "bundle": BUNDLES[0],
    }

    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"e6_sweep_N{LIQUIDITY_TOP_N}_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    for tag, v in verdict.items():
        print(f"[verdict {tag}] best_E6={v['best_E6_top10']:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
