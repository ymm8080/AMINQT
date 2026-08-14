"""扫描 legacy main 推理端 liquidity_top_n 候选池大小对 top-10 预测质量的影响.

问题: 生产 main_20260812 (353 特征) 是在全流动性谱上训练的 (apply_top_n=False),
推理端默认候选池 liquidity_top_n=400. 全池约 2000 只/日, 400 是否最优? 扫 N.

设计: 同一 bundle、同一 OOS 窗口, 只变候选池 N.
  每 N: CleaningConfig(liquidity_top_n=N) → run_inference(main) → features.build
        (pool-rank, cross_sectional_rank=True) → LabelEngine 标签
      → 评估 wIC + top-{5,10,20} 实得收益 + 子窗稳定性.
主指标: top-10 实得 10d 均值 (生产排名视界) + wIC; 选稳定>最高.

main_20260812 有 85 个推理端不可复现的 _brute_ 特征 → 缺失列补 0 (与 predictor.py
生产行为一致), 不再 abort.

护栏: 只读 bundle; 标签 t+3/5/10 前向; 帧间 del + gc 防 OOM; WORM.
用法: python scripts/_main_predict_pool_sweep.py [--eval-days=60|125|250]
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
BUNDLES = ("main_20260812.pkl",)  # 生产 353 特征 (20260811b=316 可加作对比, 翻倍耗时)
# 400 已有锚点 (h2h main_20260812 60d top10 +5.16%); 特征丰富只扫大池 (用户 08-13 定)
# 用户 08-13: 从 1000 起跑 (600/800 已跳过, 1000 最快对标 400 锚点)
POOL_SIZES = (1000, 1500, 2000)  # 2000 > 日池(~1900) ≈ 全池
FULL_SENTINEL = 0
FEATURE_WARMUP_DAYS = 270
EVAL_DAYS = 60
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


def eval_bundle(bundle: dict, cols: list[str], test: pd.DataFrame, n_sub: int) -> dict:
    present = [c for c in cols if c in test.columns]
    missing = [c for c in cols if c not in test.columns]
    labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
    keep = [c for c in (present + labels) if c in test.columns] + ["date", "symbol"]
    t = test[keep].copy()
    for c in missing:
        t[c] = 0.0  # _brute_ 特征推理端不可复现 → 补 0 (生产行为)
    trained = {"segs": {"test": t}, "feature_cols": cols, "models": bundle["models"]}
    oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)
    sub_dates = sorted(t["date"].unique())
    step = len(sub_dates) // n_sub
    sub_ics: list[float] = []
    for i in range(n_sub):
        s0, s1 = i * step, len(sub_dates) if i == n_sub - 1 else (i + 1) * step
        sub_df = t[t["date"].isin(sub_dates[s0:s1])]
        tsub = {
            "segs": {"test": sub_df},
            "feature_cols": cols,
            "models": bundle["models"],
        }
        sub_ics.append(
            DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(tsub)["weighted_ic"]
        )
    return {
        "n_features": len(cols),
        "effective_n": len(present),
        "n_zero_filled": len(missing),
        "weighted_ic": oos["weighted_ic"],
        "ics": oos["ics"],
        "sub_window_ic": sub_ics,
        "sub_window_ic_mean": float(np.mean(sub_ics)),
        "sub_window_ic_std": float(np.std(sub_ics)),
        "topn": topn_metrics(t, bundle["models"], cols),
        "rows": int(len(t)),
    }


def main() -> int:
    import sys as _sys

    _eval_days = EVAL_DAYS
    _args = [a for a in _sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days
    n_sub = max(2, _eval_days // 60)

    t0 = time.time()
    union_cols: list[str] = []
    bundle_cols: dict[str, list[str]] = {}
    for fname in BUNDLES:
        tag = fname[len("main_") : -len(".pkl")]
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        bundle_cols[tag] = list(b["feature_cols"])
        union_cols = list(dict.fromkeys(union_cols + bundle_cols[tag]))
        del b
        gc.collect()
    print(f"[cols] union={len(union_cols)}", flush=True)

    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-warmup_days]]
    print(
        f"[panel] sliced last {warmup_days} trading days -> {len(panel):,}r",
        flush=True,
    )

    features = FeatureEngineV35()
    results: dict = {}

    for N in POOL_SIZES:
        time.time()
        cap = 10**9 if N == FULL_SENTINEL else N
        cfg = CleaningConfig(liquidity_top_n=cap, liquidity_top_n_main=cap)
        cleaner = CleaningPipeline(cfg)
        main_b, dual_b, state = cleaner.run_inference(panel)
        del dual_b
        gc.collect()
        if state == "empty":
            print(f"[N={N}] FATAL valve empty -> abort", flush=True)
            return 3
        df = features.build(
            main_b, None, inference_cols=union_cols, cross_sectional_rank=True
        )
        del main_b
        gc.collect()
        df = LabelEngine.build_path_labels(df)
        df = LabelEngine.build_labels(df)
        df = LabelEngine.mask_suspension(df)
        df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)

        ddates = sorted(df["date"].unique())
        eval_start = ddates[-_eval_days]
        eval_df = df[df["date"] >= eval_start].copy()
        per_day = eval_df.groupby("date")["symbol"].nunique()
        del df
        gc.collect()
        print(
            f"[N={N}] eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
            f"{len(eval_df):,}r 池均 {per_day.mean():.0f} 只/日",
            flush=True,
        )

        results[N] = {}
        for tag, cols in bundle_cols.items():
            bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"main_{tag}.pkl"))
            r = eval_bundle(bundle, cols, eval_df, n_sub)
            results[N][tag] = r
            t10 = r["topn"]["10d_n10"]
            print(
                f"  [{tag}] wIC={r['weighted_ic']:.4f} "
                f"sub={[round(x, 4) for x in r['sub_window_ic']]} "
                f"top10 10d={t10['mean_ret']:+.3f} hit={t10['hit']:.3f}",
                flush=True,
            )
            del bundle, eval_df
            gc.collect()

    verdict: dict = {}
    for tag in bundle_cols:
        best = max(
            POOL_SIZES,
            key=lambda n: results[n][tag]["topn"]["10d_n10"]["mean_ret"],
        )
        verdict[tag] = {
            "best_N_top10": best,
            "top10_10d_by_N": {
                n: results[n][tag]["topn"]["10d_n10"]["mean_ret"] for n in POOL_SIZES
            },
            "wic_by_N": {n: results[n][tag]["weighted_ic"] for n in POOL_SIZES},
        }
    results["_verdict"] = verdict

    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"main_predict_pool_sweep_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    for tag, v in verdict.items():
        print(f"[verdict {tag}] best_N={v['best_N_top10']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
