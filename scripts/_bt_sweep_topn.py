"""扫描推理端 top-N 候选池大小 (liquidity_top_n) 对 top-10 预测质量的影响.

问题: 每日清单在 top-N 流动性池内取模型前 N 名. 上回测 (feat_serve_a_vs_b_20260812_232012)
证明"池内 (top-400) 特征 > 全板块特征". 那 400 是否最优? 扫 N.

设计: 同一 bundle、同一 60d OOS 窗口 (2026-05-20..08-12), 只变候选池大小 N.
  每 N: CleaningConfig(liquidity_top_n=N) → run_inference → features.build
        (union_cols, cross_sectional_rank=True, registry=None) → LabelEngine 标签
      → 评估两 bundle (38/328) 的 wIC + top-{5,10,20} 实得收益 + 子窗稳定性.
主指标: top-10 实得 10d 均值 (生产排名视界) + wIC; 选稳定>最高.

护栏: 只读 bundle; 标签 t+3/5/10 前向; 帧间 del + gc 防 OOM; WORM.
NOTE: 全板块 (∞ 池) 锚点已在 feat_serve_a_vs_b 测得 (38: top10=+3.7% / 328: +0.9%,
      均低于池 400), 故扫描不含全板块.
NOTE: 长窗口 (--eval-days 125/250) 时评估段含模型训练见过的历史 → 绝对实得虚高
      (非严格 OOS), 但各 N 偏差对称 → 相对最优 N 结论仍可信.

用法: python scripts/_bt_sweep_topn.py [--eval-days=125|250]
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
BUNDLES = ("dual_20260811b.pkl", "dual_20260812.pkl")
POOL_SIZES = (100, 200, 300, 400, 600, 800)
FEATURE_WARMUP_DAYS = 270  # 特征滚动窗口暖机天数 (历史统计量需先积累)
EVAL_DAYS = 60  # 评估窗口交易日数 (可 --eval-days 覆盖: 125=半年/250=一年)
N_SUB = 2  # 子窗稳定性分段数 (随 EVAL_DAYS 缩放)
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


def eval_bundle(bundle: dict, cols: list[str], test: pd.DataFrame, n_sub: int = N_SUB) -> dict:
    trained = {"segs": {"test": test}, "feature_cols": cols, "models": bundle["models"]}
    oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)
    sub_dates = sorted(test["date"].unique())
    step = len(sub_dates) // n_sub
    sub_ics: list[float] = []
    for i in range(n_sub):
        s0, s1 = i * step, len(sub_dates) if i == n_sub - 1 else (i + 1) * step
        sub_df = test[test["date"].isin(sub_dates[s0:s1])]
        tsub = {"segs": {"test": sub_df}, "feature_cols": cols, "models": bundle["models"]}
        sub_ics.append(DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(tsub)["weighted_ic"])
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
    _args = [a for a in _sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days  # 特征暖机 + 评估窗口
    n_sub = max(2, _eval_days // 60)  # 子窗约 60 交易日, 半年=2 / 一年=4

    t0 = time.time()
    union_cols: list[str] = []
    bundle_cols: dict[str, list[str]] = {}
    for fname in BUNDLES:
        tag = fname[len("dual_"): -len(".pkl")]
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        bundle_cols[tag] = list(b["feature_cols"])
        union_cols = list(dict.fromkeys(union_cols + bundle_cols[tag]))
        del b
        gc.collect()
    print(f"[cols] union={len(union_cols)} (38={len(bundle_cols['20260811b'])}, "
          f"328={len(bundle_cols['20260812'])})", flush=True)

    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-warmup_days]]
    print(f"[panel] sliced last {warmup_days} trading days (warmup=270+eval={_eval_days}) "
          f"-> {len(panel):,}r", flush=True)

    features = FeatureEngineV35()
    results: dict = {}

    for N in POOL_SIZES:
        tN = time.time()
        cleaner = CleaningPipeline(CleaningConfig(liquidity_top_n=N))
        main_b, dual_b, state = cleaner.run_inference(panel)
        del main_b
        gc.collect()
        if state == "empty":
            print(f"[N={N}] FATAL valve empty → abort", flush=True)
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
        print(f"[N={N}] 特征帧 eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
              f"{len(eval_df):,}r", flush=True)

        results[N] = {}
        for tag, cols in bundle_cols.items():
            bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"dual_{tag}.pkl"))
            labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
            keep = [c for c in (cols + labels) if c in eval_df.columns] + ["date", "symbol"]
            missing = [c for c in cols if c not in eval_df.columns]
            if missing:
                print(f"[FATAL] N={N} {tag} 缺失 {missing[:5]} → abort", flush=True)
                return 3
            test = eval_df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
            results[N][tag] = eval_bundle(bundle, cols, test, n_sub)
            r = results[N][tag]
            t10 = r["topn"]["10d_n10"]
            print(f"  [{tag}] wIC={r['weighted_ic']:.4f} sub={[round(x,4) for x in r['sub_window_ic']]} "
                  f"top10 10d={t10['mean_ret']:+.3f} hit={t10['hit']:.3f}", flush=True)
            del bundle, test
            gc.collect()
        print(f"[N={N}] done ({time.time() - tN:.0f}s)", flush=True)

    # ---- 判定: 每 bundle 选 top-10 10d 实得最高 + 子窗稳定 ----
    verdict: dict = {}
    for tag in bundle_cols:
        best = max(POOL_SIZES, key=lambda n: results[n][tag]["topn"]["10d_n10"]["mean_ret"])
        verdict[tag] = {
            "best_N_top10": best,
            "top10_10d_by_N": {n: results[n][tag]["topn"]["10d_n10"]["mean_ret"]
                               for n in POOL_SIZES},
            "wic_by_N": {n: results[n][tag]["weighted_ic"] for n in POOL_SIZES},
        }
    results["_verdict"] = verdict

    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"topn_pool_sweep_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
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
