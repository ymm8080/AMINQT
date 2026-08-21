"""扫描推理端 top-N 候选池大小 (liquidity_top_n) — 2026-08-20 重扫版.

08-13 首扫 (dual_20260811b, 38 特征): 100/200 峰值, 400 三窗全输 200 (top10 10d
250d +16.75 vs +19.63, 125d +15.70 vs +19.28, 60d +9.42 vs +12.48) → dual 锁 200.
触发重扫: 用户关注 300911 8/18 缩量洗盘日被 top-200 池切掉 (成交 2.8 亿 vs 门槛 10.3 亿),
且模型已重训两代 (20260816 新宇宙 / 20260819 neg200 特征集 + E6=10%), 旧结论未必成立.

本版: 当前生产 bundle dual_20260819 (208 特征), 当前生产 E6=10% (cleaning_pipeline 内置),
只变 N ∈ {200, 300, 400}. 主指标: top-10 实得 10d (生产排名视界) + wIC + 子窗稳定性.

用法: python scripts/_bt_sweep_topn_20260820.py [--eval-days=125|250] [--sizes=200,300,400]
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
)
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from config.settings import BACKTEST_RESULT_DIR

MODEL_DIR = "models/pipeline1"
BUNDLES = ("dual_20260819.pkl",)
POOL_SIZES = (200, 300, 400)
FEATURE_WARMUP_DAYS = 270  # 特征滚动窗口暖机天数 (历史统计量需先积累)
EVAL_DAYS = 250  # 评估窗口交易日数 (可 --eval-days 覆盖: 125=半年/250=一年)
N_SUB = 4  # 子窗稳定性分段数 (随 EVAL_DAYS 缩放)
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
    _sizes = POOL_SIZES
    _args = [a for a in _sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    _size_args = [a for a in _sys.argv[1:] if a.startswith("--sizes=")]
    if _size_args:
        _sizes = tuple(int(x) for x in _size_args[-1].split("=", 1)[1].split(","))
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days  # 特征暖机 + 评估窗口
    n_sub = max(2, _eval_days // 60)  # 子窗约 60 交易日, 半年=2 / 一年=4

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

    import pyarrow.parquet as pq

    from config.settings import PANEL_V3_PATH

    _all_dates = (
        pq.read_table(str(PANEL_V3_PATH), columns=["date"])["date"]
        .to_pandas()
        .dt.date.unique()
    )
    _all_dates = pd.Series(sorted(_all_dates))
    cutoff_date = pd.Timestamp(_all_dates.iloc[-warmup_days])
    del _all_dates
    gc.collect()
    print(
        f"[panel] cutoff {cutoff_date.date()} (last {warmup_days} trading days)",
        flush=True,
    )
    panel = pq.read_table(
        str(PANEL_V3_PATH),
        filters=[
            ("amount", ">=", CleaningConfig().min_amount),
            ("date", ">=", cutoff_date),
        ],
    ).to_pandas()
    panel = panel[panel["is_suspended"] == 0].reset_index(drop=True)
    print(
        f"[panel] date-filtered load -> {len(panel):,}r",
        flush=True,
    )

    features = FeatureEngineV35()
    results: dict = {}

    for N in _sizes:
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
        print(
            f"[N={N}] 特征帧 eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
            f"{len(eval_df):,}r",
            flush=True,
        )

        results[N] = {}
        for tag, cols in bundle_cols.items():
            bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"dual_{tag}.pkl"))
            labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
            keep = [c for c in (cols + labels) if c in eval_df.columns] + [
                "date",
                "symbol",
            ]
            missing = [c for c in cols if c not in eval_df.columns]
            if missing:
                print(f"[FATAL] N={N} {tag} 缺失 {missing[:5]} → abort", flush=True)
                return 3
            test = eval_df[keep].sort_values(["date", "symbol"]).reset_index(drop=True)
            results[N][tag] = eval_bundle(bundle, cols, test, n_sub)
            r = results[N][tag]
            t10 = r["topn"]["10d_n10"]
            print(
                f"  [{tag}] wIC={r['weighted_ic']:.4f} sub={[round(x, 4) for x in r['sub_window_ic']]} "
                f"top10 10d={t10['mean_ret']:+.3f} hit={t10['hit']:.3f}",
                flush=True,
            )
            del bundle, test
            gc.collect()
        print(f"[N={N}] done ({time.time() - tN:.0f}s)", flush=True)

    # ---- 判定: 每 bundle 选 top-10 10d 实得最高 + 子窗稳定 ----
    verdict: dict = {}
    for tag in bundle_cols:
        best = max(_sizes, key=lambda n: results[n][tag]["topn"]["10d_n10"]["mean_ret"])
        verdict[tag] = {
            "best_N_top10": best,
            "top10_10d_by_N": {
                n: results[n][tag]["topn"]["10d_n10"]["mean_ret"] for n in _sizes
            },
            "wic_by_N": {n: results[n][tag]["weighted_ic"] for n in _sizes},
        }
    results["_verdict"] = verdict

    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"topn_pool_sweep_20260820_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
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
