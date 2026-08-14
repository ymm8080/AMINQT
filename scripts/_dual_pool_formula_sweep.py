"""扫描 legacy dual 推理端 liquidity_score 公式 (N=200 固定) 对预测质量的影响.

问题: 08-13 定案 dual serving 候选池 N=200. 池内排名公式当前 = rank_5050
(0.5·rank(成交额) + 0.5·rank(自由流通换手)). 成交额与换手高度相关, 换公式
是否带来更好预测质量? 扫 6 种公式, 固定 N=200, 只变 CleaningConfig.score_mode.

公式:
  rank_5050        (基线) 0.5·rank(amount) + 0.5·rank(free_float_turnover)
  rank_amount      rank(amount)
  rank_turnover    rank(free_float_turnover)
  zlog_5050        0.5·z(log(amount)) + 0.5·z(log(free_float_turnover))
  zlog_product     z(log(amount·free_float_turnover))
  rank_5050_churnpen  rank_5050, 对倒嫌疑股 score×0.5

设计: 同一 bundle (dual_20260811b 生产固定版)、同一 OOS 窗口 (最近 eval_days),
  只变 score_mode. run_inference(dual) → features.build → LabelEngine
  → 评估 wIC + sub-window IC + top-{5,10,20} 实得收益 (三视界).
主指标: top-10 实得 10d 均值 (生产排名视界) + wIC; 选稳定>最高.
护栏: 只读 bundle; 帧间 del + gc 防 OOM; WORM.
用法: python scripts/_dual_pool_formula_sweep.py [--eval-days=125|60|250]
默认 eval-days=125 (08-13 用户定案主窗口), 60/250 为补充稳定性窗口.
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

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from config.settings import BACKTEST_RESULT_DIR

MODEL_DIR = "models/pipeline1"
BUNDLES = (
    "dual_20260811b.pkl",
)  # 生产固定版 (20260811c/20260812 gate_d 过拟合签已回退)
POOL_N = 200  # 08-13 定案 dual 池大小
FORMULAS = (
    "rank_5050",
    "rank_amount",
    "rank_turnover",
    "zlog_5050",
    "zlog_product",
    "rank_5050_churnpen",
)
FEATURE_WARMUP_DAYS = 270
EVAL_DAYS = 125  # 主窗口 (08-13 用户定案); 60/250 用 --eval-days 补充跑
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
        tag = fname[len("dual_") : -len(".pkl")]
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, fname))
        bundle_cols[tag] = list(b["feature_cols"])
        union_cols = list(dict.fromkeys(union_cols + bundle_cols[tag]))
        del b
        gc.collect()
    print(f"[cols] union={len(union_cols)}", flush=True)

    import pyarrow.parquet as pq

    from config.settings import PANEL_V3_PATH

    # 读时 pyarrow 过滤 (amount/停牌 同 load_panel_v3 口径 + date 下界),
    # 避免全量面板加载后布尔掩码导致的 30MiB 级 OOM (本机内存紧张).
    cfg0 = CleaningConfig()
    _date_col = (
        pq.read_table(str(PANEL_V3_PATH), columns=["date"]).column("date").to_pandas()
    )
    dates = sorted(pd.unique(_date_col))
    del _date_col
    cutoff = dates[-warmup_days]  # 面板 date 列为 ISO 字符串, 字典序=时间序
    panel = pq.read_table(
        str(PANEL_V3_PATH),
        filters=[
            ("amount", ">=", cfg0.min_amount),
            ("is_suspended", "=", False),
            ("date", ">=", cutoff),
        ],
    ).to_pandas()
    print(
        f"[panel] read {len(panel):,}r from {cutoff}..{dates[-1]}",
        flush=True,
    )

    features = FeatureEngineV35()
    results: dict = {}

    for formula in FORMULAS:
        time.time()
        cfg = CleaningConfig(
            score_mode=formula, liquidity_top_n=POOL_N, liquidity_top_n_main=POOL_N
        )
        cleaner = CleaningPipeline(cfg)
        main_b, dual_b, state = cleaner.run_inference(panel)
        del main_b
        gc.collect()
        if state == "empty":
            print(f"[{formula}] FATAL valve empty -> abort", flush=True)
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
        per_day = eval_df.groupby("date")["symbol"].nunique()
        del df
        gc.collect()
        print(
            f"[{formula}] eval {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} "
            f"{len(eval_df):,}r 池均 {per_day.mean():.0f} 只/日",
            flush=True,
        )

        results[formula] = {}
        for tag, cols in bundle_cols.items():
            bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"dual_{tag}.pkl"))
            r = eval_bundle(bundle, cols, eval_df, n_sub)
            results[formula][tag] = r
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
            FORMULAS,
            key=lambda f: results[f][tag]["topn"]["10d_n10"]["mean_ret"],
        )
        verdict[tag] = {
            "best_formula_top10": best,
            "top10_10d_by_formula": {
                f: results[f][tag]["topn"]["10d_n10"]["mean_ret"] for f in FORMULAS
            },
            "wic_by_formula": {f: results[f][tag]["weighted_ic"] for f in FORMULAS},
        }
    results["_verdict"] = verdict

    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"dual_pool_formula_sweep_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    for tag, v in verdict.items():
        print(f"[verdict {tag}] best_formula={v['best_formula_top10']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
