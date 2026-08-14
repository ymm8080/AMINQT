"""main_20260812 (353 特征, 生产) 大 TOP-N OOS 评估: 100/200/300/500/600/800.

与 h2h 同一 OOS 窗口 (面板最近 60 交易日) 与同一评估机制 (10d_reg 排名,
逐日 top-N 已实现收益均值 + 上涨率). 353 中 86 个推理端不可复现的 _brute_
特征与生产一致补 0 (predictor.py 行为).

额外存 eval_df (含预测列) 到结果目录, 便于后续改 N 秒出结果.
用法: python scripts/_topn_main_353_largeN.py
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

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import BACKTEST_RESULT_DIR, data_others_path

MODEL_DIR = "models/pipeline1"
BUNDLE = "main_20260812.pkl"
WARMUP_DAYS = 330  # 与 h2h 一致: 270 暖机 + 60 评估
EVAL_DAYS = 60
HORIZONS = (3, 5, 10)
TOPN = (100, 200, 300, 500, 600, 800)


def topn_metrics(test: pd.DataFrame, model10, cols: list[str]) -> dict:
    """按 10d_reg 预测排名, 逐日 top-N 已实现收益均值 + 上涨率."""
    X = np.nan_to_num(test[cols].values, nan=0.0)
    test = test.copy()
    test["_pred_10d"] = model10.predict(X)
    out: dict = {}
    for h in HORIZONS:
        lab = f"label_pm_{h}d_net"
        sub = test.dropna(subset=[lab]).copy()
        pool = {
            "mean_ret": float(sub[lab].mean()),
            "hit": float((sub[lab] > 0).mean()),
            "rows": int(len(sub)),
        }
        per_n: dict = {}
        for n in TOPN:
            top = sub.sort_values("_pred_10d", ascending=False).groupby("date").head(n)
            per_n[str(n)] = {
                "mean_ret": float(top[lab].mean()),
                "hit": float((top[lab] > 0).mean()),
                "rows": int(len(top)),
            }
        out[f"{h}d"] = {"pool": pool, "topn": per_n}
    return out, test


def main() -> int:
    panel = load_panel_v3()
    dates = sorted(panel["date"].unique())
    print(
        f"[panel] {len(panel):,}r {len(dates)}d max={panel['date'].max():%Y-%m-%d}",
        flush=True,
    )
    cut = dates[-WARMUP_DAYS]
    panel = panel[panel["date"] >= cut]
    print(f"[slice] last {WARMUP_DAYS} trading days -> {len(panel):,}r", flush=True)

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del dual_df, panel
    gc.collect()
    print(f"[main] 清洗后 {len(main_df):,}r", flush=True)

    features = FeatureEngineV35()
    registry = FeatureRegistry(
        path=os.path.join(
            str(data_others_path("data/factor_registry")), "feature_registry.json"
        )
    )
    df = prepare_board_frame(
        main_df, features, cross_sectional_rank=True, registry=registry
    )
    del main_df
    gc.collect()
    print(f"[feat] main 特征帧 {len(df):,}r x {len(df.columns)}c", flush=True)

    ddates = sorted(df["date"].unique())
    eval_start = ddates[-EVAL_DAYS]
    eval_df = df[df["date"] >= eval_start].copy()
    del df
    gc.collect()
    print(
        f"[eval] 窗口 {eval_start:%Y-%m-%d}..{ddates[-1]:%Y-%m-%d} = {len(eval_df):,}r",
        flush=True,
    )

    bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, BUNDLE))
    cols = bundle["feature_cols"]
    present = [c for c in cols if c in eval_df.columns]
    missing = [c for c in cols if c not in eval_df.columns]
    labels = sorted({lbl for _, (_, lbl) in bundle["models"].items()})
    keep = [c for c in (present + labels) if c in eval_df.columns] + ["date", "symbol"]
    test = eval_df[keep].copy()
    for c in missing:
        test[c] = 0.0
    print(
        f"[{BUNDLE}] {len(cols)} 特征 (有效 {len(present)} + 补0 {len(missing)})",
        flush=True,
    )

    reg10, label10 = bundle["models"]["10d_reg"]
    result, test_pred = topn_metrics(test, reg10, cols)
    print("[pred] 10d_reg 预测完成", flush=True)

    out_dir = (
        Path(BACKTEST_RESULT_DIR) / f"main_353_topn_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    test_pred.to_parquet(out_dir / "eval_preds.parquet")
    print(f"[done] WORM -> {out_dir}", flush=True)

    print("\n=== main_20260812 (353 特征) 逐日 top-N 已实现收益/上涨率 ===", flush=True)
    for h in HORIZONS:
        r = result[f"{h}d"]
        print(
            f"--- {h}d 池基线: mean {r['pool']['mean_ret'] * 100:+.2f}%  hit {r['pool']['hit'] * 100:.1f}% "
            f"(rows {r['pool']['rows']:,})",
            flush=True,
        )
        for n in TOPN:
            t = r["topn"][str(n)]
            print(
                f"    top-{n:>3}: mean {t['mean_ret'] * 100:+.2f}%  hit {t['hit'] * 100:.1f}%  rows {t['rows']:,}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
