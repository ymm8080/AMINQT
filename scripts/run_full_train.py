#!/usr/bin/env python3
"""全量面板训练+预测 (3227 只 + CYQ 筹码特征).

Usage:
    python scripts/run_full_train.py
"""

import json
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predict_runner import find_bundles
from app.pipeline1.predictor import V35Predictor
from app.pipeline1.train_runner import run_training
from config.settings import PANEL_V3_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TAG = "2026W32_full"
PANEL_PATH = str(PANEL_V3_PATH)


def main():
    # ---- 1. Load enriched panel (OHLCV + CYQ + metadata) ----
    panel = pd.read_parquet(PANEL_PATH)
    n_stocks = panel["symbol"].nunique()
    cyq_cov = (
        panel["pct_90_con"].notna().mean() * 100 if "pct_90_con" in panel.columns else 0
    )
    logger.info(
        f"Panel: {len(panel)} rows, {n_stocks} stocks, CYQ coverage={cyq_cov:.1f}%"
    )

    # ---- 2. Train ----
    CleaningConfig(min_list_days=180)  # 降低阈值适配200天面板
    results = run_training(panel, tag=TAG, use_ic_screen=True)
    for board, res in results.items():
        logger.info(
            f"[训练] {board}: OOS_IC(1d)={res['oos']['ics'].get('1d_reg', 0):.4f} "
            f"feats={res['n_features']} switched={res['switched']}"
        )

    # ---- 3. Predict ----
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"清洗后: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)

    output = {"tag": TAG, "panel_shape": list(panel.shape), "boards": {}}

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue

        feats = features.build(df, cross_sectional_rank=(board != "main"))
        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        # Feature importance
        with open(bundles[board], "rb") as f:
            bundle = pickle.load(f)

        board_out = {"pred_stats": {}, "model_info": {}, "oos": {}, "top_features": []}

        # Prediction stats
        for col in ["pred_ret_1d", "pred_ret_3d", "pred_ret_5d", "prob_up"]:
            if col in pred.columns:
                vals = pred[col].dropna()
                if len(vals):
                    board_out["pred_stats"][col] = {
                        "mean": round(float(vals.mean()), 6),
                        "std": round(float(vals.std()), 6),
                        "min": round(float(vals.min()), 6),
                        "max": round(float(vals.max()), 6),
                        "n": int(len(vals)),
                    }

        # Model info
        for kind in ("1d_reg", "3d_reg", "5d_reg", "1d_cls"):
            if kind in bundle.get("models", {}):
                model = bundle["models"][kind][0]
                fi = model.feature_importances_
                names = (
                    model.feature_name_
                    if hasattr(model, "feature_name_") and model.feature_name_
                    else []
                )
                idx = np.argsort(fi)[::-1]
                board_out["model_info"][kind] = {
                    "best_iter": int(getattr(model, "best_iteration_", -1)),
                    "nonzero": int((fi > 0).sum()),
                    "top5_share": round(
                        float(np.sort(fi)[::-1][:5].sum() / fi.sum()), 4
                    )
                    if fi.sum() > 0
                    else 0,
                }
                # Top 10 features
                if names:
                    board_out["top_features"] = [
                        {
                            "rank": i + 1,
                            "name": names[idx[i]],
                            "imp": round(float(fi[idx[i]]), 4),
                        }
                        for i in range(min(10, len(idx)))
                    ]

        oos = bundle.get("oos", {})
        for k, v in oos.get("ics", {}).items():
            board_out["oos"][k] = round(float(v), 4)

        output["boards"][board] = board_out
        logger.info(
            f"[预测] {board}: pred_ret_1d n={board_out['pred_stats'].get('pred_ret_1d', {}).get('n', 0)}"
        )

    # ---- 4. Save ----
    out_path = f"result_{TAG}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    # ---- 5. Quality Summary ----
    for board, brd in output["boards"].items():
        oos_1d = brd.get("oos", {}).get("1d_reg")
        if oos_1d is not None:
            quality = (
                "✅ GOOD"
                if oos_1d >= 0.03
                else "⚠️ MARGINAL"
                if oos_1d > 0
                else "❌ BAD"
            )
            logger.info(f"[质量] {board}: OOS IC(1d)={oos_1d:.4f} {quality}")


if __name__ == "__main__":
    main()
