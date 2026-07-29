#!/usr/bin/env python3
"""用3年模型输出今日预测候选清单."""
import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    # 优先 enriched 面板 (3,227 stocks + alt data)
    for _p in ["data/panel_full_enriched_v3.parquet",
               "data/panel_full_enriched_v3.parquet"]:
        if os.path.exists(_p):
            panel = pd.read_parquet(_p)
            logger.info("加载面板: %s (%d stocks)", _p, panel["symbol"].nunique())
            break
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"清洗: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag="2026W31_3y")

    all_preds = []
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue
        feats = features.build(df)
        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)
        pred["board"] = board
        all_preds.append(pred)
        logger.info(f"{board}: {len(pred)} predictions")

    if not all_preds:
        logger.error("无预测结果")
        return

    preds = pd.concat(all_preds, ignore_index=True)

    # 排序: prob_up 降序
    preds = preds.sort_values("prob_up", ascending=False).reset_index(drop=True)

    # 输出 Top 30
    cols = ["symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
            "pred_ret_5d", "prob_up"]
    if "rank_score" in preds.columns:
        cols.append("rank_score")
    if "pain_prob" in preds.columns:
        cols.append("pain_prob")

    top30 = preds[cols].head(30)
    print("\n" + "=" * 80)
    print("Top 30 候选 (按 prob_up 降序)")
    print("=" * 80)
    print(top30.to_string(index=False))

    # 统计
    print(f"\n总预测数: {len(preds)}")
    print(f"prob_up > 0.5: {(preds['prob_up'] > 0.5).sum()}")
    print(f"prob_up > 0.6: {(preds['prob_up'] > 0.6).sum()}")
    print(f"pred_ret_1d > 0: {(preds['pred_ret_1d'] > 0).sum()}")
    print(f"pred_ret_3d > 0.01: {(preds['pred_ret_3d'] > 0.01).sum()}")

    # 保存
    preds.to_csv("predictions_3y.csv", index=False)
    print("\n完整预测已保存到 predictions_3y.csv")

if __name__ == "__main__":
    main()
