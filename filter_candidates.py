#!/usr/bin/env python3
"""多维度筛选: 高收益 + 高概率(非饱和) + 低风险."""
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
        # 带 ATR 和 adv20
        latest = feats.sort_values("date").groupby("symbol").tail(1)
        for col in ["ATR_pct", "adv20", "turnover_rate", "amount"]:
            if col in latest.columns:
                pred[col] = latest.set_index("symbol").reindex(pred["symbol"])[col].values
        all_preds.append(pred)

    preds = pd.concat(all_preds, ignore_index=True)
    logger.info(f"总预测: {len(preds)} 只")

    # === 筛选条件 ===
    # 1. prob_up: 排除饱和的 1.0 (Isotonic 高端不靠谱), 取 0.55~0.95 区间
    # 2. pred_ret_3d > 1% (3日预期收益正)
    # 3. pred_ret_5d > 1% (5日预期收益正)
    # 4. pain_prob < 0.35 (痛苦预警低)
    # 5. ATR_pct < 0.06 (波动率不过高)

    mask = (
        (preds["prob_up"] >= 0.55) &
        (preds["prob_up"] < 0.99) &
        (preds["pred_ret_3d"] > 0.01) &
        (preds["pred_ret_5d"] > 0.01) &
        (preds["pain_prob"].fillna(1) < 0.35)
    )
    if "ATR_pct" in preds.columns:
        mask &= (preds["ATR_pct"].fillna(0.1) < 0.06)

    filtered = preds[mask].copy()
    logger.info(f"筛选后: {len(filtered)} 只")

    # 综合评分: prob_up * pred_ret_3d / (1 + pain_prob)
    filtered["score"] = (
        filtered["prob_up"] * filtered["pred_ret_3d"] / (1 + filtered["pain_prob"].fillna(0.3))
    )
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    # 展示
    cols = ["symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
            "pred_ret_5d", "prob_up", "pain_prob", "score"]
    if "ATR_pct" in filtered.columns:
        cols.append("ATR_pct")

    print("\n" + "=" * 100)
    print(f"多维度筛选结果 (高收益 + 高概率(非饱和) + 低风险): {len(filtered)} 只")
    print("条件: 0.55<=prob_up<0.99, pred_ret_3d>1%, pred_ret_5d>1%, pain_prob<0.35, ATR<6%")
    print("=" * 100)
    print(filtered[cols].to_string(index=False))

    # 保存
    filtered.to_csv("filtered_candidates.csv", index=False)
    print("\n已保存到 filtered_candidates.csv")

    # 分档统计
    print("\n--- 分档统计 ---")
    print(f"score > 0.005: {(filtered['score'] > 0.005).sum()}")
    print(f"score > 0.008: {(filtered['score'] > 0.008).sum()}")
    print(f"score > 0.010: {(filtered['score'] > 0.010).sum()}")

if __name__ == "__main__":
    main()
