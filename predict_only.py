#!/usr/bin/env python3
"""用已训练的 3y 模型跑预测, 不重训."""
import sys
import os
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAG = "2026W31_3y"

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
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)

    all_preds = []
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue
        feats = features.build(df, cross_sectional_rank=(board != "main"))
        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)
        pred["board"] = board

        # 带额外信息
        latest = feats.sort_values("date").groupby("symbol").tail(1)
        for col in ["ATR_pct", "adv20", "turnover_rate", "amount", "close"]:
            if col in latest.columns:
                pred[col] = latest.set_index("symbol").reindex(pred["symbol"])[col].values

        logger.info(f"{board}: {len(pred)} predictions")
        for c in ['pred_ret_1d','pred_ret_3d','pred_ret_5d','prob_up']:
            if c in pred.columns:
                v = pred[c].dropna()
                logger.info(f"  {c}: mean={v.mean():.6f} std={v.std():.6f} min={v.min():.6f} max={v.max():.6f}")

        all_preds.append(pred)

    preds = pd.concat(all_preds, ignore_index=True)

    # === 多维度筛选 ===
    # 高收益: pred_ret_3d > 1%, pred_ret_5d > 1%
    # 高概率: prob_up 0.55~0.95 (Platt 不会到1, 但仍排除极端)
    # 低风险: pain_prob < 0.35, ATR_pct < 0.06
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
    filtered["score"] = (
        filtered["prob_up"] * filtered["pred_ret_3d"] / (1 + filtered["pain_prob"].fillna(0.3))
    )
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    cols = ["symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
            "pred_ret_5d", "prob_up", "pain_prob", "score", "ATR_pct"]

    print("\n" + "=" * 100)
    print(f"候选清单 (高收益+高概率+低风险): {len(filtered)} 只 / 总 {len(preds)} 只")
    print("条件: 0.55<=prob_up<0.99, pred_ret_3d>1%, pred_ret_5d>1%, pain_prob<0.35, ATR<6%")
    print("=" * 100)
    print(filtered[cols].to_string(index=False))

    filtered.to_csv("filtered_candidates.csv", index=False)
    preds.to_csv("predictions_3y.csv", index=False)
    print("\n候选清单已保存 filtered_candidates.csv, 全量 predictions_3y.csv")

if __name__ == "__main__":
    main()
