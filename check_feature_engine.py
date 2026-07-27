#!/usr/bin/env python3
"""验证特征引擎是否正确产出模型需要的特征列，并检查预测值方差"""
import os, sys, pickle, logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predictor import V35Predictor
    from app.pipeline1.predict_runner import find_bundles

    # 1. 加载面板
    panel = pd.read_parquet("data/panel_18m.parquet")
    logger.info(f"Panel: {panel.shape}, {panel['symbol'].nunique()} stocks")

    # 2. 清洗
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"Cleaning: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    if len(main_df) == 0 and len(dual_df) == 0:
        logger.error("清洗后无样本!")
        return

    # 3. 特征工程
    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1")
    logger.info(f"Bundles: {list(bundles.keys())}")

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue

        logger.info(f"\n=== {board} ===")
        logger.info(f"清洗后样本: {len(df)}")

        # 运行特征引擎
        try:
            feats = features.build(df)
            logger.info(f"特征引擎输出: {feats.shape}")
            logger.info(f"特征列数: {len(feats.columns)}")
        except Exception as e:
            logger.error(f"特征引擎失败: {e}")
            import traceback; traceback.print_exc()
            continue

        # 加载模型包
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)
        model_cols = bundle["feature_cols"]
        logger.info(f"模型特征列数: {len(model_cols)}")

        # 检查匹配
        missing = [c for c in model_cols if c not in feats.columns]
        existing = [c for c in model_cols if c in feats.columns]
        logger.info(f"匹配: {len(existing)}/{len(model_cols)}")
        if missing:
            logger.warning(f"缺失特征列 ({len(missing)}): {missing[:15]}")

        if not existing:
            logger.error("无匹配特征列! 模型将收到全零输入")
            continue

        # 检查特征值
        latest = feats.sort_values("date").groupby("symbol").tail(1)
        X_raw = latest[existing]
        nan_rate = X_raw.isna().mean()
        all_nan_cols = nan_rate[nan_rate > 0.99].index.tolist()
        logger.info(f"几乎全NaN的特征列 (>99%): {len(all_nan_cols)}")
        if all_nan_cols:
            logger.warning(f"全NaN列: {all_nan_cols[:10]}")

        # 模拟预测器行为
        X = np.nan_to_num(latest[existing].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        logger.info(f"特征矩阵: shape={X.shape}")
        logger.info(f"特征矩阵 std (per col, 前10): {X.std(axis=0)[:10]}")
        logger.info(f"特征矩阵非零比例: {(X != 0).mean():.4f}")

        # 实际预测
        predictor = V35Predictor(bundles)
        try:
            pred = predictor.predict(feats, board)
            logger.info(f"预测结果: {len(pred)} 只")
            for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
                if col in pred.columns:
                    vals = pred[col].dropna()
                    logger.info(f"  {col}: mean={vals.mean():.6f} std={vals.std():.6f} min={vals.min():.6f} max={vals.max():.6f}")
        except Exception as e:
            logger.error(f"预测失败: {e}")
            import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
