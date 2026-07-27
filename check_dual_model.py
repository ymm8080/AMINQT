#!/usr/bin/env python3
"""检查双创板模型坍缩根因"""
import os, sys, pickle, logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35

    # 1. 检查双创板模型的训练数据
    panel = pd.read_parquet("data/panel_18m.parquet")
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)

    logger.info(f"=== 清洗后样本数 ===")
    logger.info(f"  主板: {len(main_df)} 只, {main_df['symbol'].nunique()} stocks")
    logger.info(f"  双创: {len(dual_df)} 只, {dual_df['symbol'].nunique()} stocks")

    # 2. 检查双创板特征
    features = FeatureEngineV35()
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0:
            continue
        feats = features.build(df)
        latest = feats.sort_values("date").groupby("symbol").tail(1)

        logger.info(f"\n=== {board} 最新截面特征 ===")
        logger.info(f"  样本数: {len(latest)}")

        # 检查特征方差
        numeric_cols = latest.select_dtypes(include=[np.number]).columns
        feat_stds = latest[numeric_cols].std()
        zero_std = feat_stds[feat_stds < 1e-8]
        logger.info(f"  零方差特征: {len(zero_std)}")
        if len(zero_std) > 0:
            logger.info(f"  零方差列: {list(zero_std.index[:10])}")

    # 3. 检查模型包详情
    model_dir = "models/pipeline1"
    for board in ["main", "dual"]:
        candidates = sorted([f for f in os.listdir(model_dir) if f.startswith(f"{board}_") and f.endswith(".pkl")])
        if not candidates:
            continue
        path = os.path.join(model_dir, candidates[-1])
        with open(path, 'rb') as f:
            bundle = pickle.load(f)

        logger.info(f"\n=== {board} 模型详情 ({candidates[-1]}) ===")
        models = bundle["models"]
        feature_cols = bundle["feature_cols"]

        for kind in ("1d_reg", "3d_reg", "5d_reg", "1d_cls"):
            if kind not in models:
                continue
            model = models[kind][0]
            label = models[kind][1]
            logger.info(f"\n  --- {kind} (label={label}) ---")
            logger.info(f"  n_estimators: {model.n_estimators}")
            logger.info(f"  n_features_in_: {model.n_features_in_}")
            if hasattr(model, 'best_iteration_'):
                logger.info(f"  best_iteration: {model.best_iteration_}")

            # 特征重要性
            if hasattr(model, 'feature_importances_'):
                imps = model.feature_importances_
                top_idx = np.argsort(imps)[-10:][::-1]
                logger.info(f"  特征重要性前10:")
                for idx in top_idx:
                    logger.info(f"    {feature_cols[idx]}: {imps[idx]:.4f}")

                # 检查是否只有少数特征有贡献
                nonzero_imps = imps[imps > 0]
                logger.info(f"  非零重要性特征: {len(nonzero_imps)}/{len(imps)}")
                logger.info(f"  重要性总和: {imps.sum():.4f}")
                logger.info(f"  重要性集中度 (top5占比): {np.sort(imps)[-5:].sum()/imps.sum():.4f}")

    # 4. 检查训练窗口切分
    logger.info(f"\n=== 训练窗口检查 ===")
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0:
            continue
        dates = sorted(df["date"].unique())
        n_dates = len(dates)
        logger.info(f"\n  {board}: {n_dates} 个交易日")
        if n_dates >= 720:
            train_dates = dates[:690]
            es_dates = dates[690:700]
            calib_dates = dates[700:710]
            test_dates = dates[-10:]
            logger.info(f"  训练: {train_dates[0]} ~ {train_dates[-1]} ({len(train_dates)} 天)")
            logger.info(f"  早停: {es_dates[0]} ~ {es_dates[-1]} ({len(es_dates)} 天)")
            logger.info(f"  校准: {calib_dates[0]} ~ {calib_dates[-1]} ({len(calib_dates)} 天)")
            logger.info(f"  测试: {test_dates[0]} ~ {test_dates[-1]} ({len(test_dates)} 天)")
        else:
            logger.warning(f"  {board} 日期数不足 720 ({n_dates}), 窗口切分可能异常")

if __name__ == "__main__":
    main()
