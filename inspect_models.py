#!/usr/bin/env python3
"""快速检查已训练模型的关键指标"""
import sys, os, pickle, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def inspect_bundle(path, tag):
    """检查单个模型 bundle"""
    with open(path, 'rb') as f:
        bundle = pickle.load(f)

    logger.info(f"\n{'='*60}")
    logger.info(f"模型: {tag} ({os.path.basename(path)})")
    logger.info(f"{'='*60}")

    # 模型信息
    for kind in ('1d_reg', '3d_reg', '5d_reg', '1d_cls'):
        if kind in bundle.get("models", {}):
            model_info = bundle["models"][kind]
            model = model_info[0]
            bi = getattr(model, 'best_iteration_', None)
            ne = model.n_estimators
            n_feat = model.n_features_in_ if hasattr(model, 'n_features_in_') else '?'
            logger.info(f"  {kind}: best_iter={bi}/{ne}  n_features={n_feat}")

            # 特征重要性集中度
            fi = model.feature_importances_
            if fi is not None and len(fi) > 0:
                fi_sorted = np.sort(fi)[::-1]
                top5_share = fi_sorted[:5].sum() / fi.sum() if fi.sum() > 0 else 0
                top10_share = fi_sorted[:10].sum() / fi.sum() if fi.sum() > 0 else 0
                nonzero = (fi > 0).sum()
                logger.info(f"    feat_importance: nonzero={nonzero}/{len(fi)} top5_share={top5_share:.2%} top10_share={top10_share:.2%}")

    # 校准器
    cal = bundle.get("calibrator")
    if cal:
        method = getattr(cal, 'method', type(cal).__name__)
        logger.info(f"  calibrator: {method}")
    else:
        logger.info(f"  calibrator: None")

    # 窗口信息
    for key in ('train_dates', 'es_dates', 'calib_dates', 'test_dates'):
        if key in bundle:
            dates = bundle[key]
            if hasattr(dates, '__len__'):
                logger.info(f"  {key}: {len(dates)} dates")

    # OOS 信息
    oos = bundle.get("oos", {})
    if oos:
        ics = oos.get("ics", {})
        for k, v in ics.items():
            logger.info(f"  OOS IC({k}): {v:.4f}")

    return bundle


def check_prediction_variance(panel_path, tag):
    """检查预测值方差"""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    panel = pd.read_parquet(panel_path)
    logger.info(f"\nPanel: {panel.shape}, {panel['symbol'].nunique()} stocks, "
                f"dates: {panel['date'].min()} ~ {panel['date'].max()}")

    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"清洗后: main={len(main_df)} dual={len(dual_df)}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=tag)

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            logger.info(f"\n=== {board} 跳过 (df={len(df)}, bundle={'有' if board in bundles else '无'}) ===")
            continue

        feats = features.build(df)
        logger.info(f"\n=== {board} 预测 (features: {feats.shape}) ===")

        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
            if col in pred.columns:
                vals = pred[col].dropna()
                if len(vals) > 0:
                    logger.info(f"  {col}: mean={vals.mean():.6f} std={vals.std():.6f} "
                                f"min={vals.min():.6f} max={vals.max():.6f} n={len(vals)}")
                else:
                    logger.info(f"  {col}: 全部为 NaN")
            else:
                logger.info(f"  {col}: 不存在")


if __name__ == "__main__":
    # 检查 fix 版本模型
    for board, path in [
        ("main", "models/pipeline1/main_2026W31_fix.pkl"),
        ("dual", "models/pipeline1/dual_2026W31_fix.pkl"),
    ]:
        if os.path.exists(path):
            inspect_bundle(path, board)
        else:
            logger.warning(f"模型文件不存在: {path}")

    # 检查预测方差
    panel_path = "data/panel_18m.parquet"
    if os.path.exists(panel_path):
        check_prediction_variance(panel_path, "2026W31_fix")
    else:
        logger.error(f"Panel 文件不存在: {panel_path}")
