#!/usr/bin/env python3
"""重训并验证修复效果"""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import pandas as pd
    from app.pipeline1.train_runner import run_training

    panel = pd.read_parquet("data/panel_18m.parquet")
    logger.info(f"Panel: {panel.shape}, {panel['symbol'].nunique()} stocks")

    results = run_training(panel, tag="2026W31_fix", use_ic_screen=True)

    for board, res in results.items():
        oos_1d = res["oos"]["ics"].get("1d_reg", 0)
        logger.info(f"{board}: OOS_IC(1d)={oos_1d:.4f} switched={res['switched']} feats={res['n_features']}")

    # 验证预测值方差
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor
    import numpy as np

    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag="2026W31_fix")

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue
        feats = features.build(df)
        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        logger.info(f"\n=== {board} 修复后预测 ===")
        for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
            if col in pred.columns:
                vals = pred[col].dropna()
                logger.info(f"  {col}: mean={vals.mean():.6f} std={vals.std():.6f} min={vals.min():.6f} max={vals.max():.6f}")

        # 检查模型 best_iteration
        import pickle
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)
        for kind in ('1d_reg', '3d_reg', '5d_reg', '1d_cls'):
            if kind in bundle["models"]:
                model = bundle["models"][kind][0]
                bi = getattr(model, 'best_iteration_', None)
                ne = model.n_estimators
                logger.info(f"  {kind}: best_iteration={bi} n_estimators={ne}")

        # 校准器类型
        cal = bundle.get("calibrator")
        if cal:
            logger.info(f"  calibrator method: {cal.method}")

if __name__ == "__main__":
    main()
