#!/usr/bin/env python3
"""3年数据训练+预测完整流程."""
import sys, os, json, pickle, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAG = "2026W31_3y"

def main():
    from app.pipeline1.train_runner import run_training
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    # --- 训练 ---
    panel = pd.read_parquet("data/panel_3y.parquet")
    logger.info(f"Panel: {panel.shape}, {panel['symbol'].nunique()} stocks, "
                f"dates: {panel['date'].min()} ~ {panel['date'].max()}")

    results = run_training(panel, tag=TAG, use_ic_screen=True)

    for board, res in results.items():
        logger.info(f"[训练] {board}: OOS_IC(1d)={res['oos']['ics'].get('1d_reg', 0):.4f} "
                     f"feats={res['n_features']} switched={res['switched']}")

    # --- 预测 ---
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"\n清洗后: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)

    output = {"tag": TAG, "panel_shape": list(panel.shape), "boards": {}}

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            logger.info(f"{board} 跳过")
            continue

        feats = features.build(df)
        logger.info(f"\n{'='*60}")
        logger.info(f"{board} 预测 (features: {feats.shape})")
        logger.info(f"{'='*60}")

        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        board_out = {"pred_stats": {}, "model_info": {}, "oos": {}}

        for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
            if col in pred.columns:
                vals = pred[col].dropna()
                if len(vals) > 0:
                    stats = {
                        "mean": round(float(vals.mean()), 6),
                        "std": round(float(vals.std()), 6),
                        "min": round(float(vals.min()), 6),
                        "max": round(float(vals.max()), 6),
                        "n": int(len(vals)),
                    }
                    board_out["pred_stats"][col] = stats
                    logger.info(f"  {col}: mean={vals.mean():.6f} std={vals.std():.6f} "
                                f"min={vals.min():.6f} max={vals.max():.6f} n={len(vals)}")

        # 模型信息
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)

        for kind in ('1d_reg', '3d_reg', '5d_reg', '1d_cls'):
            if kind in bundle.get("models", {}):
                model = bundle["models"][kind][0]
                bi = getattr(model, 'best_iteration_', None)
                ne = model.n_estimators
                fi = model.feature_importances_
                nonzero = int((fi > 0).sum())
                top5_share = float(np.sort(fi)[::-1][:5].sum() / fi.sum()) if fi.sum() > 0 else 0
                board_out["model_info"][kind] = {
                    "best_iter": int(bi) if bi is not None else None,
                    "n_estimators": int(ne),
                    "n_features": int(model.n_features_in_) if hasattr(model, 'n_features_in_') else None,
                    "nonzero_features": nonzero,
                    "top5_share": round(top5_share, 4),
                }
                logger.info(f"  {kind}: best_iter={bi}/{ne} features={model.n_features_in_} "
                            f"nonzero={nonzero} top5={top5_share:.2%}")

        cal = bundle.get("calibrator")
        if cal:
            method = getattr(cal, 'method', type(cal).__name__)
            board_out["calibrator"] = method
            logger.info(f"  calibrator: {method}")

        oos = bundle.get("oos", {})
        if oos:
            ics = oos.get("ics", {})
            for k, v in ics.items():
                board_out["oos"][k] = round(float(v), 4)
                logger.info(f"  OOS IC({k}): {v:.4f}")

        output["boards"][board] = board_out

    out_path = f"result_{TAG}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\n完整结果已保存到 {out_path}")

if __name__ == "__main__":
    main()
