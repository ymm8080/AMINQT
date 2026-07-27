#!/usr/bin/env python3
"""A/B 对照实验: 同一份数据, 改前 vs 改后, 对比预测方差和 OOS IC.

用法:
  python ab_test.py after   # 用当前代码跑, 结果存 ab_after.json
  python ab_test.py before  # 用当前代码跑, 结果存 ab_before.json
"""
import sys, os, json, logging, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PANEL_PATH = "data/panel_18m.parquet"
TAG_PREFIX = "ab_test_"


def run_experiment(label: str):
    from app.pipeline1.train_runner import run_training
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    panel = pd.read_parquet(PANEL_PATH)
    logger.info(f"[{label}] Panel: {panel.shape}, {panel['symbol'].nunique()} stocks, "
                f"dates: {panel['date'].min()} ~ {panel['date'].max()}")

    # 训练
    tag = f"{TAG_PREFIX}{label}"
    results = run_training(panel, tag=tag, use_ic_screen=True)

    output = {"label": label, "boards": {}}

    for board, res in results.items():
        board_info = {
            "oos_ics": res["oos"]["ics"],
            "switched": res["switched"],
            "n_features": res["n_features"],
        }
        output["boards"][board] = board_info
        logger.info(f"[{label}] {board}: OOS_IC(1d)={res['oos']['ics'].get('1d_reg', 0):.4f} "
                     f"feats={res['n_features']} switched={res['switched']}")

    # 预测方差
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=tag)

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            output["boards"].setdefault(board, {})["pred_stats"] = {"n": 0}
            continue
        feats = features.build(df)
        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        pred_stats = {}
        for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
            if col in pred.columns:
                vals = pred[col].dropna()
                if len(vals) > 0:
                    pred_stats[col] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()),
                        "min": float(vals.min()),
                        "max": float(vals.max()),
                        "n": int(len(vals)),
                    }
        output["boards"][board]["pred_stats"] = pred_stats

        # 模型 best_iteration
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)
        iter_info = {}
        for kind in ('1d_reg', '3d_reg', '5d_reg', '1d_cls'):
            if kind in bundle.get("models", {}):
                model = bundle["models"][kind][0]
                bi = getattr(model, 'best_iteration_', None)
                ne = model.n_estimators
                iter_info[kind] = {"best_iter": bi, "n_estimators": ne}
        output["boards"][board]["model_iters"] = iter_info

        # 校准器
        cal = bundle.get("calibrator")
        if cal:
            output["boards"][board]["calibrator"] = getattr(cal, 'method', type(cal).__name__)

        logger.info(f"[{label}] {board} 预测: {pred_stats}")

    out_path = f"ab_{label}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"[{label}] 结果已保存到 {out_path}")
    return output


if __name__ == "__main__":
    label = sys.argv[1] if len(sys.argv) > 1 else "test"
    run_experiment(label)
