#!/usr/bin/env python3
"""检查校准器和原始模型输出"""
import os, sys, pickle, logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles

    panel = pd.read_parquet("data/panel_18m.parquet")
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1")

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"=== {board} ===")
        logger.info(f"{'='*60}")

        # 构建特征
        feats = features.build(df)
        latest = feats.sort_values("date").groupby("symbol").tail(1).copy()

        # 加载模型包
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)

        cols = bundle["feature_cols"]
        X = np.nan_to_num(latest[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

        # 原始模型输出
        models = bundle["models"]
        raw_pred_1d = models["1d_reg"][0].predict(X)
        raw_prob = models["1d_cls"][0].predict_proba(X)[:, 1]

        logger.info(f"样本数: {len(latest)}")
        logger.info(f"\n--- 原始回归预测 (pred_ret_1d) ---")
        logger.info(f"  mean={raw_pred_1d.mean():.6f} std={raw_pred_1d.std():.6f}")
        logger.info(f"  min={raw_pred_1d.min():.6f} max={raw_pred_1d.max():.6f}")
        logger.info(f"  分位数: 10%={np.percentile(raw_pred_1d,10):.6f} 50%={np.percentile(raw_pred_1d,50):.6f} 90%={np.percentile(raw_pred_1d,90):.6f}")

        logger.info(f"\n--- 原始分类概率 (raw predict_proba) ---")
        logger.info(f"  mean={raw_prob.mean():.6f} std={raw_prob.std():.6f}")
        logger.info(f"  min={raw_prob.min():.6f} max={raw_prob.max():.6f}")
        logger.info(f"  分位数: 10%={np.percentile(raw_prob,10):.6f} 50%={np.percentile(raw_prob,50):.6f} 90%={np.percentile(raw_prob,90):.6f}")

        # 校准后概率
        calibrator = bundle["calibrator"]
        cal_method = getattr(calibrator, 'method', 'unknown')
        logger.info(f"\n--- 校准器类型: {cal_method} ---")

        calibrated_prob = calibrator.predict_proba(raw_prob)
        logger.info(f"  mean={calibrated_prob.mean():.6f} std={calibrated_prob.std():.6f}")
        logger.info(f"  min={calibrated_prob.min():.6f} max={calibrated_prob.max():.6f}")
        logger.info(f"  分位数: 10%={np.percentile(calibrated_prob,10):.6f} 50%={np.percentile(calibrated_prob,50):.6f} 90%={np.percentile(calibrated_prob,90):.6f}")

        # 校准器映射关系
        if cal_method == 'isotonic' and hasattr(calibrator, '_iso') and calibrator._iso is not None:
            iso = calibrator._iso
            logger.info(f"\n--- Isotonic 校准映射 ---")
            # 生成从0到1的网格看映射关系
            grid = np.linspace(0, 1, 21)
            mapped = iso.predict(grid)
            for g, m in zip(grid, mapped):
                logger.info(f"  raw={g:.2f} -> calibrated={m:.4f}")
        elif cal_method == 'platt' and hasattr(calibrator, '_lr') and calibrator._lr is not None:
            lr = calibrator._lr
            grid = np.linspace(0, 1, 21)
            mapped = lr.predict_proba(grid.reshape(-1,1))[:,1]
            logger.info(f"\n--- Platt 校准映射 ---")
            for g, m in zip(grid, mapped):
                logger.info(f"  raw={g:.2f} -> calibrated={m:.4f}")

        # 检查准入门槛
        logger.info(f"\n--- 准入门槛分析 ---")
        for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
            n_pass = (calibrated_prob > thresh).sum()
            logger.info(f"  prob_up > {thresh:.2f}: {n_pass} 只 ({n_pass/len(calibrated_prob)*100:.1f}%)")

        ret_thresh = 0.0013  # entry_ret_mult=1.0 * COST
        n_ret_pass = (raw_pred_1d > ret_thresh).sum()
        logger.info(f"  pred_ret_1d > {ret_thresh:.4f}: {n_ret_pass} 只 ({n_ret_pass/len(raw_pred_1d)*100:.1f}%)")

        both = ((calibrated_prob > 0.45) & (raw_pred_1d > ret_thresh)).sum()
        logger.info(f"  同时满足 prob>0.45 且 ret>0.0013: {both} 只")

        # 检查训练标签分布
        logger.info(f"\n--- 训练标签分布 (特征面板) ---")
        if "label_cls" in feats.columns:
            label_vals = feats["label_cls"].dropna()
            pos_rate = label_vals.mean()
            logger.info(f"  label_cls 正样本率: {pos_rate:.4f} ({len(label_vals)} 样本)")
        if "label_1d" in feats.columns:
            l1d = feats["label_1d"].dropna()
            logger.info(f"  label_1d: mean={l1d.mean():.6f} std={l1d.std():.6f}")
            logger.info(f"  label_1d > 0.005: {(l1d > 0.005).mean():.4f}")
            logger.info(f"  label_1d > 0: {(l1d > 0).mean():.4f}")

if __name__ == "__main__":
    main()
