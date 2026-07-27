#!/usr/bin/env python3
"""检查模型特征列与预测面板的匹配情况"""
import os, sys, pickle, logging
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. 加载模型包，查看特征列
    model_dir = "models/pipeline1"
    for board in ["main", "dual"]:
        candidates = sorted([f for f in os.listdir(model_dir) if f.startswith(f"{board}_") and f.endswith(".pkl")])
        if not candidates:
            continue
        path = os.path.join(model_dir, candidates[-1])
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        
        feature_cols = bundle["feature_cols"]
        logger.info(f"\n=== {board} ({candidates[-1]}) ===")
        logger.info(f"特征列数量: {len(feature_cols)}")
        logger.info(f"前20个特征列: {feature_cols[:20]}")
        
    # 2. 加载面板数据
    panel = pd.read_parquet("data/panel_18m.parquet")
    logger.info(f"\n面板列: {list(panel.columns)}")
    logger.info(f"面板形状: {panel.shape}")
    
    # 3. 检查面板中是否包含模型需要的特征列
    for board in ["main", "dual"]:
        candidates = sorted([f for f in os.listdir(model_dir) if f.startswith(f"{board}_") and f.endswith(".pkl")])
        if not candidates:
            continue
        path = os.path.join(model_dir, candidates[-1])
        with open(path, 'rb') as f:
            bundle = pickle.load(f)
        
        feature_cols = bundle["feature_cols"]
        missing_cols = [c for c in feature_cols if c not in panel.columns]
        existing_cols = [c for c in feature_cols if c in panel.columns]
        
        logger.info(f"\n=== {board} 特征列匹配 ===")
        logger.info(f"存在: {len(existing_cols)}/{len(feature_cols)}")
        logger.info(f"缺失: {len(missing_cols)}")
        if missing_cols:
            logger.info(f"缺失列 (前20): {missing_cols[:20]}")
        
        # 4. 检查存在列的NaN率
        if existing_cols:
            latest = panel.sort_values("date").groupby("symbol").tail(1)
            nan_rates = latest[existing_cols].isna().mean()
            high_nan = nan_rates[nan_rates > 0.5].sort_values(ascending=False)
            logger.info(f"高NaN率特征 (>50%): {len(high_nan)}")
            if len(high_nan) > 0:
                logger.info(f"  前10: {high_nan.head(10).to_dict()}")
            
            all_nan = nan_rates[nan_rates > 0.99]
            logger.info(f"几乎全NaN特征 (>99%): {len(all_nan)}")
            if len(all_nan) > 0:
                logger.info(f"  列名: {list(all_nan.index[:20])}")
            
            # 检查特征值统计
            non_nan_cols = nan_rates[nan_rates < 0.5].index.tolist()
            if non_nan_cols:
                logger.info(f"有效特征 (NaN<50%): {len(non_nan_cols)}")
                sample = latest[non_nan_cols[:5]].describe()
                logger.info(f"前5个有效特征统计:\n{sample}")

if __name__ == "__main__":
    main()
