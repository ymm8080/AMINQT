#!/usr/bin/env python3
"""
Pipeline1 Prediction Quality Diagnostic
======================================
诊断脚本：检查当前训练和预测质量问题的根本原因
"""

import os
import sys
import pickle
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_model_bundles():
    """检查模型包状态"""
    logger.info("=== 检查模型包状态 ===")
    model_dir = "models/pipeline1"
    
    if not os.path.exists(model_dir):
        logger.error(f"模型目录不存在: {model_dir}")
        return False
        
    bundles = {}
    for board in ["main", "dual"]:
        candidates = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith(f"{board}_") and f.endswith(".pkl")
        ])
        if candidates:
            latest = candidates[-1]
            path = os.path.join(model_dir, latest)
            bundles[board] = path
            logger.info(f"[{board}] 最新模型: {latest}")
            
            # 检查模型内容
            try:
                with open(path, 'rb') as f:
                    model_data = pickle.load(f)
                
                feature_count = len(model_data.get("feature_cols", []))
                models = model_data.get("models", {})
                
                logger.info(f"  - 特征数量: {feature_count}")
                logger.info(f"  - 模型类型: {list(models.keys())}")
                
                # 检查特征重要性
                if "1d_reg" in models:
                    reg_model = models["1d_reg"][0]
                    if hasattr(reg_model, 'feature_importances_'):
                        top_features = np.argsort(reg_model.feature_importances_)[-5:][::-1]
                        logger.info(f"  - 重要特征前5: {top_features}")
                        
            except Exception as e:
                logger.error(f"  - 模型加载失败: {e}")
                return False
        else:
            logger.warning(f"[{board}] 无可用模型")
            
    return len(bundles) > 0

def check_recent_predictions():
    """检查最新预测结果"""
    logger.info("\\n=== 检查最新预测结果 ===")
    list_dir = "data/lists"
    
    if not os.path.exists(list_dir):
        logger.error(f"预测列表目录不存在: {list_dir}")
        return False
        
    # 获取最新预测文件
    parquet_files = [f for f in os.listdir(list_dir) if f.endswith('.parquet')]
    if not parquet_files:
        logger.error("无预测文件")
        return False
        
    latest_file = sorted(parquet_files)[-1]
    logger.info(f"最新预测文件: {latest_file}")
    
    try:
        df = pd.read_parquet(os.path.join(list_dir, latest_file))
        logger.info(f"预测数量: {len(df)}")
        
        if len(df) == 0:
            logger.warning("预测列表为空")
            return False
            
        # 检查预测值分布
        pred_cols = ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']
        for col in pred_cols:
            if col in df.columns:
                values = df[col].dropna()
                logger.info(f"  {col}: mean={values.mean():.4f}, std={values.std():.4f}, min={values.min():.4f}, max={values.max():.4f}")
        
        # 检查预测极端情况
        extreme_low = df[df['pred_ret_1d'] < -0.1]
        extreme_high = df[df['pred_ret_1d'] > 0.1]
        logger.info(f"  极端预测: 极低({len(extreme_low)}), 极高({len(extreme_high)})")
        
        return True
        
    except Exception as e:
        logger.error(f"预测文件读取失败: {e}")
        return False

def check_data_panel():
    """检查数据面板"""
    logger.info("\\n=== 检查数据面板 ===")
    panel_path = "data/panel_18m.parquet"
    
    if not os.path.exists(panel_path):
        logger.error(f"面板文件不存在: {panel_path}")
        return False
        
    try:
        panel = pd.read_parquet(panel_path)
        logger.info(f"面板形状: {panel.shape}")
        logger.info(f"股票数量: {panel['symbol'].nunique()}")
        
        # 检查日期范围
        if 'date' in panel.columns:
            dates = pd.to_datetime(panel['date'])
            logger.info(f"日期范围: {dates.min()} 到 {dates.max()}")
            
        # 检查缺失值
        missing_summary = {}
        for col in panel.columns[:10]:  # 检查前10列
            missing_rate = panel[col].isna().mean()
            if missing_rate > 0.1:
                missing_summary[col] = missing_rate
                
        if missing_summary:
            logger.warning(f"高缺失率列: {missing_summary}")
            
        return True
        
    except Exception as e:
        logger.error(f"面板文件读取失败: {e}")
        return False

def check_forecast_accuracy():
    """检查预测准确度评估"""
    logger.info("\\n=== 检查预测准确度评估 ===")
    accuracy_dir = "data/forecast_accuracy"
    
    if not os.path.exists(accuracy_dir):
        logger.warning(f"准确度评估目录不存在: {accuracy_dir}")
        return False
        
    json_files = [f for f in os.listdir(accuracy_dir) if f.startswith('accuracy_') and f.endswith('.json')]
    if not json_files:
        logger.warning("无准确度评估文件")
        return False
        
    # 检查最新评估
    latest = sorted(json_files)[-1]
    logger.info(f"最新准确度评估: {latest}")
    
    try:
        import json
        with open(os.path.join(accuracy_dir, latest), 'r') as f:
            result = json.load(f)
            
        logger.info(f"预测日期: {result.get('forecast_date')}")
        
        for horizon, metrics in result.get('horizons', {}).items():
            if isinstance(horizon, int) or horizon.isdigit():
                h = int(horizon) if isinstance(horizon, str) else horizon
                logger.info(f"  {h}d: MAE={metrics.get('mae_1d', 'N/A'):.4f}, "
                           f"bias={metrics.get('bias_1d', 'N/A'):+.4f}, "
                           f"方向准确率={metrics.get('direction_accuracy', 'N/A'):.3f}, "
                           f"样本数={metrics.get('n_samples', 0)}")
                
                # 检查分桶偏差
                bias_buckets = {k: v for k, v in metrics.items() if k.startswith('bias_') and k != 'bias_1d'}
                for bucket, bias_val in bias_buckets.items():
                    if not np.isnan(bias_val):
                        logger.info(f"    {bucket}: {bias_val:+.4f}")
                        
        return True
        
    except Exception as e:
        logger.error(f"准确度评估文件读取失败: {e}")
        return False

def check_feature_engineering():
    """检查特征工程"""
    logger.info("\\n=== 检查特征工程 ===")
    
    try:
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35
        
        # 检查特征引擎版本
        logger.info(f"特征引擎版本: V3.5")
        
        # 创建特征引擎实例
        engine = FeatureEngineV35()
        
        # 检查特征维度常量
        logger.info(f"移动平均窗口: {engine.MA_WINDOWS}")
        logger.info(f"行业中性化列: {engine.NEUTRALIZE_COLS}")
        logger.info(f"关键因子缺失指示: {engine.MISSINGNESS_COLS}")
        
        return True
        
    except Exception as e:
        logger.error(f"特征引擎检查失败: {e}")
        return False

def check_training_config():
    """检查训练配置"""
    logger.info("\\n=== 检查训练配置 ===")
    
    try:
        from app.pipeline1.dual_track_trainer import (
            WINDOW_TOTAL, TRAIN_DAYS, ES_DAYS, CALIB_DAYS, TEST_DAYS,
            HALF_LIFE, ES_PATIENCE, OOS_IC_MIN
        )
        
        logger.info(f"训练窗口配置:")
        logger.info(f"  - 总窗口: {WINDOW_TOTAL} 天")
        logger.info(f"  - 训练: {TRAIN_DAYS} 天")
        logger.info(f"  - 早停: {ES_DAYS} 天")
        logger.info(f"  - 校准: {CALIB_DAYS} 天")
        logger.info(f"  - 测试: {TEST_DAYS} 天")
        logger.info(f"  - 半衰期: {HALF_LIFE} 天")
        logger.info(f"  - 早停耐心: {ES_PATIENCE}")
        logger.info(f"  - OOS IC 门槛: {OOS_IC_MIN}")
        
        return True
        
    except Exception as e:
        logger.error(f"训练配置检查失败: {e}")
        return False

def main():
    """主诊断函数"""
    logger.info("开始 Pipeline1 预测质量诊断")
    logger.info("=" * 50)
    
    checks = [
        ("模型包状态", check_model_bundles),
        ("最新预测", check_recent_predictions),
        ("数据面板", check_data_panel),
        ("预测准确度", check_forecast_accuracy),
        ("特征工程", check_feature_engineering),
        ("训练配置", check_training_config),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            logger.error(f"{name} 检查失败: {e}")
            results[name] = False
            
    logger.info("\\n" + "=" * 50)
    logger.info("诊断总结:")
    for name, result in results.items():
        status = "✓" if result else "✗"
        logger.info(f"  {status} {name}")
        
    # 生成建议
    logger.info("\\n建议:")
    if not results.get("模型包状态"):
        logger.info("  - 检查模型训练是否成功完成")
    if not results.get("最新预测"):
        logger.info("  - 检查预测流程是否正常运行")
    if not results.get("预测准确度"):
        logger.info("  - 运行预测准确度评估: python -c 'from app.pipeline1.forecast_accuracy import score_matured_forecasts; score_matured_forecasts(\"data/lists\", pd.read_parquet(\"data/panel_18m.parquet\"))'")
    
    logger.info("\\n诊断完成")

if __name__ == "__main__":
    main()