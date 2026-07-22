# -*- coding: utf-8 -*-
"""测试同花顺指标计算 + factor_engine 集成.

验证项：
  1. ths_indicators.add_all_ths_indicators() 输出 41 列
  2. 所有列无 NaN/inf
  3. 二值衰减信号在 0~1 之间
  4. factor_engine.build_features() 输出形状正确
  5. X 无 NaN
"""

import sys
import os
import logging

# 添加项目根目录到 path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_ohlcv(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成合成 OHLCV 数据用于测试.

    Args:
        n_days: 天数.
        seed: 随机种子.

    Returns:
        DataFrame with date, open, close, high, low, volume.
    """
    rng = np.random.default_rng(seed)
    # 随机游走 + 轻微上涨趋势
    returns = rng.normal(0.001, 0.02, n_days)
    close = 100.0 * np.cumprod(1.0 + returns)

    # OHLC 从 close 派生
    open_ = close * (1.0 + rng.normal(0, 0.005, n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.01, n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.01, n_days)))
    volume = rng.integers(1_000_000, 50_000_000, n_days).astype(float)

    dates = pd.bdate_range(start="2023-01-01", periods=n_days)

    df = pd.DataFrame({
        "date": dates,
        "open": open_,
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
    })
    return df


def test_ths_indicators():
    """测试同花顺指标模块."""
    from app.core.ths_indicators import (
        add_all_ths_indicators, THS_FACTOR_COLUMNS,
        ths_ema, ths_sma, ths_cross, ths_hhv, ths_llv, ths_ref,
        exp_decay_encode, safe_divide,
    )

    print("\n" + "=" * 60)
    print("TEST 1: THS Indicators Module (ths_indicators.py)")
    print("=" * 60)

    df = generate_synthetic_ohlcv(300)
    print(f"Input: {len(df)} rows, cols={list(df.columns)}")

    df = add_all_ths_indicators(df)

    # Check 1: all columns present
    missing = [c for c in THS_FACTOR_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"
    print(f"[OK] All {len(THS_FACTOR_COLUMNS)} columns generated")

    # Check 2: no NaN/inf
    for col in THS_FACTOR_COLUMNS:
        vals = df[col]
        assert not vals.isna().any(), f"{col} has NaN"
        assert not np.isinf(vals).any(), f"{col} has inf"
    print("[OK] No NaN/inf")

    # Check 3: decay signals in [0, 1]
    decay_cols = [c for c in THS_FACTOR_COLUMNS if "decay" in c]
    for col in decay_cols:
        vals = df[col]
        assert vals.min() >= 0, f"{col} min < 0: {vals.min()}"
        assert vals.max() <= 1.01, f"{col} max > 1: {vals.max()}"
    print(f"[OK] {len(decay_cols)} decay signals in [0, 1]")

    # Check 4: THS function correctness
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # EMA(X, 5) first value = X[0] (adjust=False)
    assert abs(ths_ema(s, 5).iloc[0] - 1.0) < 1e-10, "EMA first value should be X[0]"
    # SMA(X, 3, 1) first value = X[0]
    assert abs(ths_sma(s, 3, 1).iloc[0] - 1.0) < 1e-10, "SMA first value should be X[0]"
    # HHV(X, 3) at [3,4,5] should be 5
    assert ths_hhv(s, 3).iloc[4] == 5.0, "HHV error"
    # LLV(X, 3) at [3,4,5] should be 3
    assert ths_llv(s, 3).iloc[4] == 3.0, "LLV error"
    # REF(X, 1) at index=1 should be X[0]
    assert ths_ref(s, 1).iloc[1] == 1.0, "REF error"
    # CROSS: ascending series crosses constant
    a = pd.Series([1.0, 2.0, 3.0, 4.0])
    b = pd.Series([2.0, 2.0, 2.0, 2.0])
    cross_result = ths_cross(a, b)
    assert cross_result.iloc[2] == True, "CROSS should trigger at index=2 (a=3 > b=2)"
    assert cross_result.iloc[0] == False, "CROSS should not trigger at index=0"
    assert cross_result.iloc[1] == False, "CROSS should not trigger at index=1 (a=2 = b=2, not >)"
    print("[OK] THS functions (EMA/SMA/HHV/LLV/REF/CROSS) correct")

    # Check 5: trend line range
    for col in ["tech_ths_trend_short", "tech_ths_trend_mid", "tech_ths_trend_long"]:
        vals = df[col].iloc[50:]  # skip warmup
        assert vals.min() >= -10, f"{col} abnormally low: {vals.min()}"
        assert vals.max() <= 210, f"{col} abnormally high: {vals.max()}"
    print("[OK] Trend line values in normal range")

    # Check 6: 主力资金流向因子 (新增 8 列)
    flow_cols = [c for c in THS_FACTOR_COLUMNS if c.startswith("tech_ths_flow_")]
    assert len(flow_cols) == 8, f"Expected 8 flow columns, got {len(flow_cols)}"
    for col in flow_cols:
        assert not df[col].isna().any(), f"{col} has NaN"
    # flow_divergence 应在 {-1, 0, 1}
    assert set(df["tech_ths_flow_divergence"].unique()).issubset({-1.0, 0.0, 1.0})
    print(f"[OK] {len(flow_cols)} 主力资金流向因子 valid, divergence in {{-1, 0, 1}}")

    # Check 7: 控盘增强因子 (新增 3 列)
    ctrl_enh_cols = ["tech_ths_ctrl_ratio", "tech_ths_ctrl_concentration", "tech_ths_ctrl_change"]
    for col in ctrl_enh_cols:
        assert col in df.columns, f"{col} missing"
        assert not df[col].isna().any(), f"{col} has NaN"
    # ctrl_ratio 应在 [0, 1]
    assert df["tech_ths_ctrl_ratio"].min() >= 0 and df["tech_ths_ctrl_ratio"].max() <= 1.01
    print(f"[OK] {len(ctrl_enh_cols)} 控盘增强因子 valid, ctrl_ratio in [0, 1]")

    print("\n[PASS] TEST 1\n")
    return df


def test_factor_engine():
    """测试 factor_engine 集成."""
    from app.core.factor_engine import build_features, get_feature_names

    print("\n" + "=" * 60)
    print("TEST 2: factor_engine.build_features() Integration")
    print("=" * 60)

    df = generate_synthetic_ohlcv(300)

    X, y = build_features(df)

    # Check 1: X shape
    assert X.ndim == 3, f"X should be 3D, got {X.ndim}"
    assert X.shape[1] == 20, f"Time window should be 20, got {X.shape[1]}"
    assert X.shape[2] >= 60, f"Feature count should be >= 60, got {X.shape[2]}"
    print(f"[OK] X.shape = {X.shape} (N, 20, {X.shape[2]})")

    # Check 2: y shape
    assert y.ndim == 1, f"y should be 1D, got {y.ndim}"
    assert len(X) == len(y), f"X({len(X)}) and y({len(y)}) length mismatch"
    print(f"[OK] y.shape = {y.shape}")

    # Check 3: no NaN
    assert not np.any(np.isnan(X)), "X has NaN"
    assert not np.any(np.isnan(y)), "y has NaN"
    print("[OK] No NaN")

    # Check 4: feature names
    feat_names = get_feature_names()
    assert len(feat_names) == X.shape[2], f"Feature names({len(feat_names)}) != X cols({X.shape[2]})"
    ths_count = sum(1 for f in feat_names if f.startswith("tech_ths_"))
    print(f"[OK] Features: {len(feat_names)} total (base {len(feat_names)-ths_count} + THS {ths_count})")

    print("\n[PASS] TEST 2\n")

    # Print sample feature names
    print("First 10 features:", feat_names[:10])
    print("THS features sample:", [f for f in feat_names if f.startswith("tech_ths_")][:5])


def test_edge_cases():
    """测试边界情况."""
    from app.core.ths_indicators import add_all_ths_indicators

    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)

    # Small data (just enough)
    df_small = generate_synthetic_ohlcv(50)
    df_small = add_all_ths_indicators(df_small)
    assert len(df_small) == 50
    print("[OK] 50 rows computed successfully")

    # Zero volume
    df_zero_vol = generate_synthetic_ohlcv(100)
    df_zero_vol.loc[10:15, "volume"] = 0
    df_zero_vol = add_all_ths_indicators(df_zero_vol)
    assert not df_zero_vol["tech_ths_ctrl_low"].isna().any()
    print("[OK] Zero volume does not cause NaN")

    # Missing columns
    try:
        add_all_ths_indicators(pd.DataFrame({"close": [1, 2, 3]}))
        assert False, "Should raise KeyError"
    except KeyError:
        print("[OK] Missing columns correctly raises KeyError")

    print("\n[PASS] TEST 3\n")


if __name__ == "__main__":
    test_ths_indicators()
    test_factor_engine()
    test_edge_cases()
    print("=" * 60)
    print("[DONE] All tests passed!")
    print("=" * 60)
