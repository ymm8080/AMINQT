# -*- coding: utf-8 -*-
"""dim20_short_horizon 特征口径单测 — close_vs_low_ma5 (2026-09-09 注入).

口径锚定 09-09 日内指纹实验: mean5(close/low-1)*100, IC -0.0756 t=-13.7,
对 r5/r10 残差 IC -0.064 (独立动量). 改动本列口径须同步更新本测试.
"""
import numpy as np
import pandas as pd

from app.pipeline1.feature_engine_v35 import FeatureEngineV35


def _toy_panel(n_days: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2026-08-01", periods=n_days)
    rng = np.random.RandomState(42)
    rows = []
    for sym in ("000001.SZ", "300001.SZ"):
        base = 10.0
        for d in dates:
            ret = rng.normal(0, 0.02)
            close = base * (1 + ret)
            low = close * (1 - abs(rng.normal(0, 0.01)))
            high = close * (1 + abs(rng.normal(0, 0.01)))
            open_ = low + (high - low) * rng.uniform()
            rows.append({
                "symbol": sym, "date": d, "open": open_, "high": high,
                "low": low, "close": close, "volume": float(rng.randint(1, 5) * 1e6),
            })
            base = close
    return pd.DataFrame(rows)


def test_close_vs_low_ma5_equals_rolling5_mean():
    df = _toy_panel()
    out = FeatureEngineV35().dim20_short_horizon(df.copy())

    exp = (
        (df["close"] / df["low"] - 1.0)
        .groupby(df["symbol"])
        .transform(lambda s: s.rolling(5, min_periods=5).mean())
        * 100
    )
    got = out["close_vs_low_ma5"]
    m = exp.notna() & got.notna()
    assert m.sum() >= 10
    assert np.allclose(got[m], exp[m], atol=1e-9)


def test_close_vs_low_ma20_equals_rolling20_mean():
    df = _toy_panel(n_days=25)
    out = FeatureEngineV35().dim20_short_horizon(df.copy())

    exp = (
        (df["close"] / df["low"] - 1.0)
        .groupby(df["symbol"])
        .transform(lambda s: s.rolling(20, min_periods=20).mean())
        * 100
    )
    got = out["close_vs_low_ma20"]
    m = exp.notna() & got.notna()
    assert m.sum() >= 5
    assert np.allclose(got[m], exp[m], atol=1e-9)


def test_close_vs_low_ma5_warmup_nan_then_filled():
    df = _toy_panel()
    out = FeatureEngineV35().dim20_short_horizon(df.copy())
    for sym, g in out.groupby("symbol"):
        assert g["close_vs_low_ma5"].iloc[:4].isna().all()  # 4 日预热
        assert g["close_vs_low_ma5"].iloc[4:].notna().all()
        assert np.isinf(g["close_vs_low_ma5"]).sum() == 0
