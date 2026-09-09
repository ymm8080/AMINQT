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


def _board_panel(n_days: int = 25) -> pd.DataFrame:
    """确定性断板面板: day1 涨停(+10%), day2 断板, day2 起线性漂移.

    closes: [10.0, 11.0, 10.5+0.02*i ...]; vols: [5e6, 8e6, 4e6, 3e6, 2e6, 2e6...]
    (无 turnover_rate 列 → vol_decay 走 volume 回退路径)
    """
    dates = pd.bdate_range("2026-08-01", periods=n_days)
    closes = [10.0, 11.0] + [10.5 + 0.02 * i for i in range(n_days - 2)]
    vols = [5e6, 8e6, 4e6, 3e6] + [2e6] * (n_days - 4)
    rows = []
    for sym in ("000001.SZ",):
        for i, d in enumerate(dates):
            c = closes[i]
            rows.append({
                "symbol": sym, "date": d, "open": c * 0.995, "high": c * 1.01,
                "low": c * 0.99, "close": c, "volume": float(vols[i]),
            })
    return pd.DataFrame(rows)


def test_days_since_board_break_epoch_and_cap():
    df = _board_panel()
    out = FeatureEngineV35().dim20_short_horizon(df.copy())
    got = out["days_since_board_break"]
    # day0(无断板史)/day1(涨停日) = NaN; day2=断板日(0), day3=1, ...
    assert got.iloc[:2].isna().all()
    assert got.iloc[2] == 0
    assert got.iloc[3] == 1
    # 距断板 >20 日归 NaN (day2+21=day23 起)
    assert got.iloc[2:23].notna().all()
    assert got.iloc[23:].isna().all()
    assert got.iloc[22] == 20  # 封顶值


def test_vol_decay_ratio_epoch_mean_denominator():
    df = _board_panel()
    out = FeatureEngineV35().dim20_short_horizon(df.copy())
    got = out["vol_decay_ratio"]
    # pos>=2 才有值: day2(pos0)/day3(pos1) = NaN
    assert got.iloc[:4].isna().all()
    # day4: to=2e6, denom=mean(v[day2],v[day3])=mean(4e6,3e6)=3.5e6
    assert np.isclose(got.iloc[4], 2e6 / 3.5e6, atol=1e-12)
    # day5: denom=mean(4,3,2)e6=3e6
    assert np.isclose(got.iloc[5], 2e6 / 3e6, atol=1e-12)
    # 口径=不限期 expanding (与实验主脚本一致, IC -0.059 t=-11.5 残差 -0.030 t=-6.9):
    # 断板 epoch 内一直有值, 不随 days_since 的 20 日封顶失效
    assert got.iloc[6:23].notna().all()


def test_quiet_drift_linear_slope():
    df = _board_panel(n_days=25)
    out = FeatureEngineV35().dim20_short_horizon(df.copy())
    got = out["quiet_drift"]
    # 前 19 日预热 NaN
    assert got.iloc[:19].isna().all()
    # 窗口全落在线性段(day2 起 slope=0.02/日)的最早位置是 day21
    for k in (21, 22, 23, 24):
        exp = 0.02 / df["close"].iloc[k] * 100
        assert np.isclose(got.iloc[k], exp, atol=1e-9), k


# ---------------- fade_score (2026-09-09 注入) ----------------
def _fade_panel(n_days: int = 80) -> pd.DataFrame:
    """两票确定性面板 (含 turnover_rate), 供 fade_score 口径对拍."""
    dates = pd.bdate_range("2026-01-01", periods=n_days)
    rng = np.random.RandomState(7)
    rows = []
    for sym in ("000001.SZ", "300001.SZ"):
        base = 10.0
        for i, d in enumerate(dates):
            ret = rng.normal(0, 0.02)
            close = base * (1 + ret)
            high = close * (1 + abs(rng.normal(0, 0.01)))
            low = close * (1 - abs(rng.normal(0, 0.01)))
            rows.append({
                "symbol": sym, "date": d, "open": low, "high": high,
                "low": low, "close": close, "volume": 1e6,
                "turnover_rate": 1.0 + 0.01 * i,
            })
            base = close
    return pd.DataFrame(rows)


def test_fade_score_formula_and_warmup():
    df = _fade_panel()
    out = FeatureEngineV35().dim20_short_horizon(df.copy())
    got = out["fade_score"]

    # 独立复刻: r20/vol20/turn20/pos250 四列齐备才计, 日截面 pct-rank 等权
    g = df.sort_values(["symbol", "date"]).groupby("symbol")
    c = df.sort_values(["symbol", "date"])["close"]
    ret1 = c / g["close"].shift(1) - 1.0
    raw = pd.DataFrame({
        "r20": g["close"].pct_change(20),
        "vol20": ret1.groupby(df.sort_values(["symbol", "date"])["symbol"]).transform(
            lambda s: s.rolling(20).std()),
        "turn20": g["turnover_rate"].transform(lambda s: s.rolling(20).mean()),
        "pos250": c / g["high"].transform(
            lambda s: s.rolling(250, min_periods=60).max()) - 1.0,
    })
    m = raw.notna().all(axis=1)
    exp = raw[m].groupby(df.loc[m, "date"]).rank(pct=True).mean(axis=1)

    assert got.notna().sum() == len(exp)
    join = pd.DataFrame({"got": got, "exp": exp}).dropna()
    assert len(join) == got.notna().sum()  # 全部对齐; pos250 min_periods=60 → 80日仅末21日有值
    assert np.allclose(join["got"], join["exp"], atol=1e-9)
    # 预热期 NaN (pos250 min_periods=60 → 前59日 NaN; r20 需21日, 取严者)
    per = out.groupby("symbol")["fade_score"]
    assert per.apply(lambda s: s.iloc[:59].isna().all()).all()
    # 值域 [0,1]
    v = got.dropna()
    assert ((v >= 0) & (v <= 1)).all()
