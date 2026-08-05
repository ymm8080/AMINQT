# -*- coding: utf-8 -*-
"""Tests for app/pipeline_parallel 慢牛系统 (SLOW_BULL, ADX 设计文档 v1.0, 2026-08-05).

覆盖: ADX 指标 (强趋势 vs 震荡) / 四道硬门槛 / 权重打分缺列重归一 /
gate 预筛后 TOP-N 选股 / 买卖信号 / 移动止盈 / run_system 带 gate+权重 /
每日观察池输出 / 独立导出. 合成数据 (固定 seed), 不触检查点.
"""
import numpy as np
import pandas as pd
import pytest

from app.pipeline_parallel import indicators, screener, signals
from app.pipeline_parallel.config import (ADX_SPEC, ADX_SCORE_WEIGHTS,
                                          SLOW_BULL)
from app.pipeline_parallel.scoring import (cross_rank, pool_score,
                                           select_topn)

_T = "0000000"      # 慢牛趋势样本符号 (漂移加速)
_O = "3000002"      # 震荡样本符号 (有界正弦)


# ── 合成面板: 前 n_trend 只 = 慢牛趋势 (漂移加速, 低噪), 后 n_osc 只 = 震荡 (正弦) ──
def _slow_panel(n_trend=2, n_osc=2, n_dates=130, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-11-01", periods=n_dates)
    frames = []
    t = np.arange(n_dates)
    for i in range(n_trend + n_osc):
        sym = f"000{i:04d}" if i < n_trend else f"300{i:04d}"
        board = "main" if i < n_trend else "dual"
        if i < n_trend:
            # 慢牛: 漂移严格加速 (0.2%→~0.85%/日) → 趋势持续加强, ADX 走高且上升
            rets = 0.002 + 0.00005 * t + rng.normal(0, 0.002, n_dates)
            close = 10.0 * np.exp(np.cumsum(rets))
        else:
            # 震荡: 有界正弦 (周期 10 日) + 微噪 → 趋势翻转, ADX 低
            close = 10.0 + 0.5 * np.sin(2 * np.pi * t / 10.0) \
                + rng.normal(0, 0.02, n_dates)
        op = np.concatenate([[close[0]], close[:-1]])       # 近似平开
        hi = close * (1 + 0.004)
        lo = close * (1 - 0.004)
        vol = 100.0 * (1 + 0.008 * t) * (1 + rng.normal(0, 0.005, n_dates))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates, "board": board,
            "open": op, "high": hi, "low": lo, "close": close,
            "open_hfq": op, "high_hfq": hi, "low_hfq": lo, "close_hfq": close,
            "volume": vol, "turnover_rate": 5.0, "volume_ratio": 1.0,
            "margin_balance_chg_5d": 0.01, "pct_70_con": 0.5, "adv20": 1e8,
        }))
    return pd.concat(frames, ignore_index=True)


def _slow_work(**kw):
    """加指标列 + 信号列 + 门槛掩码 (load_panel 慢牛路径的合成版)."""
    df = _slow_panel(**kw)
    df = indicators.prepare_adx(df)
    signals.add_signal_columns(df)
    df["gate_slow_bull"] = screener.compute_gate(df, "slow_bull")
    return df


# ── ADX 指标 ──
def test_adx_trend_strong_rising_pdi_above_mdi():
    work = _slow_work()
    trend = work[work["symbol"] == _T]
    late = trend.iloc[-10:]                      # 成熟趋势段 (ma60 已生效)
    assert (late["adx"] > ADX_SPEC["adx_min"]).all()   # ADX > 25
    assert (late["pdi"] > late["mdi"]).all()            # +DI > -DI
    assert (late["adx_rise5"] > 0).all()                # 趋势仍在加强


def test_adx_oscillator_low():
    work = _slow_work()
    osc = work[work["symbol"] == _O]
    late = osc.iloc[-30:]
    assert (late["adx"] < ADX_SPEC["adx_min"]).all()    # 震荡 → ADX < 25


# ── 四道硬门槛 ──
def test_gate_passes_trend_fails_oscillator():
    work = _slow_work()
    dates = np.sort(work["date"].unique())
    late = work["date"] >= dates[-30]
    # 趋势股成熟段全部通过
    assert work.loc[(work["symbol"] == _T) & late, "gate_slow_bull"].all()
    # 震荡股从未通过
    assert not work.loc[work["symbol"] == _O, "gate_slow_bull"].any()


def test_gate_requires_ma_alignment():
    # 门槛一: 破坏 ma5>ma10>ma20>ma60 → 不通过 (手工构造)
    df = pd.DataFrame({
        "symbol": [_T] * 5,
        "date": pd.bdate_range("2025-01-06", periods=5),
        "close_cont": [10.2, 10.1, 10.0, 9.9, 9.8],   # 下行
        "ma5": [10.25, 10.15, 10.05, 9.95, 9.85],
        "ma10": [10.2, 10.2, 10.2, 10.2, 10.2],
        "ma20": [10.0, 10.0, 10.0, 10.0, 10.0],
        "ma60": [9.5, 9.5, 9.5, 9.5, 9.5],
        "ma_slope5": [0.0] * 5, "ma_slope10": [0.0] * 5,
        "ma_slope20": [0.0] * 5,
        "adx": [30.0] * 5, "pdi": [30.0] * 5, "mdi": [10.0] * 5,
        "adx_rise5": [2.0] * 5,
        "amplitude_20": [0.02] * 5, "max_drop_20": [-0.02] * 5,
        "limit_down_20": [0] * 5,
        "ma_vol_5": [120.0] * 5, "ma_vol_10": [110.0] * 5,
        "ma_vol_20": [100.0] * 5, "vol_ratio": [1.0] * 5,
        "turnover_rate": [5.0] * 5,
    })
    assert not screener.slow_bull_gate(df).any()   # 均线未多头 + 斜率 0 → 全不通过


def test_gate_vol_ratio_uses_yesterday():
    # 门槛四 PIT: 量比看昨日 (groupby symbol shift), 今日爆量不算爆
    df = pd.DataFrame({
        "symbol": [_T] * 5,
        "date": pd.bdate_range("2025-01-06", periods=5),
        "close_cont": [10.0, 10.05, 10.1, 10.15, 10.2],
        "ma5": [9.9, 9.95, 10.0, 10.05, 10.1],
        "ma10": [9.8, 9.85, 9.9, 9.95, 10.0],
        "ma20": [9.5, 9.55, 9.6, 9.65, 9.7],
        "ma60": [8.5, 8.55, 8.6, 8.65, 8.7],
        "ma_slope5": [0.01] * 5, "ma_slope10": [0.01] * 5,
        "ma_slope20": [0.01] * 5,
        "adx": [30.0] * 5, "pdi": [30.0] * 5, "mdi": [10.0] * 5,
        "adx_rise5": [2.0] * 5,
        "amplitude_20": [0.02] * 5, "max_drop_20": [-0.02] * 5,
        "limit_down_20": [0] * 5,
        "ma_vol_5": [120.0] * 5, "ma_vol_10": [110.0] * 5,
        "ma_vol_20": [100.0] * 5,
        "vol_ratio": [1.0, 1.0, 1.0, 1.0, 5.0],     # 末日爆量 (只看昨日 → 末日仍过)
        "turnover_rate": [5.0] * 5,
    })
    g = screener.slow_bull_gate(df)
    # 首日昨日量比 NaN → 不通过 (无昨日参考); 第 2-5 日昨日 1.0<3 → 过 (末日爆量不看)
    assert not g.iloc[0]
    assert g.iloc[1:].all()


def test_gate_turnover_bounds():
    # 门槛四: 换手率 3%-15%, 出界不通过. 首行作"昨日量比参考", 第 1 行起判.
    df = pd.DataFrame({
        "symbol": [_T] * 4,
        "date": pd.bdate_range("2025-01-06", periods=4),
        "close_cont": [10.0, 10.1, 10.2, 10.3],
        "ma5": [9.9, 10.0, 10.1, 10.2], "ma10": [9.8, 9.9, 10.0, 10.1],
        "ma20": [9.5, 9.6, 9.7, 9.8], "ma60": [8.5, 8.6, 8.7, 8.8],
        "ma_slope5": [0.01] * 4, "ma_slope10": [0.01] * 4,
        "ma_slope20": [0.01] * 4,
        "adx": [30.0] * 4, "pdi": [30.0] * 4, "mdi": [10.0] * 4,
        "adx_rise5": [2.0] * 4,
        "amplitude_20": [0.02] * 4, "max_drop_20": [-0.02] * 4,
        "limit_down_20": [0] * 4,
        "ma_vol_5": [120.0] * 4, "ma_vol_10": [110.0] * 4,
        "ma_vol_20": [100.0] * 4, "vol_ratio": [1.0] * 4,
        "turnover_rate": [5.0, 5.0, 2.0, 20.0],  # 行1 5%界内过, 行2 2%低界, 行3 20%高界
    })
    g = screener.slow_bull_gate(df)
    assert g.iloc[1] and not g.iloc[2] and not g.iloc[3]


def test_gate_missing_indicator_columns_raises():
    df = pd.DataFrame({"symbol": [_T], "date": [pd.Timestamp("2025-01-06")]})
    with pytest.raises(KeyError):
        screener.compute_gate(df, "slow_bull")


def test_apply_gate_missing_columns_returns_empty():
    df = pd.DataFrame({"symbol": [_T], "date": [pd.Timestamp("2025-01-06")]})
    out = screener.apply_gate(df, "slow_bull")
    assert len(out) == 0


def test_compute_gate_unregistered_raises():
    df = _slow_panel(n_dates=5, n_trend=1, n_osc=0)
    with pytest.raises(KeyError):
        screener.compute_gate(df, "no_such_gate")


# ── 权重打分: 缺列自动跳过 + 再归一化 ──
def test_pool_score_weighted_renormalizes_on_missing_col():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2025-01-06", periods=5)
    rows = []
    for d in dates:
        for i in range(10):
            rows.append({"date": d, "symbol": f"{i:04d}",
                         "a": rng.normal(), "b": rng.normal()})
    df = pd.DataFrame(rows)
    w = {"a": 0.6, "b": 0.4}
    s_full = pool_score(df, ("a", "b"), weights=w)
    # 缺 b → a 独权, 等权于 cross_rank(a)
    s_skip = pool_score(df, ("a", "c_missing"), weights=w)
    assert np.allclose(s_skip, cross_rank(df, "a"))
    # 权重不同 → 两结果不一致 (再归一化生效, 非简单跳过)
    assert not np.allclose(s_full, s_skip)
    assert np.isclose(sum(w.values()), 1.0)


def test_slowbull_pool_weights_all_present():
    assert set(ADX_SCORE_WEIGHTS) == set(SLOW_BULL.pool)
    # 北向资金 10% 停更 (2024-08) → 文档 §2.3 的 8 项权重去掉北向 = 0.90;
    # pool_score 在打分时对可用列再归一化 (sum→1.0), 静态 dict 保持文档原比例.
    assert np.isclose(sum(ADX_SCORE_WEIGHTS.values()), 0.90)
    assert SLOW_BULL.pool_weights == ADX_SCORE_WEIGHTS


# ── 门槛预筛后按日 Top-N ──
def test_select_topn_on_gated_pool():
    work = _slow_work()
    sub = work[work["gate_slow_bull"]]
    assert len(sub) > 0
    score = pool_score(sub, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
    top = select_topn(sub, score, SLOW_BULL.top_n)
    assert len(top) > 0
    # 每日期至多 Top-20, 且选中的都是通过门槛的
    assert top.groupby("date").size().le(SLOW_BULL.top_n).all()
    assert set(top["symbol"]) <= set(sub["symbol"])


# ── 买入信号 (手算) ──
def test_buy_signals_pullback_ma5():
    df = pd.DataFrame({
        "symbol": [_T] * 3,
        "date": pd.bdate_range("2025-01-06", periods=3),
        "close_cont": [10.08, 10.12, 10.15],
        "low_hfq": [9.98, 10.16, 9.95],
        "ma5": [10.00, 10.05, 10.10],
        "ma_slope5": [0.01] * 3,
        "ma10": [9.90, 9.95, 10.00],
        "ma_slope10": [0.01] * 3,
        "open_hfq": [10.00, 10.05, 10.10],
        "vol_ratio": [1.0] * 3,
    })
    signals.buy_signals(df)
    # 最低价触到 ma5*(1+1%) 且收于 ma5 上 → 回踩; 中间行低点太高 → 不回踩
    assert df["pullback_ma5"].tolist() == [True, False, True]
    assert not df["pullback_ma10"].any()        # close 未跌破 ma5 → 无 ma10 回踩


def test_buy_signals_pullback_ma10_and_shrink_vol():
    df = pd.DataFrame({
        "symbol": [_T] * 3,
        "date": pd.bdate_range("2025-01-06", periods=3),
        "close_cont": [10.00, 10.02, 10.04],
        "low_hfq": [9.99, 9.98, 9.97],
        "ma5": [10.05] * 3,
        "ma_slope5": [0.01] * 3,
        "ma10": [9.96] * 3,
        "ma_slope10": [0.01] * 3,
        "vol_ratio": [0.6, 0.7, 1.5],
        "open_hfq": [10.02, 10.03, 10.00],
    })
    signals.buy_signals(df)
    # 跌破 ma5 未破 ma10 + ma10 斜率向上 → 回踩 ma10
    assert df["pullback_ma10"].tolist() == [True, True, True]
    # 缩量 + 收小阴线 (close<open 且实体小) → 第 1/2 行缩量小阴, 第 3 行放量 → 否
    assert df["shrink_vol"].tolist() == [True, True, False]


def test_no_buy_flags_chase_high_and_vol_spike():
    df = pd.DataFrame({
        "symbol": [_T] * 3,
        "date": pd.bdate_range("2025-01-06", periods=3),
        "close_cont": [10.0, 11.0, 12.0],
        "ma5": [10.0, 10.0, 10.0],
        "adx_rise5": [1.0, 1.0, -1.0],
        "vol_ratio": [1.0, 2.0, 2.0],
    })
    signals.no_buy_flags(df)
    assert df["chase_high"].tolist() == [False, True, True]     # 偏离 >8%
    assert df["vol_spike_up"].tolist() == [False, True, True]   # 放量上涨 >5%
    assert df["adx_falling"].tolist() == [False, False, True]   # ADX 转降


# ── 卖出信号 (手算) ──
def test_sell_signals_below_ma20_adx_broken_big_drop():
    df = pd.DataFrame({
        "symbol": [_T] * 3,
        "date": pd.bdate_range("2025-01-06", periods=3),
        "close_cont": [10.5, 10.3, 9.5],        # 末日跌破 ma20
        "ma5": [10.4, 10.35, 10.3],
        "ma20": [10.4, 10.2, 10.0],
        "adx": [30.0, 30.0, 18.0],              # 末 ADX<20 → 破位
        "pdi": [30.0, 30.0, 25.0],
        "mdi": [10.0, 10.0, 20.0],
        "vol_ratio": [1.0, 2.0, 2.0],
        "turnover_rate": [5.0, 5.0, 5.0],
        "adx_rise5": [2.0, 2.0, 2.0],
    })
    signals.sell_signals(df)
    assert df["below_ma20"].tolist() == [False, False, True]
    assert df["adx_broken"].tolist() == [False, False, True]
    # 末日单日 -7.8% (10.3→9.5) + 量比 2 > 1.5 → 放量大跌
    assert df["big_drop"].tolist() == [False, False, True]


def test_sell_signals_below_ma5_3d():
    n = 6
    df = pd.DataFrame({
        "symbol": [_T] * n,
        "date": pd.bdate_range("2025-01-06", periods=n),
        "close_cont": [10.5, 10.4, 10.3, 10.2, 10.1, 10.0],
        "ma5": [10.45] * n,                     # 后 4 日收于 ma5 下
        "ma20": [10.0] * n,
        "adx": [30.0] * n, "pdi": [30.0] * n, "mdi": [10.0] * n,
        "vol_ratio": [1.0] * n,
        "turnover_rate": [5.0] * n,
        "adx_rise5": [2.0] * n,
    })
    signals.sell_signals(df)
    # 第 4 行起连续 3 日收于 ma5 下 → 触发
    assert df["below_ma5_3d"].tolist() == [False, False, False, True, True, True]


def test_sell_signals_turnover_spike():
    n = 22
    turn = [5.0] * (n - 1) + [10.0]           # 末一日换手突增
    df = pd.DataFrame({
        "symbol": [_T] * n,
        "date": pd.bdate_range("2025-01-06", periods=n),
        "close_cont": np.linspace(10.0, 10.6, n),
        "ma5": np.linspace(10.0, 10.55, n),
        "ma20": np.linspace(10.0, 10.5, n),
        "adx": [30.0] * n, "pdi": [30.0] * n, "mdi": [10.0] * n,
        "vol_ratio": [1.0] * n,
        "turnover_rate": turn,
        "adx_rise5": [2.0] * n,
    })
    signals.sell_signals(df)
    assert df["turnover_spike"].tolist() == [False] * (n - 1) + [True]


def test_sell_signals_tp80_with_cost():
    df = pd.DataFrame({
        "symbol": [_T] * 2,
        "date": pd.bdate_range("2025-01-06", periods=2),
        "close_cont": [10.0, 19.0],            # 成本 5 → 累计 +280% (>80%)
        "ma5": [9.5, 9.5], "ma20": [9.0, 9.0],
        "adx": [30.0, 30.0], "pdi": [30.0, 30.0], "mdi": [10.0, 10.0],
        "vol_ratio": [1.0, 1.0],
        "turnover_rate": [5.0, 5.0],
        "adx_rise5": [1.0, -1.0],
    })
    signals.sell_signals(df, cost=pd.Series([5.0, 5.0], index=df.index))
    assert df["tp_80_div"].tolist() == [False, True]


# ── 移动止盈 (纯函数) ──
def test_trailing_stop_pct_tiers():
    assert signals.trailing_stop_pct(0.30) == 0.15     # >20% → 锁 15%
    assert signals.trailing_stop_pct(0.60) == 0.40     # >50% → 锁 40%
    assert signals.trailing_stop_pct(1.20) is None     # >100% → ma20 规则
    assert signals.trailing_stop_pct(0.10) is None     # <20% → 无移动止盈


def test_trailing_stop_price():
    assert np.isclose(signals.trailing_stop_price(10.0, 0.30, ma20=11.0), 11.5)
    assert np.isclose(signals.trailing_stop_price(10.0, 1.20, ma20=14.0), 14.0)


# ── run_system 带 gate + 权重 (慢牛长视界) ──
def test_run_system_slowbull_full_flow():
    from app.pipeline_parallel.backtest import add_mfe_labels, run_system

    work = _slow_work()
    work = add_mfe_labels(work, horizons=(10, 20, 40))
    res = run_system(work, SLOW_BULL, top_n=SLOW_BULL.top_n)
    assert set(res["per_horizon"]) == {"10d", "20d", "40d"}
    assert res["per_horizon"]["10d"]["n"] > 0
    # 成熟趋势段每日期 2 只趋势股入池 → Top-20 每日期至多 2 行
    assert 0 < res["n_picks"] <= work["date"].nunique() * 2
    assert isinstance(res["passed"], list)


def test_run_system_slowbull_gate_empty_on_unprepared_panel():
    from app.pipeline_parallel.backtest import add_mfe_labels, run_system

    df = _slow_panel(n_trend=2, n_osc=0, n_dates=60)
    df = add_mfe_labels(df, horizons=(10,))
    # 未 prepare 指标列 → 慢牛无候选, 不崩
    res = run_system(df, SLOW_BULL, top_n=SLOW_BULL.top_n)
    assert res["n_picks"] == 0


# ── 每日观察池输出 ──
def test_daily_slowbull_pool_columns_and_signals():
    work = _slow_work()
    date = work["date"].max()
    pool = signals.daily_slowbull_pool(work, date, "main",
                                       SLOW_BULL, SLOW_BULL.top_n)
    assert len(pool) > 0
    required = {"board", "date", "symbol", "rk", "score", "dev5",
                "close_cont", "ma5", "ma10", "ma20", "ma60",
                "adx", "pdi", "mdi",
                "pullback_ma5", "pullback_ma10", "shrink_vol",
                "chase_high", "vol_spike_up", "adx_falling",
                "below_ma20", "adx_broken", "big_drop",
                "below_ma5_3d", "turnover_spike", "tp_80_div"}
    assert required.issubset(set(pool.columns))
    assert (pool["board"] == "main").all()
    assert pool["rk"].is_monotonic_increasing
    # 趋势股在末段应仍保持慢牛特征 (收于 ma20 上方等)
    assert (pool["close_cont"] > pool["ma20"]).all()


def test_daily_slowbull_pool_requires_gate_column():
    df = _slow_panel()
    with pytest.raises(KeyError):
        signals.daily_slowbull_pool(df, df["date"].max(), "main",
                                    SLOW_BULL, SLOW_BULL.top_n)


# ── 独立导出 (70% 资金仓, 不并入 merged) ──
def test_export_slowbull_lists(monkeypatch, tmp_path):
    import app.pipeline_parallel.backtest as bt
    from app.pipeline_parallel.backtest import add_mfe_labels, export_stock_lists
    from app.pipeline_parallel.config import (ALL_HORIZON_INTS, FUSION,
                                              SNIPER)

    work = _slow_work(n_trend=4, n_osc=0)
    # 4 只趋势股: 前 2 归 main, 后 2 归 dual (导出需两板块都有慢牛候选)
    work["board"] = work["symbol"].map(
        {"0000000": "main", "0000001": "main",
         "0000002": "dual", "0000003": "dual"})
    rng = np.random.default_rng(3)
    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    for c in pool:                                   # 补齐狙击/融合池列 (导出会跑其循环)
        work[c] = rng.normal(0, 1, len(work))
    work = add_mfe_labels(work, horizons=ALL_HORIZON_INTS)
    monkeypatch.setattr(bt, "BACKTEST_RESULT_DIR", tmp_path)
    dates = np.sort(work["date"].unique())
    files = export_stock_lists(work, str(pd.Timestamp(dates[-20])), tmp_path)
    for b in ("main", "dual"):
        assert f"stocks_{b}_slow_bull_full.csv" in files, f"缺 {b} full"
        assert f"stocks_{b}_slow_bull_oos.csv" in files, f"缺 {b} oos"
        csv = pd.read_csv(tmp_path / f"stocks_{b}_slow_bull_oos.csv")
        assert {"date", "symbol", "score", "rk",
                "label_mfe_10d_net", "label_mfe_20d_net",
                "label_mfe_40d_net"}.issubset(set(csv.columns))
        assert len(csv) > 0
    # 慢牛独立仓: 不并入狙击/融合 merged 名单
    merged = pd.read_csv(tmp_path / "stocks_merged_oos_main.csv")
    assert "slow_bull" not in set(merged["systems"].str.split("+").explode())


def test_system_spec_gate_and_weights_configured():
    assert SLOW_BULL.gate == "slow_bull"
    assert SLOW_BULL.horizons == ("10d", "20d", "40d")
    assert SLOW_BULL.pool_weights == ADX_SCORE_WEIGHTS
    assert "slow_bull" in screener.GATES
