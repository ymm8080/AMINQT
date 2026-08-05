# -*- coding: utf-8 -*-
"""ADX 慢牛系统 — 买入/卖出/移动止盈信号 (2026-08-05, ADX 设计文档 v1.0 §3-4).

买入 (§3.1, 均线附近低吸): pullback_ma5 / pullback_ma10 / shrink_vol;
不买 (§3.2): chase_high / vol_spike_up / adx_falling;
卖出 (§4.2, 满足任一即卖): below_ma20 / adx_broken / big_drop /
  below_ma5_3d / turnover_spike / tp_80_div;
移动止盈 (§4.3, 纯函数): 浮盈>20%→锁+15%, >50%→锁+40%, >100%→ma20.

信号列须在**全面板**上一次性计算 (add_signal_columns): below_ma5_3d 连续计数与
turnover_spike 近20日最高都依赖个股历史, 单日切片算不出. 全部 PIT (只用 t 及更早,
滚动窗口不含未来). 连续价统一后复权 (close_cont/low_hfq/open_hfq), 与连续 MA 自洽.
"""

from __future__ import annotations

import pandas as pd

from app.pipeline_parallel.config import ADX_SPEC
from app.pipeline_parallel.scoring import pool_score, select_topn


def _consecutive_true(df: pd.DataFrame, cond: pd.Series) -> pd.Series:
    """按 symbol 连续满足 cond 的天数 (当日及之前; 到 0 为止)."""
    flag = cond.astype(int)
    grp = flag.groupby(df["symbol"])
    cum = grp.cumsum()
    last_zero = cum.where(flag == 0).groupby(df["symbol"]).ffill().fillna(0)
    return cum - last_zero


def _per_symbol_ret(df: pd.DataFrame) -> pd.Series:
    """单日收益 (按 symbol 内 pct_change, 防符号边界串行)."""
    return df["close_cont"].groupby(df["symbol"]).pct_change()


def buy_signals(df: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """买入信号 (文档 §3.1, 三选一即可买). 就地加三列布尔."""
    if spec is None:
        spec = ADX_SPEC
    low = df["low_hfq"] if "low_hfq" in df.columns else df["low"]
    open_ = df["open_hfq"] if "open_hfq" in df.columns else df["open"]
    tol = spec["pullback_tol"]
    df["pullback_ma5"] = (
        (low <= df["ma5"] * (1 + tol))  # 最低价回踩到 ma5 附近
        & (df["close_cont"] >= df["ma5"])  # 且收于 ma5 上方 (从上方回踩)
        & (df["ma_slope5"] > 0)  # 且 ma5 斜率仍向上
    )
    df["pullback_ma10"] = (
        (df["close_cont"] < df["ma5"])  # 跌破 ma5
        & (low > df["ma10"])  # 未破 ma10
        & (df["ma_slope10"] > 0)  # 且 ma10 斜率向上
    )
    candle = (df["close_cont"] - open_) / open_
    df["shrink_vol"] = (
        (df["vol_ratio"] < spec["shrink_vol_ratio_max"])  # 缩量 (量比<0.8)
        & (df["close_cont"] < open_)  # 收小阴线
        & (candle > -spec["small_candle_max"])
    )
    return df


def no_buy_flags(df: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """不买情形 (文档 §3.2, 任一为真则不追). 就地加三列布尔."""
    if spec is None:
        spec = ADX_SPEC
    df["chase_high"] = (df["close_cont"] / df["ma5"] - 1) > spec["dev5_max"]
    df["vol_spike_up"] = (_per_symbol_ret(df) > spec["vol_spike_up_max"]) & (
        df["vol_ratio"] > spec["vol_spike_ratio_min"]
    )
    df["adx_falling"] = df["adx_rise5"] < 0
    return df


def sell_signals(
    df: pd.DataFrame, spec: dict | None = None, cost: pd.Series | None = None
) -> pd.DataFrame:
    """卖出信号 (文档 §4.2, 满足任一即卖). 就地加六列布尔.

    tp_80_div 默认 (无持仓成本) 用"近 60 日新高 + ADX 顶背离"代理;
    传入 cost (与 df 对齐的成本序列) 则用真实"累计涨幅>80% + ADX 衰竭".
    """
    if spec is None:
        spec = ADX_SPEC
    ret = _per_symbol_ret(df)
    df["below_ma20"] = df["close_cont"] < df["ma20"]
    df["adx_broken"] = (df["adx"] < spec["adx_broken_min"]) | (df["pdi"] < df["mdi"])
    df["big_drop"] = (
        (ret < -spec["big_drop_sell"])
        & (df["vol_ratio"] > spec["vol_spike_ratio_min"])  # 放量大跌
    )
    df["below_ma5_3d"] = (
        _consecutive_true(df, df["close_cont"] < df["ma5"]) >= spec["below_ma5_days"]
    )
    turn = df["turnover_rate"]
    prev = turn.groupby(df["symbol"]).shift(1)
    prev_max = (
        prev.groupby(df["symbol"])
        .rolling(spec["turnover_spike_win"], min_periods=spec["turnover_spike_win"])
        .max()
        .reset_index(level=0, drop=True)
    )
    df["turnover_spike"] = turn > prev_max
    if "adx_falling" not in df.columns:  # 独立调用 (未跑 no_buy_flags) → 就地算
        df["adx_falling"] = df["adx_rise5"] < 0
    if cost is not None:
        df["tp_80_div"] = ((df["close_cont"] / cost - 1) > spec["tp_gain"]) & df[
            "adx_falling"
        ]
    else:
        win = spec["tp_high_window"]
        near_high = df["close_cont"] >= df["close_cont"].groupby(df["symbol"]).rolling(
            win, min_periods=win
        ).max().reset_index(level=0, drop=True) * (1 - spec["tp_high_near"])
        df["tp_80_div"] = near_high & df["adx_falling"]
    return df


def add_signal_columns(df: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """全面板一次性加全部买卖信号布尔列 (就地, 返回原 df)."""
    buy_signals(df, spec)
    no_buy_flags(df, spec)
    sell_signals(df, spec)
    return df


def trailing_stop_pct(profit_pct: float, spec: dict | None = None) -> float | None:
    """移动止盈: 浮盈对应锁盈比 (成本之上比例). 盈利>=100% → 用 ma20 规则 (返回 None)."""
    if spec is None:
        spec = ADX_SPEC
    if profit_pct >= 1.0:
        return None  # >100%: 止盈线 = ma20 (趋势不破不走)
    if profit_pct >= 0.5:
        return 0.40  # >50%: 锁 40%
    if profit_pct >= 0.2:
        return 0.15  # >20%: 锁 15%
    return None  # <20%: 无移动止盈 (用硬卖出)


def trailing_stop_price(
    cost: float, profit_pct: float, ma20: float, spec: dict | None = None
) -> float:
    """止盈线价格: 有锁盈比 → 成本×(1+比); 否则 (盈利>=100%) → ma20 趋势线."""
    pct = trailing_stop_pct(profit_pct, spec)
    return cost * (1 + pct) if pct is not None else ma20


def daily_slowbull_pool(
    work: pd.DataFrame, date, board: str, spec, top_n: int
) -> pd.DataFrame:
    """运营日输出: 该日该板 → 硬门槛 → 权重打分 → Top-N, 附买卖信号列.

    work 须含 prepare_adx 指标列 + gate_slow_bull 掩码列 + 信号布尔列
    (load_panel 已一次性算好). 返回含 rk/score/dev5/各信号列的清单.
    """
    gc = f"gate_{spec.gate}"
    if gc not in work.columns:
        raise KeyError(
            f"daily_slowbull_pool 需要预计算掩码列 {gc} (load_panel 已算); "
            f"单日切片无法重算门槛 (量比昨日值需个股历史)"
        )
    mask = (work["date"] == date) & (work["board"] == board) & work[gc]
    day = work[mask]
    if day.empty:
        return pd.DataFrame()
    score = pool_score(day, spec.pool, weights=spec.pool_weights)
    top = select_topn(day, score, top_n)
    if top.empty:
        return top
    top = top.copy()
    top["rk"] = top["score"].rank(ascending=False, method="first").astype(int)
    cols = (
        "symbol",
        "date",
        "close_cont",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "adx",
        "pdi",
        "mdi",
        "adx_rise5",
        "turnover_rate",
        "vol_ratio",
        "pullback_ma5",
        "pullback_ma10",
        "shrink_vol",
        "chase_high",
        "vol_spike_up",
        "adx_falling",
        "below_ma20",
        "adx_broken",
        "big_drop",
        "below_ma5_3d",
        "turnover_spike",
        "tp_80_div",
    )
    have = [c for c in cols if c in day.columns and c not in ("symbol", "date")]
    out = top.merge(day[["symbol", "date"] + have], on=["symbol", "date"], how="left")
    out["dev5"] = out["close_cont"] / out["ma5"] - 1
    out.insert(0, "board", board)
    return out.sort_values(["rk"]).reset_index(drop=True)
