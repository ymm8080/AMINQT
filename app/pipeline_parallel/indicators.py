# -*- coding: utf-8 -*-
"""ADX 慢牛系统 — 指标 + 打分因子列 (2026-08-05, ADX 设计文档 v1.0).

铁律:
  - 全向量化: 一律 groupby("symbol") + rolling/shift/ewm/pct_change, 禁 for 循环遍历股票
    (逐股子帧物化会触发 pandas block consolidation → 本机 15.8GB 物理 OOM);
  - 按 [symbol, date] 排序 (对齐 feature_engine_v35);
  - PIT: 只用 t 及更早数据 (shift/rolling 不引用未来);
  - 连续价格用后复权 (close_hfq/high_hfq/low_hfq), 避免除权跳变误判趋势破位.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from app.pipeline_parallel.config import ADX_SPEC

_MA_WINDOWS = (5, 10, 20, 60)
_CYQ_PANEL = os.path.join("data", "cyq_panel.parquet")


def _gshift(s: pd.Series, key: pd.Series, n: int = 1) -> pd.Series:
    """按 symbol 分组 shift n 日 (PIT, 不越符号边界)."""
    return s.groupby(key, sort=False).shift(n)


def _gpct(s: pd.Series, key: pd.Series, periods: int = 1) -> pd.Series:
    """按 symbol 分组 pct_change (组内首日 NaN)."""
    return s.groupby(key, sort=False).pct_change(periods)


def _groll(s: pd.Series, key: pd.Series, n: int, agg: str) -> pd.Series:
    """按 symbol 分组滚动 n 日聚合, 返回对齐全帧的 Series (无逐股子帧物化)."""
    r = getattr(s.groupby(key, sort=False).rolling(n, min_periods=n), agg)()
    return r.reset_index(level=0, drop=True)


def _gema(s: pd.Series, key: pd.Series, n: int) -> pd.Series:
    """按 symbol 分组 Wilder EMA (ewm span=n, adjust=False)."""
    r = s.groupby(key, sort=False).ewm(span=n, adjust=False).mean()
    return r.reset_index(level=0, drop=True)


def _limit_pct_col(df: pd.DataFrame) -> pd.Series:
    """每行跌停阈值: 创业板/科创 (30/68 前缀) 20%, 主板 10% (ST 不细分, 保守取板限)."""
    if "board" in df.columns:
        return pd.Series(np.where(df["board"] == "dual", 0.20, 0.10), index=df.index)
    pre = df["symbol"].astype(str).str[:2]
    return pd.Series(np.where(pre.isin(("30", "68")), 0.20, 0.10), index=df.index)


def _merge_pct_70(work: pd.DataFrame) -> pd.DataFrame:
    """CYQ 筹码集中度 pct_70_con 补列 (cyq_panel.parquet); 缺失文件则跳过 (打分自动缺列归一).

    不用 pd.merge: 宽表 join 会触发 block consolidation (~2× 表大小连续块) → 本机 OOM
    (见 memory/machine-ram-block-consolidation). 改 MultiIndex keyed map 向量查找.
    """
    if not os.path.exists(_CYQ_PANEL):
        return work
    cyq = pd.read_parquet(_CYQ_PANEL, columns=["symbol", "date", "pct_70_con"])
    cyq = cyq.drop_duplicates(["symbol", "date"], keep="last")
    lut = cyq.set_index(["symbol", "date"])["pct_70_con"]
    work["pct_70_con"] = pd.MultiIndex.from_arrays([work["symbol"], work["date"]]).map(
        lut
    )
    del cyq, lut
    gc.collect()
    return work


def prepare_adx(work: pd.DataFrame, spec: dict | None = None) -> pd.DataFrame:
    """整面板一次性计算 ADX 慢牛指标 + 打分因子列 (全向量化, PIT), 返回加列后的 work.

    调用方应 `work = indicators.prepare_adx(work)`. 幂等: 已存在的同名列会被覆盖.
    前提: 面板含 volume/turnover_rate/close_hfq(或 close) 等列; 缺失列自动跳过.
    """
    if spec is None:
        spec = ADX_SPEC
    # 排序保证: load_panel 已按 [symbol,date] 排; 合成/其他调用未排时才重排 (省一次全表拷贝)
    need_sort = not work["symbol"].is_monotonic_increasing
    if not need_sort:
        need_sort = not work.groupby("symbol", sort=False)[
            "date"
        ].is_monotonic_increasing.all()
    work = (
        work.sort_values(["symbol", "date"]).reset_index(drop=True)
        if need_sort
        else work.reset_index(drop=True)
    )
    key = work["symbol"]

    c = "close_hfq" if "close_hfq" in work.columns else "close"
    h = "high_hfq" if "high_hfq" in work.columns else "high"
    low = "low_hfq" if "low_hfq" in work.columns else "low"
    work["_c"] = work[c]
    work["_h"] = work[h]
    work["_l"] = work[low]
    work["_vol"] = work["volume"]
    work["_limit_pct"] = _limit_pct_col(work)
    # 连续价 (后复权) 供门槛/信号对比连续 MA, 避免除权跳变误判破位
    work["close_cont"] = work["_c"]
    close, high, low_ = work["_c"], work["_h"], work["_l"]
    work["_ret"] = _gpct(close, key)  # 单日收益 (组内)
    ret = work["_ret"]

    # MA 系 + 斜率 (斜率 = 近 slope_lookback 日变化率)
    for n in _MA_WINDOWS:
        work[f"ma{n}"] = _groll(close, key, n, "mean")
    k = spec["slope_lookback"]
    for n in (5, 10, 20):
        ma = work[f"ma{n}"]
        prev = _gshift(ma, key, k)
        work[f"ma_slope{n}"] = (ma - prev) / prev

    # ADX/+DI/-DI (Wilder's, EMA 平滑, 文档附录 Step1-4)
    prev_h, prev_l, prev_c = _gshift(high, key), _gshift(low_, key), _gshift(close, key)
    up, down = high - prev_h, prev_l - low_
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum.reduce([high - low_, (high - prev_c).abs(), (low_ - prev_c).abs()])
    n = spec["adx_period"]
    ema_tr = _gema(pd.Series(tr, index=work.index), key, n)
    denom = ema_tr.replace(0, np.nan)
    pdi = 100 * _gema(pd.Series(plus_dm, index=work.index), key, n) / denom
    mdi = 100 * _gema(pd.Series(minus_dm, index=work.index), key, n) / denom
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    work["pdi"] = pdi
    work["mdi"] = mdi
    work["adx"] = _gema(dx, key, n)
    work["adx_rise5"] = work["adx"] - _gshift(
        work["adx"], key, spec["adx_rise_lookback"]
    )
    # 打分因子: ADX 25-40 最佳, >40 过热不再加分 (clip 封顶)
    work["adx_score"] = work["adx"].clip(lower=0.0, upper=spec["adx_optimal_max"])

    # 低波动约束 + 跌停计数
    lp = work["_limit_pct"]
    amp = (high - low_) / _gshift(close, key)
    work["amplitude_20"] = _groll(amp, key, 20, "mean")
    work["max_drop_20"] = _groll(ret, key, 20, "min")
    work["limit_down_20"] = _groll(ret <= -(lp - 0.001), key, 20, "sum")

    # 量能 / 换手
    for wn in (5, 10, 20):
        work[f"ma_vol_{wn}"] = _groll(work["_vol"], key, wn, "mean")
    if "volume_ratio" in work.columns:
        work["vol_ratio"] = work["volume_ratio"]
    else:  # 兜底: 当日量 / 前5日均量 (量比标准口径)
        work["vol_ratio"] = work["_vol"] / _gshift(work["ma_vol_5"], key)

    # 打分因子: 均线紧密度 / 夏普 / 量价相关 / RPS 用 60日涨幅
    work["ma_tightness"] = -(work["ma5"] - work["ma20"]) / close
    mean20 = _groll(ret, key, spec["sharpe_lookback"], "mean")
    std20 = _groll(ret, key, spec["sharpe_lookback"], "std")
    work["sharpe_20"] = (mean20 / std20).replace([np.inf, -np.inf], np.nan)
    work["_vol_pct"] = _gpct(work["_vol"], key)
    x, y = ret, work["_vol_pct"]
    xy = (x * y).replace([np.inf, -np.inf], np.nan)
    mxy = _groll(xy, key, 5, "mean")
    mx = _groll(x, key, 5, "mean")
    my = _groll(y, key, 5, "mean")
    varx = _groll(x * x, key, 5, "mean") - mx * mx
    vary = _groll(y * y, key, 5, "mean") - my * my
    corr = (mxy - mx * my) / np.sqrt(np.maximum(varx * vary, 0.0))
    work["pv_corr_5"] = corr.replace([np.inf, -np.inf], np.nan)
    work["ret60"] = _gpct(close, key, spec["rps_lookback"])

    # 截面因子: RPS = 全市场 60日涨幅分位 (文档 "vs 全市场")
    work["rps_60"] = work.groupby("date")["ret60"].rank(pct=True)
    # 筹码集中度 pct_70 (CYQ 面板补列)
    if "pct_70_con" not in work.columns:
        work = _merge_pct_70(work)
    # 中性填充 (防 pool_score NaN 中毒): 无数据不给信用 → 0 (pct_70 也是 0 不信用)
    for col in ("margin_balance_chg_5d", "pv_corr_5", "sharpe_20", "pct_70_con"):
        if col in work.columns:
            work[col] = work[col].fillna(0.0)

    work = work.drop(
        columns=["_c", "_h", "_l", "_vol", "_limit_pct", "_ret", "_vol_pct", "ret60"],
        errors="ignore",
    )
    gc.collect()
    return work
