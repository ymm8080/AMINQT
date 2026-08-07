"""ADX 慢牛系统 — 硬门槛 (2026-08-05, ADX 设计文档 v1.0 §2.2).

四道门槛全部满足 (AND) 才进打分池:
  门槛一 均线多头排列: close>ma5>ma10>ma20>ma60, ma5/10/20 斜率>0, (ma5-ma10)/ma10 乖离<5%
  门槛二 ADX 趋势确认: adx>25 且 pdi>mdi 且 adx 近5日上升 (adx_rise5>0)
  门槛三 低波动约束: 20日均振幅<6% 且 20日最大单日跌幅<5% 且 20日无跌停
  门槛四 量价健康: 5日均量>10日>20日 且 昨日量比<3 且 换手率 3%-15%

所有判断 PIT (只用 t 及更早): 量比取昨日值 (groupby symbol shift, 防符号边界串行);
均线/ADX/波动/量能滚动列已由 indicators.prepare_adx 计算且只用 t 及更早。
价格一律用连续价 close_cont (后复权), 与连续 MA 自洽, 避免除权跳变误判破位。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from app.pipeline_parallel.config import ADX_SPEC

GATES: dict[str, Callable] = {}


def compute_gate(df: pd.DataFrame, name: str) -> pd.Series:
    """按注册名对 df 计算硬门槛掩码 (索引对齐 df). 未注册 → 大声失败."""
    if name not in GATES:
        raise KeyError(f"未注册的硬门槛: {name} (GATES={list(GATES)})")
    return GATES[name](df)


def apply_gate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """按门槛过滤 df; 缺指标列 (该面板未 prepare 慢牛) → 返回空帧 (无候选, 非配置错误).

    run_system/导出路径的兜底: 生产 load_panel 已预计算 gate 掩码列, 用缓存路径;
    此分支只服务未 prepare 的通用面板 (如合成测试帧), 慢牛无候选即空, 不崩.
    """
    try:
        return df[compute_gate(df, name)]
    except KeyError:
        return df.iloc[0:0]


def _yesterday(df: pd.DataFrame, col: str) -> pd.Series:
    """昨日值 (PIT): 按 symbol 内 shift(1), 符号边界不串行."""
    return df.groupby("symbol")[col].shift(1)


def slow_bull_gate(df: pd.DataFrame, spec: dict | None = None) -> pd.Series:
    """慢牛四门槛 AND. 输入 df 必须含 indicators.prepare_adx 输出的指标列."""
    if spec is None:
        spec = ADX_SPEC
    req = (
        "close_cont",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "ma_slope5",
        "ma_slope10",
        "ma_slope20",
        "adx",
        "pdi",
        "mdi",
        "adx_rise5",
        "amplitude_20",
        "max_drop_20",
        "limit_down_20",
        "ma_vol_5",
        "ma_vol_10",
        "ma_vol_20",
        "vol_ratio",
        "turnover_rate",
    )
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"慢牛硬门槛缺指标列: {missing} (先调 indicators.prepare_adx)")

    g1 = (
        (df["close_cont"] > df["ma5"])
        & (df["ma5"] > df["ma10"])
        & (df["ma10"] > df["ma20"])
        & (df["ma20"] > df["ma60"])
        & (df["ma_slope5"] > 0)
        & (df["ma_slope10"] > 0)
        & (df["ma_slope20"] > 0)
        & ((df["ma5"] - df["ma10"]).abs() / df["ma10"] < spec["ma_bias_max"])
    )
    g2 = (df["adx"] > spec["adx_min"]) & (df["pdi"] > df["mdi"]) & (df["adx_rise5"] > 0)
    g3 = (
        (df["amplitude_20"] < spec["amplitude_20_max"])
        & (df["max_drop_20"] > -spec["max_drop_20_max"])  # 最大单日跌幅 < 5%
        & (df["limit_down_20"] == 0)
    )
    g4 = (
        (df["ma_vol_5"] > df["ma_vol_10"])
        & (df["ma_vol_10"] > df["ma_vol_20"])
        & (_yesterday(df, "vol_ratio") < spec["vol_ratio_max"])
        & df["turnover_rate"].between(spec["turnover_min"], spec["turnover_max"])
    )
    return g1 & g2 & g3 & g4


GATES["slow_bull"] = slow_bull_gate
