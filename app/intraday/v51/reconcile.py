"""
逐笔核对 + 滑点偏差日报 (V5.1 P20.2, Shadow 门禁条件 4)
==============================================================
回测与 shadow/实盘逐笔一致率 > 99% (Shadow 门禁, 检查清单 #15);
滑点偏差日报: 实盘滑点 vs 模型假设 (E5 分层), 超标 → 修 E5 分层 (P19.2 W9).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .safe_div import safe_divide

logger = logging.getLogger(__name__)

MATCH_RATE_MIN = 0.99  # 逐笔一致率门禁 > 99%
SLIPPAGE_WARN_MULT = 2.0  # 实盘滑点 > 2× 模型假设 → 告警修 E5


def trade_match_rate(backtest_trades: pd.DataFrame, live_trades: pd.DataFrame) -> dict:
    """回测 vs shadow/实盘逐笔一致率 (信号维度: symbol+side+rule).

    Returns:
        {'match_rate', 'n_backtest', 'n_live', 'missing_in_live', 'extra_in_live',
         'pass'} pass = 一致率 > 99% (Shadow 门禁条件 4).
    """
    key = ["symbol", "side", "rule"]
    bt = set(map(tuple, backtest_trades[key].values)) if len(backtest_trades) else set()
    lv = set(map(tuple, live_trades[key].values)) if len(live_trades) else set()
    if not bt:
        return {
            "match_rate": 1.0,
            "n_backtest": 0,
            "n_live": len(lv),
            "missing_in_live": [],
            "extra_in_live": sorted(lv),
            "pass": len(lv) == 0,
        }
    matched = bt & lv
    rate = safe_divide(float(len(matched)), float(len(bt)))
    missing = sorted(bt - lv)
    extra = sorted(lv - bt)
    if missing or extra:
        logger.error(
            "逐笔核对不一致: 缺失 %s / 多出 %s (一致率 %.1f%%)",
            missing,
            extra,
            rate * 100,
        )
    return {
        "match_rate": round(rate, 4),
        "n_backtest": len(bt),
        "n_live": len(lv),
        "missing_in_live": missing,
        "extra_in_live": extra,
        "pass": rate > MATCH_RATE_MIN,
    }


def slippage_report(fills: pd.DataFrame, assumed_slippage: pd.Series) -> dict:
    """滑点偏差日报: 实盘滑点 vs 模型假设 (E5 分层).

    Args:
        fills: 成交记录 (含 symbol, slippage 实际值)
        assumed_slippage: {symbol: E5 分层假设滑点}
    Returns:
        {'mean_actual', 'mean_assumed', 'max_ratio', 'exceed_symbols', 'alert'}
        alert=True → 滑点超标, 修 E5 分层 (P19.2 W9 处置规则).
    """
    df = fills.copy()
    df["assumed"] = df["symbol"].map(assumed_slippage).fillna(0.0)
    df["ratio"] = df["slippage"].abs() / df["assumed"].replace(0, np.nan)
    exceed = df[df["ratio"] > SLIPPAGE_WARN_MULT]
    alert = len(exceed) > 0
    if alert:
        logger.error(
            "滑点超标: %s 实际滑点 > 2× 模型假设, 修 E5 分层", sorted(exceed["symbol"])
        )
    return {
        "mean_actual": round(float(df["slippage"].abs().mean()), 6) if len(df) else 0.0,
        "mean_assumed": round(float(df["assumed"].mean()), 6) if len(df) else 0.0,
        "max_ratio": round(float(df["ratio"].max()), 2) if len(df) else 0.0,
        "exceed_symbols": sorted(exceed["symbol"]) if alert else [],
        "alert": alert,
    }
