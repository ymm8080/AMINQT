# -*- coding: utf-8 -*-
"""涨跌停判断 (P1, ARCH §4).

按 Universe 区分涨跌停幅度: 主板 ±10%, 创业板/科创板 ±20%, ST ±5%。
买入前查非涨停, 卖出前查非跌停 (铁律)。

涨停价计算: ``round(prev_close * (1 ± limit_pct), 2)`` (A股最小价位 0.01)。
"""

import logging

from app.core.universe_manager import Universe, classify_symbol

logger = logging.getLogger(__name__)

# ── 涨跌停幅度常量 (ARCH §4, 可用 config 覆盖) ──────────────────────
LIMIT_PCT_ST = 0.05        # ST / *ST / 退市整理
LIMIT_PCT_MAIN = 0.10      # 沪深主板
LIMIT_PCT_GROWTH = 0.20    # 创业板 + 科创板

_EPS = 1e-9                # 浮点比较容差


def get_limit_pct(symbol: str, universe: Universe, is_st: bool = False) -> float:
    """返回股票涨跌停幅度 (0.10 / 0.20 / 0.05).

    Args:
        symbol: 股票代码。
        universe: 所属 Universe; Universe.ALL 时按代码前缀路由。
        is_st: 是否 ST 股 (ST 优先于板块, 一律 ±5%)。

    Returns:
        涨跌停幅度 (小数)。
    """
    if is_st:
        return LIMIT_PCT_ST
    if universe == Universe.ALL:
        universe = classify_symbol(symbol)
    if universe == Universe.GROWTH_BOARDS:
        return LIMIT_PCT_GROWTH
    return LIMIT_PCT_MAIN


def is_limit_up(symbol: str, current_price: float, prev_close: float,
                universe: Universe, is_st: bool = False) -> bool:
    """判断是否涨停 (买入前检查, 涨停禁止买入).

    涨停价 = round(prev_close * (1 + limit_pct), 2);
    current_price >= 涨停价 视为涨停。

    Args:
        symbol: 股票代码。
        current_price: 当前价。
        prev_close: 昨收价; <=0 或 NaN 时返回 False (数据异常不拦截)。
        universe: 所属 Universe。
        is_st: 是否 ST 股。

    Returns:
        True 表示已涨停。
    """
    if prev_close is None or not (prev_close > 0):
        logger.warning("%s prev_close 异常 (%s), 涨停检查跳过", symbol, prev_close)
        return False
    pct = get_limit_pct(symbol, universe, is_st)
    limit_price = round(prev_close * (1.0 + pct), 2)
    return current_price >= limit_price - _EPS


def is_limit_down(symbol: str, current_price: float, prev_close: float,
                  universe: Universe, is_st: bool = False) -> bool:
    """判断是否跌停 (卖出前检查, 跌停禁止卖出).

    跌停价 = round(prev_close * (1 - limit_pct), 2);
    current_price <= 跌停价 视为跌停。

    Args:
        symbol: 股票代码。
        current_price: 当前价。
        prev_close: 昨收价; <=0 或 NaN 时返回 False (数据异常不拦截)。
        universe: 所属 Universe。
        is_st: 是否 ST 股。

    Returns:
        True 表示已跌停。
    """
    if prev_close is None or not (prev_close > 0):
        logger.warning("%s prev_close 异常 (%s), 跌停检查跳过", symbol, prev_close)
        return False
    pct = get_limit_pct(symbol, universe, is_st)
    limit_price = round(prev_close * (1.0 - pct), 2)
    return current_price <= limit_price + _EPS
