"""
E6/E8/E9 组合构建风控覆盖层 + P23.1-P23.4 分位数驱动扩展 (IMPLEMENTATION_PLAN_v3.2)
====================================================================================
[E6] 流动性承载约束: 单票买入金额 ≤ ADV20×1% (bear 收紧至 0.5%);
     全清单 ADV 异常 (中位数 < 近250日20%分位) → 系统性流动性枯竭, 锁仓不交易.
[E8] 相关性簇阻断: 持仓+候选内任意两票 20 日收益 corr>0.7 → 归同簇,
     簇总权重 ≤ 15% (bear 收紧至 12%).
[E9] 波动率熔断 (前置型, P23.4 分位数驱动): 全市场 5 日平均振幅 > P90 → 熔断;
     短期波动率 > 1.5×基线 → ×0.6.
[P23.1] VWAP 支撑过滤 (D-12): 尾盘价格在 VWAP 之上 = 机构成本线有支撑.
[P23.2] 14:50 复检闸门 (D-13): pre_buy_check + knife_catch_filter (分位数驱动).
[P23.3] 板块动量过滤 (D-14): 板块今日收益高于自身历史中位.
[P23.4] 波动率异常过滤 (D-15/D-18/D-19): volatility_fuse (分位数) + volatility_stock_filter
        + volatility_knife_catch.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# E6
LIQUIDITY_RATIO = 0.01  # 单票买入 ≤ ADV20 × 1%
LIQUIDITY_RATIO_BEAR = 0.005  # bear 收紧至 0.5%
# E8
CORR_THRESHOLD = 0.7  # 20 日收益 corr > 0.7 → 归同簇
CLUSTER_CAP = 0.15  # 簇总权重上限
CLUSTER_CAP_BEAR = 0.12  # bear 收紧
# E9 — 分位数驱动 (P23.4), 旧固定阈值保留作 fallback
AMPLITUDE_FUSE = 0.03  # 旧固定阈值 (分位数不可用时 fallback)
FUSE_MULTIPLIER = 0.10  # 熔断后仓位乘数
VOL_SURGE_RATIO = 1.5  # 短期波动率 > 1.5×基线 → ×0.6
VOL_SURGE_DAMP = 0.6
# P23.4 分位数驱动
AMPLITUDE_P90_FUSE = 0.90  # 振幅 P90 分位触发熔断
AMPLITUDE_P75_WARN = 0.75  # 振幅 P75 分位警告
VOL_STOCK_SPIKE_PCT = 0.90  # 个股振幅 > P90
VOL_STOCK_CLUSTER_PCT = 0.85  # 5 日内 2 日 > P85
VOL_STOCK_CLUSTER_DAYS = 2  # 5 日内 ≥2 日异常


# ============================================================
# E6 流动性承载
# ============================================================
def liquidity_cap(order_value: float, adv20: float, bear: bool = False) -> float:
    """单票流动性上限乘数: min(1.0, ADV20 × ratio / order_value).

    单票买入金额不得超过其 20 日日均成交额的 1% (bear 0.5%).
    """
    ratio = LIQUIDITY_RATIO_BEAR if bear else LIQUIDITY_RATIO
    if pd.isna(adv20) or adv20 <= 0:
        return 0.0  # ADV 未知 → 不允许买入 (保守)
    return float(min(1.0, adv20 * ratio / max(order_value, 1.0)))


def systemic_liquidity_check(
    adv20_today: pd.Series, adv20_hist_250d: pd.Series
) -> bool:
    """全清单 ADV 异常 → True = 系统性流动性枯竭, 锁仓不交易.

    判定: 当日 ADV20 中位数 < 近 250 日 ADV20 中位数的 20% 分位.
    """
    if len(adv20_today) == 0 or len(adv20_hist_250d) < 20:
        return False
    threshold = float(np.nanpercentile(adv20_hist_250d, 20))
    med = float(np.nanmedian(adv20_today))
    exhausted = med < threshold
    if exhausted:
        logger.error(
            "E6 系统性流动性枯竭: ADV20 中位数 %.2e < 250日20%%分位 %.2e, 锁仓",
            med,
            threshold,
        )
    return exhausted


# ============================================================
# E8 相关性簇阻断
# ============================================================
def cluster_block(
    symbols: list[str],
    ret_window_20d: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> list[set]:
    """20 日收益 corr > threshold 归同簇.

    Args:
        symbols: 持仓+候选代码
        ret_window_20d: 日收益矩阵 (index=date, columns=symbol)
    Returns:
        [set(symbol, ...), ...] 每簇总权重 ≤ CLUSTER_CAP (bear 12%).
    """
    cols = [s for s in symbols if s in ret_window_20d.columns]
    corr = ret_window_20d[cols].corr().abs()
    clusters, assigned = [], set()
    for s in cols:
        if s in assigned:
            continue
        members = {s} | {t for t in cols if t != s and corr.loc[s, t] > threshold}
        assigned |= members
        clusters.append(members)
    # 无收益数据的票各自独立成簇 (不阻断)
    for s in symbols:
        if s not in assigned:
            clusters.append({s})
    return clusters


def apply_cluster_caps(
    weights: pd.Series,
    clusters: list[set],
    cap: float = CLUSTER_CAP,
) -> pd.Series:
    """簇总权重 ≤ cap, 超出的簇内按比例缩减.

    Args:
        weights: index=symbol 的目标权重
        clusters: cluster_block 输出
    Returns:
        调整后的权重 (不重新归一 — 缩减部分留现金).
    """
    out = weights.copy()
    for members in clusters:
        idx = [s for s in members if s in out.index]
        total = float(out.loc[idx].sum()) if idx else 0.0
        if total > cap and total > 0:
            out.loc[idx] = out.loc[idx] * (cap / total)
            logger.warning(
                "E8 簇阻断: %s 总权重 %.1f%% > %.0f%%, 按比例缩减",
                sorted(members),
                total * 100,
                cap * 100,
            )
    return out


# ============================================================
# E9 波动率熔断 (前置型) — 分位数驱动版 (P23.4)
# ============================================================
def percentile_rank(hist: np.ndarray, current: float) -> float:
    """current 在 hist 中的分位 (0.0~1.0)."""
    hist = np.asarray(hist)
    if len(hist) < 2:
        return 0.5
    return float((hist < current).mean())


def volatility_fuse(market_state: dict) -> tuple:
    """D-19: 全市场波动率熔断 — 分位数驱动 (替代固定 AMPLITUDE_FUSE=0.03).

    Args:
        market_state: 需含 'amplitude_5d_250d' (历史振幅序列) 和 'amplitude_5d' (当前).
    Returns:
        (fused: bool, label: str)
    """
    hist_amp = np.asarray(market_state.get("amplitude_5d_250d", []))
    current_amp = market_state.get("amplitude_5d", 0.0)

    if len(hist_amp) >= 20:
        p90_amp = np.percentile(hist_amp, 90)
        if current_amp > p90_amp:
            logger.error(
                "E9 分位数熔断: 5日振幅 %.2f%% > P90 %.2f%%",
                current_amp * 100,
                p90_amp * 100,
            )
            return True, f"VOL_FUSE_90PCT_{current_amp:.2%}"
        p75_amp = np.percentile(hist_amp, 75)
        if current_amp > p75_amp:
            logger.warning(
                "E9 分位数警告: 5日振幅 %.2f%% > P75 %.2f%%",
                current_amp * 100,
                p75_amp * 100,
            )
            return False, f"VOL_WARN_75PCT_{current_amp:.2%}"
    else:
        # 历史不足, fallback 固定阈值
        if current_amp > AMPLITUDE_FUSE:
            return True, f"VOL_FUSE_FIXED_{current_amp:.2%}"

    return False, "NORMAL"


def vol_breaker_multiplier(
    avg_amplitude_5d: float,
    short_vol: float,
    baseline_250d: float,
) -> float:
    """E9 仓位乘数 (与 D4 position_multiplier 串联, 取乘积).

    Args:
        avg_amplitude_5d: 全市场 5 日平均振幅 (high-low)/pre_close
        short_vol: 短期波动率 (如 20 日日收益 std)
        baseline_250d: 滚动 250 日波动率中位数 (熊市自适应基线)
    Returns:
        乘数: 振幅熔断 → 0.10; 短期波动率>1.5×基线 → 额外 ×0.6.
    """
    m = 1.0
    if avg_amplitude_5d > AMPLITUDE_FUSE:
        logger.error(
            "E9 波动率熔断: 5日平均振幅 %.2f%% > 3%%, 仓位压至 0.10",
            avg_amplitude_5d * 100,
        )
        m = min(m, FUSE_MULTIPLIER)
    if baseline_250d > 0 and short_vol > VOL_SURGE_RATIO * baseline_250d:
        logger.warning(
            "E9 短期波动率 %.4f > 1.5×基线 %.4f, 乘数 ×0.6", short_vol, baseline_250d
        )
        m *= VOL_SURGE_DAMP
    return m


# ============================================================
# P23.1 VWAP 支撑过滤 (D-12)
# ============================================================
def vwap_support_filter(stock: dict) -> bool:
    """D-12: 尾盘价格在 VWAP 之上 = 机构成本线有支撑."""
    price_1455 = stock["price_1455"] if "price_1455" in stock else 0
    vwap_1455 = stock["vwap_1455"] if "vwap_1455" in stock else 0
    if vwap_1455 <= 0:
        return False  # 无 VWAP 数据 → 保守拒绝
    return bool(price_1455 >= vwap_1455)


# ============================================================
# P23.2 14:50 复检闸门 — 分位数驱动版 (D-13 + D-19 接飞刀)
# ============================================================
def knife_catch_filter(stock: dict) -> bool:
    """D-19: 接飞刀检测 — 分位数驱动.

    对该票而言是极端下跌 → 禁止买入.
    """
    hist_drop = np.asarray(stock.get("daily_return_20d", []))
    hist_drop = hist_drop[hist_drop < 0]
    if len(hist_drop) < 5:
        return False
    today_drop = stock.get("day_change", 0)
    p90_drop = np.percentile(hist_drop, 90)
    return bool(today_drop < p90_drop)


def pre_buy_check(stock: dict, market_state: dict) -> list:
    """D-13: 14:50 复检闸门 — 买入前最后一道防线.

    Returns:
        blocks: 阻断原因列表 (空列表 = 通过).
    """
    blocks = []

    # P1 个股：分位数驱动接飞刀
    if knife_catch_filter(stock):
        blocks.append("INDIVIDUAL_KNIFE_CATCH")
        logger.warning(
            "14:50 复检: %s 接飞刀阻断 (今日跌幅在历史 P90 之外)",
            stock.get("symbol", "?"),
        )

    # P2 板块：个股弱于板块
    sector_drop = stock.get("sector_index_change", 0)
    stock_drop = stock.get("day_change", 0)
    if stock_drop < sector_drop - 0.02:
        blocks.append("WEAKER_THAN_SECTOR")
        logger.warning(
            "14:50 复检: %s 弱于板块 (个股 %.2f%% vs 板块 %.2f%%)",
            stock.get("symbol", "?"),
            stock_drop * 100,
            sector_drop * 100,
        )

    # P3 早盘恐慌
    if market_state.get("morning_median_drop", 0) > 0.03:
        blocks.append("MORNING_PANIC")
        logger.warning("14:50 复检: 早盘恐慌 (中位跌幅 > 3%%)")

    return blocks


# ============================================================
# P23.3 板块动量过滤 (D-14)
# ============================================================
def sector_momentum_filter(sector: dict) -> bool:
    """D-14: 板块今日收益高于自身历史中位."""
    daily_rets = np.asarray(sector.get("daily_return_250d", []))
    if len(daily_rets) < 20:
        return True  # 数据不足 → 放行
    median_return = float(np.median(daily_rets))
    return bool(sector.get("return_today", 0) > median_return)


def rank_sectors_by_momentum(sectors: list[dict], top_n: int = 5) -> list[dict]:
    """D-14: 板块动量排名, 取前 N 且今日收益 > 0."""
    ranked = [s for s in sectors if sector_momentum_filter(s)]
    ranked.sort(key=lambda s: s.get("return_today", 0), reverse=True)
    return ranked[:top_n]


# ============================================================
# P23.4 波动率异常过滤 — 分位数驱动 (D-15/D-18/D-19)
# ============================================================
def volatility_stock_filter(stock: dict) -> str:
    """D-18: 个股波动率异常 — 历史分位.

    Returns:
        'SPIKE_90PCT' | 'VOL_CLUSTER_85PCT' | 'PASS'
    """
    hist_amp = np.asarray(stock.get("amplitude_250d", []))
    current_amp = stock.get("amplitude_today", 0)
    if len(hist_amp) < 20:
        return "PASS"  # 数据不足不判定

    if percentile_rank(hist_amp, current_amp) > VOL_STOCK_SPIKE_PCT:
        return "SPIKE_90PCT"

    recent_amps = np.asarray(stock.get("amplitude_5d", [current_amp]))
    spike_days = sum(
        percentile_rank(hist_amp, a) > VOL_STOCK_CLUSTER_PCT for a in recent_amps
    )
    if spike_days >= VOL_STOCK_CLUSTER_DAYS:
        return "VOL_CLUSTER_85PCT"

    return "PASS"


def volatility_knife_catch(stock: dict) -> bool:
    """D-19: 波动率维度接飞刀 — 当前波动率 > P90."""
    hist_vol = np.asarray(stock.get("volatility_20d", []))
    current_vol = stock.get("volatility_today", 0)
    if len(hist_vol) < 20:
        return False
    return bool(percentile_rank(hist_vol, current_vol) > 0.90)
