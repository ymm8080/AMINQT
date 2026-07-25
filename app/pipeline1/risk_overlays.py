"""
E6/E8/E9 组合构建风控覆盖层 (PIPELINE1_V3.8 §四, 安全网 #20, 检查清单 #74-#78)
================================================================================
[E6] 流动性承载约束: 单票买入金额 ≤ ADV20×1% (bear 收紧至 0.5%);
     全清单 ADV 异常 (中位数 < 近250日20%分位) → 系统性流动性枯竭, 锁仓不交易.
[E8] 相关性簇阻断: 持仓+候选内任意两票 20 日收益 corr>0.7 → 归同簇,
     簇总权重 ≤ 15% (bear 收紧至 12%).
[E9] 波动率熔断 (前置型): 全市场 5 日平均振幅 > 3% → multiplier 压至 0.10;
     基线 = 滚动 250 日中位数 (熊市自适应); 短期波动率 > 1.5×基线 → multiplier×0.6.
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
# E9
AMPLITUDE_FUSE = 0.03  # 5 日平均振幅 > 3% → 熔断
FUSE_MULTIPLIER = 0.10  # 熔断后仓位乘数
VOL_SURGE_RATIO = 1.5  # 短期波动率 > 1.5×基线 → ×0.6
VOL_SURGE_DAMP = 0.6


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


def systemic_liquidity_check(adv20_today: pd.Series, adv20_hist_250d: pd.Series) -> bool:
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
        members = {s} | {
            t for t in cols if t != s and corr.loc[s, t] > threshold
        }
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
            logger.warning("E8 簇阻断: %s 总权重 %.1f%% > %.0f%%, 按比例缩减",
                           sorted(members), total * 100, cap * 100)
    return out


# ============================================================
# E9 波动率熔断 (前置型)
# ============================================================
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
        logger.error("E9 波动率熔断: 5日平均振幅 %.2f%% > 3%%, 仓位压至 0.10",
                     avg_amplitude_5d * 100)
        m = min(m, FUSE_MULTIPLIER)
    if baseline_250d > 0 and short_vol > VOL_SURGE_RATIO * baseline_250d:
        logger.warning("E9 短期波动率 %.4f > 1.5×基线 %.4f, 乘数 ×0.6",
                       short_vol, baseline_250d)
        m *= VOL_SURGE_DAMP
    return m
