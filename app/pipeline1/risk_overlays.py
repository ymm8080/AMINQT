"""
E6/E8/E9 组合构建风控覆盖层 + FINAL STOCK SCAN 过滤
====================================================================================
[E6] 流动性承载约束: 单票买入金额 ≤ ADV20×1% (bear 收紧至 0.5%).
[E8] 相关性簇阻断: 持仓+候选内任意两票 20 日收益 corr>0.7 → 归同簇,
     簇总权重 ≤ 15% (bear 收紧至 12%).
[E9] 波动率熔断仓位乘数: 全市场 5 日平均振幅 > 3% → 0.10; 短期波动率 > 1.5×基线 → ×0.6.
[FINAL STOCK SCAN] 出名单风控: 近一周大宗交易剔除 + 未来 N 天解禁比例超阈值剔除.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.config_loader import load_config

logger = logging.getLogger(__name__)

# E6
LIQUIDITY_RATIO = 0.01  # 单票买入 ≤ ADV20 × 1%
LIQUIDITY_RATIO_BEAR = 0.005  # bear 收紧至 0.5%
# E8
CORR_THRESHOLD = 0.7  # 20 日收益 corr > 0.7 → 归同簇
CLUSTER_CAP = 0.15  # 簇总权重上限
CLUSTER_CAP_BEAR = 0.12  # bear 收紧
# E9 固定阈值 (vol_breaker_multiplier fallback)
AMPLITUDE_FUSE = 0.03
FUSE_MULTIPLIER = 0.10  # 熔断后仓位乘数
VOL_SURGE_RATIO = 1.5  # 短期波动率 > 1.5×基线 → ×0.6
VOL_SURGE_DAMP = 0.6

# ============================================================
# FINAL STOCK SCAN — 大宗交易风控过滤 (用户定案 2026-08-03)
# 出名单时读 raw 缓存, 剔除近一周有大宗交易的候选 (风控, 非 alpha)
# ============================================================
try:
    _FS_CFG = load_config("data_pipeline_config").get("final_stock_scan", {})
except Exception:
    _FS_CFG = {}

BLOCK_TRADE_CACHE = _FS_CFG.get(
    "block_trade_cache",
    "data/supply_cache/alt_data/block_trade/block_trade_full.parquet",
)
BLOCK_TRADE_LOOKBACK_DAYS = int(_FS_CFG.get("lookback_days", 5))
BLOCK_TRADE_STALE_MAX_DAYS = int(_FS_CFG.get("stale_max_days", 7))

# share_float (限售股解禁) — 未来 N 天内解禁比例累计超阈值 → 剔除 (2026-08-03 用户定案)
SHARE_FLOAT_CACHE = _FS_CFG.get(
    "share_float_cache",
    "data/supply_cache/alt_data/share_float/share_float_full.parquet",
)
# 相对路径按 repo 根解析 (与 _daily_fetch.py 一致), 避免 CWD != repo root 时读不到缓存.
# 本文件在 app/pipeline1/ 下, 到 repo 根需 3 层 parent (曾用 2 层 → 错解析到 app/data/...).
SHARE_FLOAT_CACHE = (
    SHARE_FLOAT_CACHE
    if os.path.isabs(SHARE_FLOAT_CACHE)
    else str(Path(__file__).resolve().parent.parent.parent / SHARE_FLOAT_CACHE)
)
UNLOCK_WINDOW_DAYS = int(_FS_CFG.get("unlock_window_days", 30))
UNLOCK_RATIO_THRESHOLD = float(_FS_CFG.get("unlock_ratio_threshold", 5.0))
UNLOCK_STALE_MAX_DAYS = int(_FS_CFG.get("unlock_stale_max_days", 7))


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
# E9 波动率熔断仓位乘数
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
# FINAL STOCK SCAN — 大宗交易风控过滤 (用户定案 2026-08-03)
# ============================================================
def block_trade_recent_scan(
    symbols: list[str],
    ref_date,
    cache_path: str = BLOCK_TRADE_CACHE,
    lookback_days: int = BLOCK_TRADE_LOOKBACK_DAYS,
    stale_max_days: int = BLOCK_TRADE_STALE_MAX_DAYS,
) -> set[str]:
    """FINAL STOCK SCAN: 返回 symbols 中近 lookback_days 个交易日有大宗交易的子集.

    名单生成时读 raw 缓存 (block_trade_full.parquet), 剔除近一周有大宗的候选 —
    用户定案 (2026-08-03): 大宗特征 IC 全负不做 alpha, 只做最后风控筛选.
    数据缺失/过期 → 返回空集 (fail-open, 只告警不阻断):
    安全网不应因数据源抖动而清空当日名单.
    """
    if not symbols:
        return set()
    ref = pd.Timestamp(ref_date)
    try:
        bt = pd.read_parquet(cache_path, columns=["symbol", "date"])
    except Exception as exc:
        logger.warning("FINAL STOCK SCAN: 读大宗缓存失败, 跳过扫描: %s", exc)
        return set()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt[bt["date"] <= ref]
    if len(bt) == 0:
        return set()
    if (ref - bt["date"].max()).days > stale_max_days:
        logger.warning(
            "FINAL STOCK SCAN: 缓存过期 (最新 %s vs 名单日 %s, 差 >%d 天), 跳过扫描",
            bt["date"].max().date(),
            ref.date(),
            stale_max_days,
        )
        return set()
    days = np.sort(bt["date"].unique())
    window_start = days[-lookback_days]
    recent = set(bt.loc[bt["date"] >= window_start, "symbol"].unique())
    excluded = {s for s in symbols if s in recent}
    if excluded:
        logger.warning(
            "FINAL STOCK SCAN: 剔除 %d 只近 %d 交易日有大宗: %s",
            len(excluded),
            lookback_days,
            ",".join(sorted(excluded)),
        )
    return excluded


# ============================================================
# FINAL STOCK SCAN — 限售股解禁风控过滤 (用户定案 2026-08-03)
# 出名单时读 share_float 缓存, 剔除未来 N 天解禁比例累计超阈值的候选 (风控, 非 alpha)
# ============================================================
def share_float_upcoming_scan(
    symbols: list[str],
    ref_date,
    cache_path: str = SHARE_FLOAT_CACHE,
    window_days: int = UNLOCK_WINDOW_DAYS,
    ratio_threshold: float = UNLOCK_RATIO_THRESHOLD,
    stale_max_days: int = UNLOCK_STALE_MAX_DAYS,
) -> set[str]:
    """FINAL STOCK SCAN: 返回 symbols 中未来 window_days 天内解禁比例累计 > ratio_threshold 的子集.

    读 raw 缓存 (share_float_full.parquet, 限售股解禁日历), 剔除临近大额解禁的候选 —
    解禁是抛压前兆, 用户定案纳入 SCAN (2026-08-03). PIT-safe: 只认名单日前已公告
    (ann_date <= ref) 的解禁, 不 look-ahead. 缓存缺失/过期 → 返回空集 (fail-open):
    安全网不应因数据源抖动而清空当日名单.
    float_ratio 单位 = % (解禁股份占总股本比例, 已对 daily_basic total_share 验证).
    """
    if not symbols:
        return set()
    ref = pd.Timestamp(ref_date)
    try:
        sf = pd.read_parquet(
            cache_path, columns=["symbol", "ann_date", "float_date", "float_ratio"]
        )
    except Exception as exc:
        logger.warning("FINAL STOCK SCAN: 读解禁缓存失败, 跳过扫描: %s", exc)
        return set()
    if len(sf) == 0:
        return set()
    sf = sf.dropna(subset=["ann_date", "float_date"])

    def _parse_dates(col: pd.Series) -> pd.Series:
        if col.dtype != object:
            return col
        parsed = pd.to_datetime(col, format="%Y%m%d", errors="coerce")
        bad = parsed.isna() & col.notna()
        if bad.any():
            parsed.loc[bad] = pd.to_datetime(col.loc[bad], errors="coerce")
        return parsed

    sf["ann_date"] = _parse_dates(sf["ann_date"])
    sf["float_date"] = _parse_dates(sf["float_date"])
    sf = sf.dropna(subset=["ann_date", "float_date"])
    if sf["ann_date"].isna().all() or sf["float_date"].isna().all():
        logger.warning("FINAL STOCK SCAN: 解禁缓存日期无法解析, 跳过扫描")
        return set()
    sf = sf[sf["ann_date"] <= ref]  # PIT-safe: 只认名单日前已公告的解禁
    if len(sf) == 0:
        return set()
    if (ref - sf["ann_date"].max()).days > stale_max_days:
        logger.warning(
            "FINAL STOCK SCAN: 解禁缓存过期 (最新公告 %s vs 名单日 %s, 差 >%d 天), 跳过扫描",
            sf["ann_date"].max().date(),
            ref.date(),
            stale_max_days,
        )
        return set()
    sf["float_ratio"] = pd.to_numeric(sf["float_ratio"], errors="coerce")
    cutoff = ref + pd.Timedelta(days=window_days)
    win = sf[(sf["float_date"] > ref) & (sf["float_date"] <= cutoff)]
    if len(win) == 0:
        return set()
    accum = win.groupby("symbol")["float_ratio"].sum()
    excluded = {s for s in symbols if accum.get(s, 0.0) > ratio_threshold}
    if excluded:
        logger.warning(
            "FINAL STOCK SCAN: 剔除 %d 只未来 %d 天解禁比例累计 >%.1f%%: %s",
            len(excluded),
            window_days,
            ratio_threshold,
            ",".join(sorted(excluded)),
        )
    return excluded
