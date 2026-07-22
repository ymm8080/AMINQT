# -*- coding: utf-8 -*-
"""第一步: 基础流动性/活跃度过滤 (P6, DESIGN_V1 §4 STEP1 第一步).

全市场 → 基础池。保留流动性好、活跃度高的股票; 剔除风险股。
阈值不写死 — 从 adaptive_config.yaml 读取 (边界+初始值, AdaptiveEngine 可调)。
"""

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 涨停判定阈值: 主板 9.8% (创业板/科创板 19.8%, 简化统一用 9.8%)
LIMIT_UP_PCT = 0.098
# 年化交易日数 (用于波动率年化)
TRADING_DAYS_PER_YEAR = 244


def _resolve(value: Any, default: float) -> float:
    """解析配置值: 支持标量或 {initial: x, bounds: [...]} 结构.

    Args:
        value: 配置原始值。
        default: 缺省默认值。

    Returns:
        标量阈值。
    """
    if isinstance(value, dict):
        return float(value.get("initial", default))
    if value is None:
        return float(default)
    return float(value)


class BaseLiquidityFilter:
    """基础流动性/活跃度过滤器.

    保留条件 (AND):
        1. 近一年日均换手率 > 5%
        2. 近一年日均成交额 > 5 亿
        3. 年平均振幅 > 5%
        4. 近一年涨停次数 > 0

    剔除条件 (任一):
        - ST / 退市整理股
        - 重大风险公告个股
        - 连续亏损个股
        - 大市值白马低波动股
    """

    def __init__(self, config: dict = None) -> None:
        """加载阈值配置 (adaptive_config.yaml: base_filter 段).

        Args:
            config: 配置字典, 含 min_turnover/min_amount/min_amplitude/
                    min_limit_up_count 等阈值 (初始值, 可被 AdaptiveEngine 调整)。
                    支持标量或 {initial, bounds} 结构。
        """
        self.config = config or {}
        self.min_turnover = _resolve(self.config.get("min_turnover"), 0.05)
        self.min_amount = _resolve(self.config.get("min_amount"), 500_000_000.0)
        self.min_amplitude = _resolve(self.config.get("min_amplitude"), 0.05)
        self.min_limit_up_count = int(
            _resolve(self.config.get("min_limit_up_count"), 1)
        )
        # 大市值白马低波动剔除阈值 (数据列存在时才启用)
        self.large_mktcap_threshold = _resolve(
            self.config.get("large_mktcap_threshold"),
            100_000_000_000.0,  # 1000 亿
        )
        self.low_volatility_threshold = _resolve(
            self.config.get("low_volatility_threshold"),
            0.20,  # 年化波动率
        )
        logger.info(
            "BaseLiquidityFilter 初始化, min_turnover=%.3f min_amount=%.0f "
            "min_amplitude=%.3f min_limit_up_count=%d",
            self.min_turnover,
            self.min_amount,
            self.min_amplitude,
            self.min_limit_up_count,
        )

    # ── 公共入口 ────────────────────────────────────────────────────

    def apply(self, all_stocks: Dict[str, pd.DataFrame]) -> List[str]:
        """全市场 → 基础池.

        Args:
            all_stocks: {symbol: 日线 DataFrame} (需含近一年数据)。

        Returns:
            通过过滤的 symbol 列表。
        """
        passed: List[str] = []
        for symbol, df in all_stocks.items():
            if df is None or df.empty:
                logger.debug("%s: 空数据, 剔除", symbol)
                continue
            try:
                if self.check_exclusions(symbol, df):
                    logger.debug("%s: 命中剔除条件", symbol)
                    continue
                if self.check_liquidity(df):
                    passed.append(symbol)
            except Exception:  # noqa: BLE001 — 单股异常不阻断全市场过滤
                logger.exception("%s: 过滤异常, 剔除", symbol)
        logger.info("基础过滤完成: %d/%d 只通过", len(passed), len(all_stocks))
        return passed

    # ── 保留条件 ────────────────────────────────────────────────────

    def check_liquidity(self, df: pd.DataFrame) -> bool:
        """流动性/活跃度四条件 AND 检查.

        Args:
            df: 单只股票近一年日线 (turnover/amount/high/low/close 列)。

        Returns:
            True=满足全部四条件。
        """
        if df is None or len(df) < 20:
            return False

        # 1. 近一年日均换手率 > min_turnover
        avg_turnover = self._avg_turnover(df)
        if avg_turnover is None or avg_turnover <= self.min_turnover:
            return False

        # 2. 近一年日均成交额 > min_amount
        avg_amount = self._avg_amount(df)
        if avg_amount is None or avg_amount <= self.min_amount:
            return False

        # 3. 年平均振幅 > min_amplitude
        avg_amplitude = self._avg_amplitude(df)
        if avg_amplitude is None or avg_amplitude <= self.min_amplitude:
            return False

        # 4. 近一年涨停次数 >= min_limit_up_count (默认 1, 即 > 0)
        limit_up_count = self._limit_up_count(df)
        if limit_up_count < self.min_limit_up_count:
            return False

        return True

    # ── 剔除条件 ────────────────────────────────────────────────────

    def check_exclusions(self, symbol: str, df: pd.DataFrame) -> bool:
        """剔除条件检查.

        Args:
            symbol: 股票代码。
            df: 日线数据。

        Returns:
            True=应剔除 (命中任一剔除条件)。
        """
        if df is None or df.empty:
            return True

        # 1. ST / 退市整理股 (name 列存在时按名称判断)
        if self._is_st_or_delisting(symbol, df):
            logger.debug("%s: ST/退市, 剔除", symbol)
            return True

        # 2. 重大风险公告 (ann_risk_warning_flag 列存在时)
        if "ann_risk_warning_flag" in df.columns:
            flag = float(np.nan_to_num(df["ann_risk_warning_flag"].iloc[-1]))
            if flag > 0:
                logger.debug("%s: 重大风险公告, 剔除", symbol)
                return True

        # 3. 连续亏损 (net_profit 列存在时: 最近两期均 < 0)
        if "net_profit" in df.columns:
            profits = df["net_profit"].dropna()
            if len(profits) >= 2 and (profits.iloc[-2:] < 0).all():
                logger.debug("%s: 连续亏损, 剔除", symbol)
                return True

        # 4. 大市值白马低波动 (mktcap 列存在时)
        if self._is_large_cap_low_vol(df):
            logger.debug("%s: 大市值低波动白马, 剔除", symbol)
            return True

        return False

    # ── 内部计算 (全部 trailing rolling, 无未来函数) ────────────────

    @staticmethod
    def _avg_turnover(df: pd.DataFrame) -> float:
        """日均换手率 (自动识别 0~1 / 0~100 两种量纲)."""
        col = next((c for c in ("turnover", "turnover_rate") if c in df.columns), None)
        if col is None:
            return None
        avg = float(np.nan_to_num(df[col].mean()))
        if avg > 1.5:  # 百分数量纲 → 转小数
            avg /= 100.0
        return avg

    @staticmethod
    def _avg_amount(df: pd.DataFrame) -> float:
        """日均成交额 (元); amount 列缺失时用 close*volume 近似."""
        if "amount" in df.columns:
            return float(np.nan_to_num(df["amount"].mean()))
        if {"close", "volume"} <= set(df.columns):
            amount = df["close"] * df["volume"]
            return float(np.nan_to_num(amount.mean()))
        return None

    @staticmethod
    def _avg_amplitude(df: pd.DataFrame) -> float:
        """年平均振幅 = mean((high-low)/prev_close); amplitude 列优先."""
        if "amplitude" in df.columns:
            avg = float(np.nan_to_num(df["amplitude"].mean()))
            if avg > 1.5:  # 百分数量纲 → 转小数
                avg /= 100.0
            return avg
        if {"high", "low", "close"} <= set(df.columns):
            prev_close = df["close"].shift(1)
            amplitude = (df["high"] - df["low"]) / prev_close.replace(0, np.nan)
            return float(np.nan_to_num(amplitude.mean()))
        return None

    @staticmethod
    def _limit_up_count(df: pd.DataFrame) -> int:
        """近一年涨停次数.

        判定: 日涨幅 >= 9.8% (主板), 或 close == round(prev_close*1.1, 2)。
        仅用 shift(1) 历史数据, 无未来函数。
        """
        if "close" not in df.columns:
            return 0
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        pct_chg = close / prev_close.replace(0, np.nan) - 1.0
        by_pct = pct_chg >= LIMIT_UP_PCT
        by_price = close == (prev_close * 1.1).round(2)
        return int(np.nansum((by_pct | by_price).fillna(False)))

    @staticmethod
    def _is_st_or_delisting(symbol: str, df: pd.DataFrame) -> bool:
        """ST/退市整理股判定 (name 列存在时按名称, 否则按代码前缀)."""
        name = None
        if "name" in df.columns and len(df) > 0:
            name = str(df["name"].iloc[-1])
        if name:
            upper = name.upper()
            if "ST" in upper or "退" in name:
                return True
        # 代码兜底: 退市整理期常见前缀不适用于 symbol, 仅名称可靠
        return False

    def _is_large_cap_low_vol(self, df: pd.DataFrame) -> bool:
        """大市值白马低波动: 市值大 且 年化波动率低."""
        col = next(
            (c for c in ("mktcap", "total_mv", "market_cap") if c in df.columns),
            None,
        )
        if col is None or "close" not in df.columns:
            return False
        mktcap = float(np.nan_to_num(df[col].iloc[-1]))
        if mktcap <= self.large_mktcap_threshold:
            return False
        ret = df["close"].pct_change().dropna()
        if len(ret) < 20:
            return False
        annual_vol = float(ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        return annual_vol < self.low_volatility_threshold
