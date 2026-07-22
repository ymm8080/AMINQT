# -*- coding: utf-8 -*-
"""扩展买入检测器 (P10.15, ARCH §5.15.7, DESIGN_V1 §5.1/§5.3).

扩展买入规则:
  1. 跳空放量路径: 日线买点 + 开盘跳空 > 4% + 成交量放大 → 日内买点
  2. 换手率过滤: 换手率 > 50% 的票不选 (日线买点排除项)
  3. 不连续买入: 股票上涨时不得连续两次买入同一只 (可手工覆盖)
"""

import logging


logger = logging.getLogger(__name__)


class ExtendedBuyDetector:
    """扩展买入检测器."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (trading_config.yaml: extended_buy 段).

        Args:
            config: extended_buy 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}

    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    def check_gap_volume_buy(
        self,
        is_daily_buy: bool,
        open_price: float,
        prev_close: float,
        volume: float,
        volume_ma5: float,
    ) -> bool:
        """跳空放量日内买点 (与六条件路径 OR).

        Args:
            is_daily_buy: 日线买点已标记。
            open_price: 今日开盘价。
            prev_close: 昨日收盘价。
            volume: 当前成交量。
            volume_ma5: 5 日均量。

        Returns:
            True = 跳空 > 4% 且放量 → 设置日内买点。
        """
        gap_th = float(self._cfg("gap_volume_buy", "gap_threshold", 0.04))
        surge_ratio = float(self._cfg("gap_volume_buy", "volume_surge_ratio", 1.5))
        if not is_daily_buy:
            return False
        if prev_close <= 0 or volume_ma5 <= 0:
            logger.warning(
                "check_gap_volume_buy: 价格/均量非法 (prev_close=%s, volume_ma5=%s)",
                prev_close,
                volume_ma5,
            )
            return False
        gap_pct = open_price / prev_close - 1.0
        volume_ok = volume > volume_ma5 * surge_ratio
        triggered = gap_pct > gap_th and volume_ok
        if triggered:
            logger.info(
                "跳空放量买入: 跳空 %.2f%% > %.2f%%, 量 %.0f > %.1f × MA5",
                gap_pct * 100,
                gap_th * 100,
                volume,
                surge_ratio,
            )
        return triggered

    def filter_high_turnover(self, turnover: float, max_turnover: float = 0.50) -> bool:
        """换手率过滤: True=超过上限应排除 (默认 50%, 自适应可调).

        Args:
            turnover: 换手率 (小数)。
            max_turnover: 换手率上限。

        Returns:
            True = 换手率过高, 应排除 (不标记日线买点)。
        """
        max_turnover = float(self._cfg("turnover_filter", "max_turnover", max_turnover))
        excluded = float(turnover) > max_turnover
        if excluded:
            logger.info(
                "换手率过滤: %.2f%% > %.2f%% → 排除", turnover * 100, max_turnover * 100
            )
        return excluded

    def check_consecutive_buy_constraint(
        self, symbol: str, last_buy_symbol: str, price_rising: bool
    ) -> bool:
        """不连续买入约束.

        Args:
            symbol: 本次拟买入股票。
            last_buy_symbol: 上一次买入的股票。
            price_rising: 该股价格是否上涨中。

        Returns:
            True=允许买入; False=违反约束 (同股上涨中重复买入)。
        """
        if symbol == last_buy_symbol and bool(price_rising):
            logger.info("不连续买入约束: %s 上涨中重复买入 → 拦截", symbol)
            return False
        return True
