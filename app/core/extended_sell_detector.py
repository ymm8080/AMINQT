# -*- coding: utf-8 -*-
"""扩展卖出检测器 (P10.15, ARCH §5.14.6, DESIGN_V1 §4 STEP4 + §5.4).

日线卖出扩展 (任一 → 标记日线卖出):
  CASE2: 获利盘筹码 < 40% (tech_ths_chip_profit_ratio < 0.40)
  CASE3: 日线 MACD dif 由正转负 (缺列时由 close 计算 12/26/9)
  (CASE1 收盘<4日前 在 dual_layer_sell_detector / P10.9)

日内卖出场景 (前置: 日线卖出已标记):
  C: 集合竞价涨 > 5% → 立刻全仓卖
  D: 日内涨 > 7% 量价背离 → 立刻卖
  E: 涨 > 7% + 放量 + 换手率 > 40% → 卖一半; 涨 > 15% → 全卖
  F: 涨 > 7% + 主力筹码 < 40% + 下午上涨 → 全卖
  G: 盈利 > 25% → 标记 + WARNING
  H: 换手率 > 40% + 涨 > 8% → 全卖
  (A 三峰下降 / B 急跌4% 在 P10.9)

未来函数禁止: MACD 用 ewm(adjust=False), 仅依赖历史数据。
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COL_CHIP_PROFIT = "tech_ths_chip_profit_ratio"
COL_DIF = "tech_ths_dif"


class ExtendedSellDetector:
    """扩展卖出检测器."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (trading_config.yaml: extended_sell 段).

        Args:
            config: extended_sell 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}

    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    @staticmethod
    def _compute_macd_dif(close: pd.Series) -> pd.Series:
        """MACD 12/26/9 dif 线 (ewm adjust=False, 无未来函数)."""
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        return ema12 - ema26

    def check_extended_daily_sell(self, df: pd.DataFrame) -> dict:
        """日线卖出扩展 CASE2/CASE3.

        Args:
            df: 日线数据, 含 close 及 tech_ths_chip_profit_ratio
                (可选 tech_ths_dif; 缺省时由 close 计算 MACD)。

        Returns:
            {is_daily_sell, case2_profit_chip, case3_macd_negative}
        """
        chip_th = float(self._cfg("daily_sell_extended", "chip_threshold",
                                  0.40))
        result = {"is_daily_sell": False, "case2_profit_chip": False,
                  "case3_macd_negative": False}
        if df is None or len(df) < 2:
            return result

        # CASE2: 获利盘筹码 < 40%
        if COL_CHIP_PROFIT in df.columns:
            chip_now = float(df[COL_CHIP_PROFIT].iloc[-1])
            case2 = chip_now < chip_th
            result["case2_profit_chip"] = case2
            if case2:
                logger.info("CASE2 触发: 获利盘 %.2f < %.2f", chip_now, chip_th)
        else:
            logger.warning("check_extended_daily_sell: 缺少 %s 列, CASE2 跳过",
                           COL_CHIP_PROFIT)

        # CASE3: MACD dif 由正转负
        if COL_DIF in df.columns:
            dif = df[COL_DIF].astype(float)
        elif "close" in df.columns:
            dif = self._compute_macd_dif(df["close"].astype(float))
        else:
            dif = None
        if dif is not None and len(dif) >= 2:
            dif_today, dif_yesterday = float(dif.iloc[-1]), float(dif.iloc[-2])
            case3 = dif_today < 0 <= dif_yesterday
            result["case3_macd_negative"] = case3
            if case3:
                logger.info("CASE3 触发: MACD dif %.4f 由正转负", dif_today)

        result["is_daily_sell"] = (result["case2_profit_chip"]
                                   or result["case3_macd_negative"])
        return result

    def check_auction_surge_sell(self, auction_pct: float) -> bool:
        """场景 C: 集合竞价涨 > 5% → 立刻全仓卖.

        Args:
            auction_pct: 集合竞价涨幅 (小数)。

        Returns:
            True = 立刻全仓卖出。
        """
        th = float(self._cfg("intraday_sell_extended", "auction_threshold",
                             0.05))
        triggered = float(auction_pct) > th
        if triggered:
            logger.info("场景C 触发: 竞价涨 %.2f%% > %.2f%% → 全仓卖",
                        auction_pct * 100, th * 100)
        return triggered

    def check_vol_price_divergence_sell(self, intraday_df: pd.DataFrame) -> bool:
        """场景 D: 涨 > 7% 但量能不明显, 高峰后价量齐降 → 立刻卖.

        判定:
          1. 盘中最高涨幅 (峰值 close / 参考价 - 1) >= surge_threshold
          2. 峰值处量能不明显: volume[peak] / MA5(volume)[peak] < vol_ratio_th
          3. 峰值已过 (非最后一根), 且当前 close < 峰值 close,
             当前 volume < 峰值 volume (高峰后价跌量缩)

        Args:
            intraday_df: 当日分钟 K 线, 含 close/volume;
                可选 pct_chg 列 (相对前收), 缺省时以首根 open 为参考。

        Returns:
            True = 量价背离卖出。
        """
        surge_th = float(self._cfg("intraday_sell_extended",
                                   "surge_threshold", 0.07))
        vol_ratio_th = float(self._cfg("intraday_sell_extended",
                                       "volume_surge_ratio", 1.2))
        if intraday_df is None or len(intraday_df) < 6:
            return False
        if "close" not in intraday_df.columns or "volume" not in intraday_df.columns:
            return False

        close = intraday_df["close"].astype(float)
        volume = intraday_df["volume"].astype(float)
        if "pct_chg" in intraday_df.columns:
            pct = intraday_df["pct_chg"].astype(float)
        else:
            ref = float(intraday_df["open"].iloc[0]) if "open" in intraday_df.columns \
                else float(close.iloc[0])
            pct = (close - ref) / ref if ref > 0 else pd.Series(0.0, index=close.index)

        peak_idx = int(np.argmax(close.to_numpy()))
        if peak_idx >= len(close) - 1:
            return False  # 峰值未确认 (右侧无 K 线)
        if float(pct.iloc[peak_idx]) < surge_th:
            return False

        vol_ma5 = volume.rolling(5, min_periods=1).mean()
        ma_at_peak = float(vol_ma5.iloc[peak_idx])
        vol_ratio = (float(volume.iloc[peak_idx]) / ma_at_peak
                     if ma_at_peak > 0 else 0.0)
        weak_volume = vol_ratio < vol_ratio_th

        price_falling = float(close.iloc[-1]) < float(close.iloc[peak_idx])
        volume_falling = float(volume.iloc[-1]) < float(volume.iloc[peak_idx])

        triggered = weak_volume and price_falling and volume_falling
        if triggered:
            logger.info("场景D 触发: 峰值涨幅 %.2f%%, 量比 %.2f, 峰后价量齐降",
                        float(pct.iloc[peak_idx]) * 100, vol_ratio)
        return triggered

    def check_turnover_partial_sell(self, pct_chg: float, turnover: float,
                                    volume_surge: bool) -> dict:
        """场景 E: 涨>7%+放量+换手>40% → 卖一半; 涨>15% → 全卖.

        Args:
            pct_chg: 日内涨幅 (小数)。
            turnover: 换手率 (小数)。
            volume_surge: 量能是否上涨 (放量)。

        Returns:
            {action: 'none'|'half'|'all'}
        """
        half_th = float(self._cfg("intraday_sell_extended",
                                  "surge_half_threshold", 0.07))
        all_th = float(self._cfg("intraday_sell_extended",
                                 "surge_all_threshold", 0.15))
        turn_th = float(self._cfg("intraday_sell_extended",
                                  "turnover_threshold", 0.40))

        action = "none"
        if pct_chg >= all_th:
            action = "all"
        elif pct_chg >= half_th and volume_surge and turnover > turn_th:
            action = "half"
        if action != "none":
            logger.info("场景E 触发: pct=%.2f%% turnover=%.2f%% surge=%s → %s",
                        pct_chg * 100, turnover * 100, volume_surge, action)
        return {"action": action}

    def check_chip_afternoon_sell(self, pct_chg: float, chip_ratio: float,
                                  is_afternoon: bool) -> bool:
        """场景 F: 涨>7% + 主力筹码<40% + 下午上涨 → 全卖.

        Args:
            pct_chg: 日内涨幅 (小数)。
            chip_ratio: 获利盘比例 (0~1)。
            is_afternoon: 上涨是否发生在下午 (12:00 后)。

        Returns:
            True = 全仓卖出。
        """
        surge_th = float(self._cfg("intraday_sell_extended",
                                   "surge_threshold", 0.07))
        chip_th = float(self._cfg("intraday_sell_extended",
                                  "chip_threshold", 0.40))
        triggered = (float(pct_chg) >= surge_th
                     and float(chip_ratio) < chip_th
                     and bool(is_afternoon))
        if triggered:
            logger.info("场景F 触发: pct=%.2f%% chip=%.2f 下午拉升 → 全仓卖",
                        pct_chg * 100, chip_ratio)
        return triggered

    def check_profit_warning(self, buy_price: float, current_price: float) -> dict:
        """场景 G: 盈利 > 25% → 标记 + WARNING.

        Args:
            buy_price: 买入价 (<=0 时安全返回不标记)。
            current_price: 当前价。

        Returns:
            {is_marked, warning_message}
        """
        profit_th = float(self._cfg("intraday_sell_extended",
                                    "profit_threshold", 0.25))
        if buy_price <= 0:
            return {"is_marked": False, "warning_message": None}
        profit = current_price / buy_price - 1.0
        is_marked = profit >= profit_th
        warning = None
        if is_marked:
            warning = (f"WARNING: 盈利 {profit * 100:.1f}% 超过 "
                       f"{profit_th * 100:.0f}%, 异常大涨请关注")
            logger.warning("场景G 触发: %s", warning)
        return {"is_marked": is_marked, "warning_message": warning}

    def check_high_turnover_surge_sell(self, pct_chg: float,
                                       turnover: float) -> bool:
        """场景 H: 换手率 > 40% 且涨 > 8% → 全卖.

        Args:
            pct_chg: 日内涨幅 (小数)。
            turnover: 换手率 (小数)。

        Returns:
            True = 全仓卖出。
        """
        turn_th = float(self._cfg("intraday_sell_extended",
                                  "turnover_threshold", 0.40))
        surge_th = float(self._cfg("intraday_sell_extended",
                                   "surge_threshold_h", 0.08))
        triggered = float(turnover) > turn_th and float(pct_chg) > surge_th
        if triggered:
            logger.info("场景H 触发: 换手 %.2f%% + 涨 %.2f%% → 全仓卖",
                        turnover * 100, pct_chg * 100)
        return triggered
