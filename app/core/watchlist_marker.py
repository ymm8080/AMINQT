# -*- coding: utf-8 -*-
"""自选标记规则 STEP2 (P10.14, ARCH §5.19, DESIGN_V1 §4 STEP2).

前置条件: 主力控盘比例 > 30%。
满足任一 CASE → 打"自选"标记:
  CASE1: 一个月内吸筹峰 + 从峰起益盟红线一直在蓝线之上
  CASE2: 两周内有单日涨幅 > 8%
  CASE3: 主力筹码控盘周均线本周 > 上周
  CASE4: 底部突破平台放量 + 前两周 >=3 天上下影线

未来函数禁止: 全部条件仅用截至当日的数据 (tail/shift/rolling)。
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

COL_CTRL_RATIO = "tech_ths_ctrl_ratio"
COL_PULLUP = "tech_ths_pullup_flag_decay10"
COL_TREND_SHORT = "tech_ths_trend_short"
COL_TREND_MID = "tech_ths_trend_mid"


class WatchlistMarker:
    """自选标记器 (Pipeline 1 盘后执行)."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (selection_config.yaml: watchlist_marker 段).

        Args:
            config: watchlist_marker 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}

    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    def check_prerequisite(self, df: pd.DataFrame) -> bool:
        """前置条件: tech_ths_ctrl_ratio > 0.30.

        Args:
            df: 日线数据 (含 tech_ths_ctrl_ratio)。

        Returns:
            True = 满足前置条件。
        """
        th = float(self._cfg("prerequisite", "ctrl_ratio_threshold", 0.30))
        if df is None or len(df) == 0 or COL_CTRL_RATIO not in df.columns:
            return False
        return bool(float(df[COL_CTRL_RATIO].iloc[-1]) > th)

    def check_case1_pullup_trend(self, df: pd.DataFrame) -> bool:
        """CASE1: 吸筹峰(30天) + 红线持续在蓝线之上.

        一个月内 pullup_flag_decay10 > 0 (峰值日 = 窗口内 flag 最大日),
        且自峰值日起 trend_short > trend_mid 全程成立。
        """
        lookback = int(self._cfg("case1_pullup_trend", "pullup_lookback_days",
                                 30))
        cols = [COL_PULLUP, COL_TREND_SHORT, COL_TREND_MID]
        if df is None or len(df) == 0 or not all(c in df.columns for c in cols):
            return False
        window = df.tail(lookback)
        pullup = window[COL_PULLUP].to_numpy(dtype=float)
        if not np.any(pullup > 0):
            return False
        peak_pos = int(np.argmax(pullup))
        short = window[COL_TREND_SHORT].to_numpy(dtype=float)[peak_pos:]
        mid = window[COL_TREND_MID].to_numpy(dtype=float)[peak_pos:]
        passed = bool(np.all(short > mid))
        if passed:
            logger.info("CASE1 命中: 吸筹峰后红线持续在蓝线之上")
        return passed

    def check_case2_short_surge(self, df: pd.DataFrame) -> bool:
        """CASE2: 两周内单日涨幅 > 8%.

        过去 10 个交易日内任一日 close.pct_change() > 0.08。
        """
        lookback = int(self._cfg("case2_short_surge", "lookback_days", 10))
        th = float(self._cfg("case2_short_surge", "surge_threshold", 0.08))
        if df is None or len(df) < 2 or "close" not in df.columns:
            return False
        ret = df["close"].astype(float).pct_change().tail(lookback)
        passed = bool((ret > th).any())
        if passed:
            logger.info("CASE2 命中: %d 日内存在涨幅 > %.0f%%", lookback,
                        th * 100)
        return passed

    def check_case3_weekly_ctrl_rising(self, df: pd.DataFrame) -> bool:
        """CASE3: 控盘周均线本周 > 上周.

        本周 = 最近 5 个交易日 ctrl_ratio 均值;
        上周 = 之前 5 个交易日均值。
        """
        window = int(self._cfg("case3_weekly_ctrl", "weekly_ma_window", 5))
        if df is None or len(df) < window * 2 or COL_CTRL_RATIO not in df.columns:
            return False
        ctrl = df[COL_CTRL_RATIO].astype(float)
        this_week = float(ctrl.iloc[-window:].mean())
        last_week = float(ctrl.iloc[-window * 2:-window].mean())
        passed = this_week > last_week
        if passed:
            logger.info("CASE3 命中: 本周控盘均值 %.3f > 上周 %.3f",
                        this_week, last_week)
        return passed

    def check_case4_breakout(self, df: pd.DataFrame) -> bool:
        """CASE4: 底部平台突破放量 + 前两周 >=3 天上下影线.

        条件 (AND):
          1. 平台突破: close[-1] > max(high[-N-1:-1]) * (1 + 2%)
             (N=20 平台窗口, 不含当日, 无未来函数)
          2. 放量: volume[-1] > 1.5 × MA5(volume)
          3. 前 10 个交易日内 >= 3 天同时存在上影线和下影线
        """
        platform_window = int(self._cfg("case4_breakout", "platform_window",
                                        20))
        breakout_th = float(self._cfg("case4_breakout", "breakout_threshold",
                                      0.02))
        vol_ratio = float(self._cfg("case4_breakout", "volume_surge_ratio",
                                    1.5))
        shadow_lookback = int(self._cfg("case4_breakout", "shadow_lookback_days",
                                        10))
        min_shadow_days = int(self._cfg("case4_breakout", "min_shadow_days",
                                        3))

        cols = ["open", "high", "low", "close", "volume"]
        if df is None or len(df) < platform_window + 1 or not all(
                c in df.columns for c in cols):
            return False

        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        open_ = df["open"].astype(float)
        volume = df["volume"].astype(float)

        # 1. 平台突破 (平台 = 前 N 根 high, 不含当日)
        platform_high = float(high.iloc[-(platform_window + 1):-1].max())
        breakout = float(close.iloc[-1]) > platform_high * (1.0 + breakout_th)

        # 2. 放量
        vol_ma5 = float(volume.rolling(5, min_periods=1).mean().iloc[-1])
        volume_surge = (float(volume.iloc[-1]) > vol_ma5 * vol_ratio
                        if vol_ma5 > 0 else False)

        # 3. 上下影线天数 (上影线 = high - max(open,close) > 0;
        #    下影线 = min(open,close) - low > 0; 同日皆有)
        recent = df.tail(shadow_lookback)
        upper = recent["high"].astype(float) - np.maximum(
            recent["open"].astype(float), recent["close"].astype(float))
        lower = np.minimum(
            recent["open"].astype(float),
            recent["close"].astype(float)) - recent["low"].astype(float)
        shadow_days = int(((upper > 0) & (lower > 0)).sum())
        shadows_ok = shadow_days >= min_shadow_days

        passed = breakout and volume_surge and shadows_ok
        if passed:
            logger.info("CASE4 命中: 突破平台 %.3f (+%.0f%%), 放量 %.1f×, "
                        "影线天数 %d", platform_high, breakout_th * 100,
                        volume.iloc[-1] / vol_ma5 if vol_ma5 > 0 else 0.0,
                        shadow_days)
        return passed

    def mark(self, pool: List[str],
             data: dict) -> List[str]:
        """对股票池批量执行标记.

        Args:
            pool: 股票池 symbol 列表。
            data: {symbol: 日线 DataFrame}。

        Returns:
            应打"自选"标记的 symbol 列表 (前置 + 任一 CASE)。
        """
        marked: List[str] = []
        for symbol in pool or []:
            df = (data or {}).get(symbol)
            if df is None:
                logger.warning("mark: %s 无日线数据, 跳过", symbol)
                continue
            if not self.check_prerequisite(df):
                continue
            if (self.check_case1_pullup_trend(df)
                    or self.check_case2_short_surge(df)
                    or self.check_case3_weekly_ctrl_rising(df)
                    or self.check_case4_breakout(df)):
                marked.append(symbol)
                logger.info("自选标记: %s", symbol)
        return marked
