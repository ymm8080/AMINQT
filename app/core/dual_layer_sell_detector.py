# -*- coding: utf-8 -*-
"""双层卖出信号检测 (P10.9, ARCH §5.14).

Layer 1 (日线, 盘后): 今日收盘价 < 4 日前收盘价 → 标记日线卖出 (准备信号)
Layer 2 (日内, 盘中): 场景 A 三峰连续下降 (第三峰后卖) /
                     场景 B 日内急跌 >= 4% (立刻卖, 优先级 > A)

未来函数禁止: 仅使用截至当前的数据 (两点比较 / 已确认局部峰值)。
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DualLayerSellDetector:
    """双层卖出检测器."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (trading_config.yaml: dual_layer_sell 段).

        Args:
            config: dual_layer_sell 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}

    # ── 配置读取辅助 ────────────────────────────────────────────────
    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    @staticmethod
    def _find_local_peaks(values: np.ndarray) -> List[int]:
        """简单局部极大值: v[i] >= v[i-1] 且 v[i] > v[i+1] (严格大于右侧).

        只使用 i 及之前的数据确认峰值 (需右侧一根 K 线确认, 无未来函数:
        确认发生在 i+1 时刻, 不引用更晚数据)。
        """
        peaks: List[int] = []
        for i in range(1, len(values) - 1):
            if values[i] >= values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    def check_daily_close_break(self, df: pd.DataFrame) -> bool:
        """Layer 1: close[today] < close[t-4] (两点比较, 非连续递减).

        Args:
            df: 日线数据 (至少 lookback_days+1 根, 含 close 列)。

        Returns:
            True = 标记日线卖出。
        """
        lookback = int(self._cfg("daily_close_break", "lookback_days", 4))
        if df is None or len(df) < lookback + 1 or "close" not in df.columns:
            logger.warning("check_daily_close_break: 数据不足 (%s 根, 需 %d)",
                           0 if df is None else len(df), lookback + 1)
            return False
        close_today = float(df["close"].iloc[-1])
        close_n_ago = float(df["close"].iloc[-(lookback + 1)])
        is_break = close_today < close_n_ago
        if is_break:
            logger.info("日线收盘跌破: close=%.3f < %d日前 close=%.3f",
                        close_today, lookback, close_n_ago)
        return is_break

    def check_three_peaks_decline(self, intraday_df: pd.DataFrame) -> dict:
        """场景 A: 当日三个峰值连续下降 → 第三个峰值后卖出.

        Args:
            intraday_df: 当日 5min K 线, 含 close 列 (有 high 列则用 high)。

        Returns:
            {is_signal, peaks: [...], sell_after_peak: 3}
        """
        result = {"is_signal": False, "peaks": [], "sell_after_peak": 3}
        if intraday_df is None or len(intraday_df) < 5:
            return result
        price_col = "high" if "high" in intraday_df.columns else "close"
        values = intraday_df[price_col].to_numpy(dtype=float)
        peak_idx = self._find_local_peaks(values)
        peaks = [float(values[i]) for i in peak_idx]
        result["peaks"] = peaks

        # 扫描峰值序列中任意 3 个连续严格递减的峰
        for k in range(len(peaks) - 2):
            p1, p2, p3 = peaks[k], peaks[k + 1], peaks[k + 2]
            if p3 < p2 < p1:
                # 第 3 峰需已被后续 K 线确认 (第3峰索引 < 最后一根)
                third_peak_bar = peak_idx[k + 2]
                if third_peak_bar < len(values) - 1:
                    result["is_signal"] = True
                    logger.info("三峰连续下降: %.3f > %.3f > %.3f → 卖出", p1, p2, p3)
                    break
        return result

    def check_intraday_crash(self, pct_chg: float,
                             threshold: float = -0.04) -> bool:
        """场景 B: 日内急跌 >= 4% → 立刻卖 (优先级高于场景 A).

        Args:
            pct_chg: 日内涨跌幅 (小数, 如 -0.045)。
            threshold: 急跌阈值 (负值, 默认 -0.04)。

        Returns:
            True = 急跌触发, 立刻卖出。
        """
        cfg_threshold = self._cfg("intraday_sell", "threshold", None)
        if cfg_threshold is not None:
            threshold = -abs(float(cfg_threshold))
        triggered = float(pct_chg) <= threshold
        if triggered:
            logger.info("日内急跌触发: pct=%.2f%% <= %.2f%%",
                        pct_chg * 100, threshold * 100)
        return triggered

    def detect(self, daily_df: pd.DataFrame,
               intraday_df: pd.DataFrame) -> dict:
        """双层综合检测.

        场景 B (急跌) 优先级高于场景 A (三峰): 先查急跌, 触发即立刻卖。

        Args:
            daily_df: 日线数据 (含 close)。
            intraday_df: 当日分钟 K 线 (含 open/close, 可选 high)。

        Returns:
            {daily_sell_marked, intraday_signal, scenario, action}
            scenario ∈ {None, 'A_three_peaks', 'B_crash'};
            action ∈ {'hold', 'sell'}。
        """
        result = {
            "daily_sell_marked": False,
            "intraday_signal": False,
            "scenario": None,
            "action": "hold",
        }
        daily_marked = self.check_daily_close_break(daily_df)
        result["daily_sell_marked"] = daily_marked
        if not daily_marked:
            return result

        # 日内涨跌幅: 相对前收 (日线倒数第2根), 无前收则用当日开盘
        reference = self._cfg("intraday_sell", "reference", "prev_close")
        if intraday_df is None or len(intraday_df) == 0:
            return result
        current_price = float(intraday_df["close"].iloc[-1])
        if reference == "open" or len(daily_df) < 2:
            ref_price = float(intraday_df["open"].iloc[0])
        else:
            ref_price = float(daily_df["close"].iloc[-2])
        pct_chg = (current_price - ref_price) / ref_price if ref_price > 0 else 0.0

        # 场景 B 优先
        if self.check_intraday_crash(pct_chg):
            result.update(intraday_signal=True, scenario="B_crash", action="sell")
            return result

        # 场景 A
        peaks_result = self.check_three_peaks_decline(intraday_df)
        if peaks_result["is_signal"]:
            result.update(intraday_signal=True, scenario="A_three_peaks",
                          action="sell")
        return result
