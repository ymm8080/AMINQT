# -*- coding: utf-8 -*-
"""
峰值跟踪器 — 右侧确认, 消除未来函数 (DESIGN §15.5.1, rule_engine_v2 §4)
============================================================================
峰/谷值需回落(回升) >= peak_confirm_drop 且持续 peak_confirm_bars 根 bar 才确认.
盘中信号全部因果: 确认时刻 <= 当前 bar, 无未来信息.
"""

from __future__ import annotations

from typing import Optional

from .config import Config


class PeakTracker:
    """2分钟 bar 峰/谷右侧确认."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.bars: list[float] = []
        self.confirmed_peaks: list[float] = []
        self.confirmed_troughs: list[float] = []
        self._candidate_high: Optional[float] = None
        self._drop_bars: int = 0
        self._candidate_low: Optional[float] = None
        self._rise_bars: int = 0

    def update(self, price: float) -> None:
        """喂入一根 bar 的价格. 确认只依赖历史 bar, 无未来函数."""
        self.bars.append(price)
        # ---- 峰值确认: 从候选高点回落 >= drop 且持续 N 根 ----
        if self._candidate_high is None or price > self._candidate_high:
            self._candidate_high, self._drop_bars = price, 0
        elif price <= self._candidate_high * (1 - self.cfg.peak_confirm_drop):
            self._drop_bars += 1
            if self._drop_bars >= self.cfg.peak_confirm_bars:
                if (
                    not self.confirmed_peaks
                    or self.confirmed_peaks[-1] != self._candidate_high
                ):
                    self.confirmed_peaks.append(self._candidate_high)
                self._candidate_high, self._drop_bars = None, 0
        # ---- 谷值确认: 从候选低点回升 >= drop 且持续 N 根 ----
        if self._candidate_low is None or price < self._candidate_low:
            self._candidate_low, self._rise_bars = price, 0
        elif price >= self._candidate_low * (1 + self.cfg.peak_confirm_drop):
            self._rise_bars += 1
            if self._rise_bars >= self.cfg.peak_confirm_bars:
                if (
                    not self.confirmed_troughs
                    or self.confirmed_troughs[-1] != self._candidate_low
                ):
                    self.confirmed_troughs.append(self._candidate_low)
                self._candidate_low, self._rise_bars = None, 0

    def three_peaks_descending(self) -> bool:
        """三个已确认峰值连续下降 (P8 卖出条件)."""
        p = self.confirmed_peaks
        return len(p) >= 3 and p[-1] < p[-2] < p[-3]
