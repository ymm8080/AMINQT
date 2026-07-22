# -*- coding: utf-8 -*-
"""
CompositeFeed — IndicatorFeed 协议的生产组合 (P16, DESIGN §8.3, 架构 §15.2)
================================================================================
聚合 4 个复刻指标源 + 资金流双路径, 注入 RuleEngine(feed=...).
子源缺失 → 保守默认值 + 告警 (不崩溃).

接口映射 (app/rules/rule_engine.py IndicatorFeed 协议):
  control_ratio                    ← CapitalFeed (C路径同花顺手工 > B路径东财代理)
  had_accumulation_peak            ← zhuli_lasheng (吸筹峰, 仅盘后!)
  red_above_blue_since_peak        ← YimengFeed (自最近底部区域起红在蓝上)
  red_blue_distance_min            ← YimengFeed (红蓝距离 N 日最小)
  control_weekly_up                ← chip/control 周均线环比 (行情数据计算)
  bottom_breakout_volume           ← 底部平台突破放量 (行情数据计算)
  recent_shadow_lines              ← 两周内连续上下影线 (行情数据计算)
  red_bar_rising_and_majority      ← ChipFeed (A04 红柱升且 >50)
  profit_chip_ratio                ← ChipFeed (获利盘 %)
  latest_prob_up                   ← 外部注入的最新 V3.5 清单概率 (P12 用)
"""

from __future__ import annotations

import logging

import pandas as pd

from .capital_feed import CapitalFeed
from .chip_distribution import ChipFeed
from .yimeng_dingdi import YimengFeed
from .zhuli_lasheng import had_accumulation_peak as _had_peak

logger = logging.getLogger(__name__)


class CompositeFeed:
    """生产组合 Provider.

    Args:
        yimeng:   YimengFeed (红蓝线) — None 则红蓝接口返回保守 False
        chip:     ChipFeed (控盘红柱/获利盘) — None 则返回保守默认
        capital:  CapitalFeed (控盘比例/大单净量) — None 则 control_ratio=0
        zhuli_hist: {code: zhuli_lasheng() 后的 DataFrame} (吸筹峰判定用, 盘后计算)
        daily_hist: {code: 日线 DataFrame} (周线环比/突破/影线计算用)
        prob_provider: callable(code) -> float, 最新 V3.5 清单 prob_up (P12)
    """

    def __init__(self, yimeng: YimengFeed | None = None,
                 chip: ChipFeed | None = None,
                 capital: CapitalFeed | None = None,
                 zhuli_hist: dict | None = None,
                 daily_hist: dict | None = None,
                 prob_provider=None):
        self.yimeng = yimeng
        self.chip = chip
        self.capital = capital
        self.zhuli_hist = zhuli_hist or {}
        self.daily_hist = daily_hist or {}
        self._prob_provider = prob_provider

    # ---------------- 控盘比例 / 大单净量 ----------------
    def control_ratio(self, code: str) -> float:
        if self.capital is None:
            logger.warning("CapitalFeed 缺失, control_ratio 返回 0 (保守)")
            return 0.0
        return float(self.capital.control_ratio(code)["value"])

    def big_order_net(self, code: str) -> float:
        if self.capital is None:
            return 0.0
        return float(self.capital.big_order_net(code)["value"])

    # ---------------- 吸筹峰 (盘后专用) ----------------
    def had_accumulation_peak(self, code: str, lookback: int) -> bool:
        df = self.zhuli_hist.get(code)
        if df is None:
            logger.warning("zhuli_hist 缺 %s, had_accumulation_peak 返回 False", code)
            return False
        return _had_peak(df, lookback)

    # ---------------- 益盟红蓝线 ----------------
    def red_above_blue_since_peak(self, code: str) -> bool:
        if self.yimeng is None:
            return False
        return self.yimeng.red_above_blue_since_peak(code)

    def red_blue_distance_min(self, code: str) -> bool:
        if self.yimeng is None:
            return False
        return self.yimeng.red_blue_distance_min(code)

    # ---------------- 控盘红柱 / 获利盘 ----------------
    def red_bar_rising_and_majority(self, code: str) -> bool:
        if self.chip is None:
            return False
        return self.chip.red_bar_rising_and_majority(code)

    def profit_chip_ratio(self, code: str) -> float:
        if self.chip is None:
            return 100.0    # 保守: 不触发低筹码卖出
        return self.chip.profit_chip_ratio(code)

    def control_signal_A0A(self, code: str) -> bool:
        if self.chip is None:
            return False
        return self.chip.control_signal_A0A(code)

    # ---------------- 行情数据计算 (控盘周线/突破/影线) ----------------
    def control_weekly_up(self, code: str) -> bool:
        """主力控盘周均线本周 > 上周. 无周级控盘数据时用 ChipFeed A04 5日均环比近似."""
        if self.chip is None or code not in self.chip.hist:
            return False
        a04 = self.chip.hist[code]["A04"]
        if len(a04) < 10:
            return False
        return bool(a04.tail(5).mean() > a04.iloc[-10:-5].mean())

    def bottom_breakout_volume(self, code: str) -> bool:
        """底部突破平台伴成交量放大: 收盘创20日新高 且 量 > 5日均量×1.5 且 近60日处于低位区."""
        df = self.daily_hist.get(code)
        if df is None or len(df) < 60:
            return False
        c, v = df["close"], df["volume"]
        low_zone = c.iloc[-1] < c.rolling(60).max().iloc[-1] * 1.15  # 距60日高点<15%
        breakout = c.iloc[-1] >= c.rolling(20).max().iloc[-1]
        vol_up = v.iloc[-1] > v.rolling(5).mean().iloc[-1] * 1.5
        return bool(low_zone and breakout and vol_up)

    def recent_shadow_lines(self, code: str) -> bool:
        """两周 (10交易日) 内有连续上影线下影线 (>=2 天双向影线均 > 实体)."""
        df = self.daily_hist.get(code)
        if df is None or len(df) < 10:
            return False
        g = df.tail(10)
        body = (g["close"] - g["open"]).abs()
        upper = g["high"] - g[["close", "open"]].max(axis=1)
        lower = g[["close", "open"]].min(axis=1) - g["low"]
        both_shadow = (upper > body) & (lower > body)
        return bool(both_shadow.sum() >= 2)

    # ---------------- P12 概率源 ----------------
    def latest_prob_up(self, code: str) -> float:
        """最新 V3.5 清单 prob_up. 未注入 provider → 1.0 (保守: 不触发概率衰减)."""
        if self._prob_provider is None:
            return 1.0
        return float(self._prob_provider(code))
