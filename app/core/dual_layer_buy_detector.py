# -*- coding: utf-8 -*-
"""双层买入信号检测 (P10.10, ARCH §5.15, DESIGN_V1 §4 STEP3 + §5.1/§5.2).

Layer 1 (日线, 盘后): 选股池 + 吸筹峰(30天) + 控盘>30% +
                     益盟红蓝线距离最小 → 标记日线买点
Layer 2 (日内, 盘中): 六条件 AND → 标记日内买点
                     (或: 跳空>4%+放量 路径, extended_buy_detector)
Layer 3 (开盘10分钟): 场景 A 下行→低峰后回升买 /
                     场景 B 上行→第二高峰>第一高峰后第二低峰上涨买 (互斥)

未来函数禁止: 吸筹峰/趋势判断只使用截至当日的数据。
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 因子列名 (以 ths_indicators.py 为准)
COL_PULLUP = "tech_ths_pullup_flag_decay10"
COL_CTRL_RATIO = "tech_ths_ctrl_ratio"
COL_TREND_SHORT = "tech_ths_trend_short"  # 红线
COL_TREND_MID = "tech_ths_trend_mid"  # 蓝线
COL_CTRL_LOW = "tech_ths_ctrl_low"  # 红柱 (0~100)
COL_FLOW_NET = "tech_ths_flow_net"


class DualLayerBuyDetector:
    """双层买入检测器."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (trading_config.yaml: dual_layer_buy 段).

        Args:
            config: dual_layer_buy 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}

    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    @staticmethod
    def _hhmm(t) -> str:
        """时间归一化为 'HH:MM' 字符串 (字符串比较即可判断先后)."""
        if isinstance(t, (pd.Timestamp,)):
            return t.strftime("%H:%M")
        if hasattr(t, "strftime"):
            return t.strftime("%H:%M")
        return str(t)[:5]

    @staticmethod
    def _find_local_peaks(values: np.ndarray) -> List[int]:
        """简单局部极大值 (需右侧一根确认, 无未来函数)."""
        peaks: List[int] = []
        for i in range(1, len(values) - 1):
            if values[i] >= values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def _find_local_troughs(values: np.ndarray) -> List[int]:
        """简单局部极小值 (需右侧一根确认, 无未来函数)."""
        troughs: List[int] = []
        for i in range(1, len(values) - 1):
            if values[i] <= values[i - 1] and values[i] < values[i + 1]:
                troughs.append(i)
        return troughs

    def check_daily_buy_point(
        self,
        symbol: str,
        selection_pool: list,
        daily_df: pd.DataFrame,
        pullup_lookback_days: int = 30,
        ctrl_ratio_threshold: float = 0.30,
    ) -> dict:
        """Layer 1: 日线买点检测 (四条件 AND, ARCH §5.15.2).

        条件:
          1. symbol ∈ selection_pool
          2. 过去 pullup_lookback_days 内 pullup_flag_decay10 > 0 (吸筹峰)
          3. 当前 ctrl_ratio > ctrl_ratio_threshold
          4. 从吸筹峰日期起 trend_short > trend_mid 全程成立,
             且当前 |short-mid| 为回看窗口内最小 (红蓝收敛)

        Returns:
            {is_daily_buy, condition_1_pool, condition_2_pullup_peak,
             condition_3_ctrl_ratio, condition_4_trend, pullup_peak_date,
             failed_conditions}
        """
        lookback = int(
            self._cfg("daily_buy", "pullup_lookback_days", pullup_lookback_days)
        )
        ctrl_th = float(
            self._cfg("daily_buy", "ctrl_ratio_threshold", ctrl_ratio_threshold)
        )

        result = {
            "is_daily_buy": False,
            "condition_1_pool": False,
            "condition_2_pullup_peak": False,
            "condition_3_ctrl_ratio": False,
            "condition_4_trend": False,
            "pullup_peak_date": None,
            "failed_conditions": [],
        }

        # 条件 1: 选股池
        cond1 = symbol in (selection_pool or [])
        result["condition_1_pool"] = cond1
        if not cond1:
            result["failed_conditions"].append("condition_1_pool")

        required_cols = [COL_PULLUP, COL_CTRL_RATIO, COL_TREND_SHORT, COL_TREND_MID]
        if (
            daily_df is None
            or len(daily_df) == 0
            or not all(c in daily_df.columns for c in required_cols)
        ):
            logger.warning("check_daily_buy_point(%s): 缺少 THS 因子列", symbol)
            result["failed_conditions"] += [
                "condition_2_pullup_peak",
                "condition_3_ctrl_ratio",
                "condition_4_trend",
            ]
            return result

        window = daily_df.tail(lookback)

        # 条件 2: 吸筹峰 (回看窗口内 flag > 0, 峰值日 = 窗口内 flag 最大日)
        pullup = window[COL_PULLUP].to_numpy(dtype=float)
        cond2 = bool(np.any(pullup > 0))
        result["condition_2_pullup_peak"] = cond2
        peak_pos: Optional[int] = None
        if cond2:
            peak_pos = int(np.argmax(pullup))
            result["pullup_peak_date"] = window.index[peak_pos]
        else:
            result["failed_conditions"].append("condition_2_pullup_peak")

        # 条件 3: 控盘比例
        ctrl_now = float(daily_df[COL_CTRL_RATIO].iloc[-1])
        cond3 = ctrl_now > ctrl_th
        result["condition_3_ctrl_ratio"] = cond3
        if not cond3:
            result["failed_conditions"].append("condition_3_ctrl_ratio")

        # 条件 4: 红线 > 蓝线 (自吸筹峰起全程) + 当前红蓝距离窗口最小
        cond4 = False
        if cond2 and peak_pos is not None:
            short = window[COL_TREND_SHORT].to_numpy(dtype=float)
            mid = window[COL_TREND_MID].to_numpy(dtype=float)
            since_peak_short = short[peak_pos:]
            since_peak_mid = mid[peak_pos:]
            red_above_blue = bool(np.all(since_peak_short > since_peak_mid))
            dist = np.abs(short - mid)
            dist_min = float(np.min(dist))
            current_is_min = bool(dist[-1] <= dist_min + 1e-12)
            cond4 = red_above_blue and current_is_min
        result["condition_4_trend"] = cond4
        if not cond4:
            result["failed_conditions"].append("condition_4_trend")

        result["is_daily_buy"] = cond1 and cond2 and cond3 and cond4
        if result["is_daily_buy"]:
            logger.info(
                "日线买点标记: %s (ctrl=%.2f, peak=%s)",
                symbol,
                ctrl_now,
                result["pullup_peak_date"],
            )
        return result

    def check_intraday_buy_point(
        self,
        is_daily_buy_marked: bool,
        current_time: str,
        advancing_stocks: int,
        total_stocks: int,
        flow_net: float,
        ctrl_ratio: float,
        ctrl_low_today: float,
        ctrl_low_yesterday: float,
        deadline: str = "10:40",
        breadth_threshold: float = 0.6,
        ctrl_ratio_threshold: float = 0.30,
        ctrl_low_threshold: float = 50.0,
    ) -> dict:
        """Layer 2: 日内买点检测 (六条件 AND, ARCH §5.15.3).

        条件 6: 红柱比前一天高 且 ctrl_low_today > ctrl_low_threshold
        (ctrl_low 量纲 0~100, 过半 = 50 绝对阈值, 以 ths_indicators.py 为准)。

        Returns:
            {is_intraday_buy, condition_1..6, failed_conditions}
        """
        deadline = self._cfg("intraday_buy", "deadline", deadline)
        breadth_th = float(
            self._cfg("intraday_buy", "breadth_threshold", breadth_threshold)
        )
        ctrl_th = float(
            self._cfg("intraday_buy", "ctrl_ratio_threshold", ctrl_ratio_threshold)
        )
        ctrl_low_th = float(
            self._cfg("intraday_buy", "ctrl_low_threshold", ctrl_low_threshold)
        )

        cond1 = bool(is_daily_buy_marked)
        cond2 = self._hhmm(current_time) <= self._hhmm(deadline)
        breadth = (advancing_stocks / total_stocks) if total_stocks > 0 else 0.0
        cond3 = breadth > breadth_th
        cond4 = float(flow_net) > 0.0
        cond5 = float(ctrl_ratio) > ctrl_th
        cond6 = (
            float(ctrl_low_today) > float(ctrl_low_yesterday)
            and float(ctrl_low_today) > ctrl_low_th
        )

        conds = {
            "condition_1_daily_mark": cond1,
            "condition_2_time": cond2,
            "condition_3_breadth": cond3,
            "condition_4_flow_net": cond4,
            "condition_5_ctrl_ratio": cond5,
            "condition_6_ctrl_low": cond6,
        }
        failed = [k for k, v in conds.items() if not v]
        is_buy = len(failed) == 0
        if is_buy:
            logger.info(
                "日内买点标记: breadth=%.2f flow=%.0f ctrl=%.2f ctrl_low=%.1f>%.1f",
                breadth,
                flow_net,
                ctrl_ratio,
                ctrl_low_today,
                ctrl_low_yesterday,
            )
        return {"is_intraday_buy": is_buy, **conds, "failed_conditions": failed}

    def check_opening_10min_buy_timing(
        self, intraday_df: pd.DataFrame, observation_minutes: int = 10
    ) -> dict:
        """Layer 3: 开盘 10 分钟走势 → 买入时机 (两场景互斥, ARCH §5.15.4).

        场景 A (开盘下行): 找到 10 分钟内最低点 → 当前价自低点回升 → 确认。
        场景 B (开盘上行): 第二峰 > 第一峰 → 第二峰后回落出第二低点 →
                          当前价自该低点回升 → 确认。

        Args:
            intraday_df: 开盘后 K 线 (含 open/close, 可选 high/low)。
            observation_minutes: 观察窗口 (分钟, 仅记录用)。

        Returns:
            {scenario: 'A'|'B'|None, buy_timing_confirmed, wait_reason}
        """
        result = {
            "scenario": None,
            "buy_timing_confirmed": False,
            "wait_reason": "数据不足",
        }
        if intraday_df is None or len(intraday_df) < 3:
            return result

        close = intraday_df["close"].to_numpy(dtype=float)
        low = (
            intraday_df["low"] if "low" in intraday_df.columns else intraday_df["close"]
        ).to_numpy(dtype=float)
        open_first = float(intraday_df["open"].iloc[0])
        last_idx = len(close) - 1
        direction_down = close[-1] < open_first

        if direction_down:
            # ── 场景 A: 开盘下行 → 低峰后回升 ──
            result["scenario"] = "A"
            trough_idx = int(np.argmin(low))
            if trough_idx >= last_idx:
                result["wait_reason"] = "最低点刚出现, 等待回升确认"
                return result
            if close[-1] > low[trough_idx]:
                result["buy_timing_confirmed"] = True
                result["wait_reason"] = None
                logger.info(
                    "开盘10min 场景A: 低点 %.3f → 回升至 %.3f, 买入",
                    low[trough_idx],
                    close[-1],
                )
            else:
                result["wait_reason"] = "尚未自低点回升"
            return result

        # ── 场景 B: 开盘上行 → 第二峰 > 第一峰 → 第二低点回升 ──
        result["scenario"] = "B"
        peak_idx = self._find_local_peaks(close)
        if len(peak_idx) < 2:
            result["wait_reason"] = "尚未形成两个峰值"
            return result
        p1, p2 = peak_idx[0], peak_idx[1]
        if not close[p2] > close[p1]:
            result["wait_reason"] = "第二峰未超过第一峰"
            return result
        # 第二峰之后的局部低点 (第二低峰)
        troughs_after_p2 = [t for t in self._find_local_troughs(close) if t > p2]
        if not troughs_after_p2:
            result["wait_reason"] = "第二峰后尚未形成低点"
            return result
        trough_idx = troughs_after_p2[0]
        if trough_idx >= last_idx:
            result["wait_reason"] = "第二低点刚出现, 等待回升确认"
            return result
        if close[-1] > low[trough_idx]:
            result["buy_timing_confirmed"] = True
            result["wait_reason"] = None
            logger.info(
                "开盘10min 场景B: 峰2 %.3f > 峰1 %.3f, 低点 %.3f 回升至 %.3f, 买入",
                close[p2],
                close[p1],
                low[trough_idx],
                close[-1],
            )
        else:
            result["wait_reason"] = "尚未自第二低点回升"
        return result

    def detect(
        self,
        symbol: str,
        selection_pool: list,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        market_context: dict = None,
    ) -> dict:
        """三层综合检测: 全部确认 → final_buy_signal.

        Args:
            symbol: 股票代码。
            selection_pool: Pipeline 1 选股池。
            daily_df: 日线数据 (含 THS 因子列)。
            intraday_df: 当日分钟 K 线。
            market_context: 可选 {current_time, advancing_stocks,
                total_stocks}; 缺省时时间取 intraday 末根, 广度视为满足
                (记 WARNING, 由上层 Pipeline 注入真实广度)。

        Returns:
            {final_buy_signal, failed_layer, layer1, layer2, layer3}
        """
        ctx = market_context or {}

        # ── Layer 1: 日线买点 ──
        layer1 = self.check_daily_buy_point(symbol, selection_pool, daily_df)
        if not layer1["is_daily_buy"]:
            return {
                "final_buy_signal": False,
                "failed_layer": 1,
                "layer1": layer1,
                "layer2": None,
                "layer3": None,
            }

        # ── Layer 2: 日内买点 ──
        if intraday_df is not None and len(intraday_df) > 0:
            last_bar_time = intraday_df.index[-1]
            if "datetime" in intraday_df.columns:
                last_bar_time = intraday_df["datetime"].iloc[-1]
        else:
            last_bar_time = "15:00"
        current_time = ctx.get("current_time", self._hhmm(last_bar_time))
        advancing = ctx.get("advancing_stocks")
        total = ctx.get("total_stocks")
        if advancing is None or total is None:
            logger.warning("detect(%s): 未提供市场广度, 视为满足 (仅测试用)", symbol)
            advancing, total = 1, 1  # breadth = 1.0
        flow_net = (
            float(daily_df[COL_FLOW_NET].iloc[-1])
            if COL_FLOW_NET in daily_df.columns
            else 0.0
        )
        ctrl_ratio = float(daily_df[COL_CTRL_RATIO].iloc[-1])
        ctrl_low_today = (
            float(daily_df[COL_CTRL_LOW].iloc[-1])
            if COL_CTRL_LOW in daily_df.columns
            else 0.0
        )
        ctrl_low_yesterday = (
            float(daily_df[COL_CTRL_LOW].iloc[-2])
            if COL_CTRL_LOW in daily_df.columns and len(daily_df) >= 2
            else 0.0
        )

        layer2 = self.check_intraday_buy_point(
            True,
            current_time,
            advancing,
            total,
            flow_net,
            ctrl_ratio,
            ctrl_low_today,
            ctrl_low_yesterday,
        )
        if not layer2["is_intraday_buy"]:
            return {
                "final_buy_signal": False,
                "failed_layer": 2,
                "layer1": layer1,
                "layer2": layer2,
                "layer3": None,
            }

        # ── Layer 3: 开盘10分钟时机 ──
        layer3 = self.check_opening_10min_buy_timing(intraday_df)
        final = bool(layer3["buy_timing_confirmed"])
        return {
            "final_buy_signal": final,
            "failed_layer": None if final else 3,
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
        }
