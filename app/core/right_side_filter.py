# -*- coding: utf-8 -*-
"""右侧交易预筛选器 (P10.8b, ARCH §5.13.7.B, DESIGN_V1 §9 #2).

右侧交易 (Right-Side Trading): 只在股票确认上升趋势后参与。
独立组件, 供 AdaptiveWeighter / SelectionPipeline / WatchlistMarker 复用。
均线周期由回测选优, 从配置读取, 不写死。
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 20 日趋势斜率窗口 (ARCH §5.13.7.B 条件 4)
SLOPE_WINDOW = 20
# 5 日收益率窗口 (ARCH §5.13.7.B 条件 3)
RETURN_WINDOW = 5


class RightSideFilter:
    """右侧交易预筛选器 — 选股 Pipeline 前置过滤.

    判断条件 (全部满足 → 上行股票):
        1. close > MA(ma_long)
        2. MA(ma_short) > MA(ma_mid) > MA(ma_long)  (均线多头排列)
        3. 5 日收益率 > 0
        4. 20 日线性回归斜率 > 0
        5. (可选) 大盘在 MA20 之上
        6. 成交额 > 5000 万
    """

    def __init__(self, ma_short: int = 5, ma_mid: int = 10, ma_long: int = 20,
                 min_amount: float = 50_000_000,
                 require_market_above_ma20: bool = False) -> None:
        """初始化预筛选参数.

        Args:
            ma_short: 短均线周期 (回测选优, 默认 5)。
            ma_mid: 中均线周期 (默认 10)。
            ma_long: 长均线周期 (默认 20, 候选 15)。
            min_amount: 最低成交额 (默认 5000 万)。
            require_market_above_ma20: 是否要求大盘在 MA20 之上。
        """
        self.ma_short = ma_short
        self.ma_mid = ma_mid
        self.ma_long = ma_long
        self.min_amount = min_amount
        self.require_market_above_ma20 = require_market_above_ma20

    def is_uptrend(self, stock_df: pd.DataFrame,
                   market_above_ma20: bool = True) -> bool:
        """单只股票右侧判断.

        Args:
            stock_df: 日线 DataFrame (close/amount 列)。
            market_above_ma20: 大盘是否在 MA20 之上。

        Returns:
            True=上行 (参与权重计算), False=非上行 (跳过)。
        """
        min_rows = max(self.ma_long, SLOPE_WINDOW, RETURN_WINDOW + 1)
        if stock_df is None or len(stock_df) < min_rows:
            return False
        if "close" not in stock_df.columns:
            return False

        # 条件 5 (可选): 大盘环境
        if self.require_market_above_ma20 and not market_above_ma20:
            return False

        close = stock_df["close"].astype(float)
        close = pd.Series(np.nan_to_num(close.values), index=close.index)
        if (close <= 0).any():
            close = close.replace(0, np.nan).ffill().fillna(0.0)
            if (close <= 0).any():
                return False

        # 条件 6: 成交额达标 (amount 缺失时用 close*volume 近似)
        amount = self._last_amount(stock_df)
        if amount is None or amount <= self.min_amount:
            return False

        ma_s = close.rolling(self.ma_short, min_periods=self.ma_short).mean()
        ma_m = close.rolling(self.ma_mid, min_periods=self.ma_mid).mean()
        ma_l = close.rolling(self.ma_long, min_periods=self.ma_long).mean()

        last_close = float(close.iloc[-1])
        last_s, last_m, last_l = (
            float(ma_s.iloc[-1]), float(ma_m.iloc[-1]), float(ma_l.iloc[-1])
        )
        if np.isnan(last_s) or np.isnan(last_m) or np.isnan(last_l):
            return False

        # 条件 1: 价格在长期均线之上
        if last_close <= last_l:
            return False

        # 条件 2: 均线多头排列
        if not (last_s > last_m > last_l):
            return False

        # 条件 3: 5 日收益率 > 0
        ret_5d = last_close / float(close.iloc[-(RETURN_WINDOW + 1)]) - 1.0
        if ret_5d <= 0:
            return False

        # 条件 4: 20 日线性回归斜率 > 0 (trailing 窗口, 无未来函数)
        window = close.iloc[-SLOPE_WINDOW:].values
        slope = float(np.polyfit(np.arange(SLOPE_WINDOW), window, 1)[0])
        if slope <= 0:
            return False

        return True

    def batch_filter(self, all_stocks: Dict[str, pd.DataFrame],
                     market_above_ma20: bool = True) -> Dict[str, bool]:
        """批量预筛选.

        Args:
            all_stocks: {symbol: 日线 DataFrame}。
            market_above_ma20: 大盘状态。

        Returns:
            {symbol: is_uptrend}。典型通过率 20-30%。
        """
        result: Dict[str, bool] = {}
        for symbol, df in all_stocks.items():
            try:
                result[symbol] = self.is_uptrend(df, market_above_ma20)
            except Exception:  # noqa: BLE001 — 单股异常按非上行处理
                logger.exception("%s: 右侧预筛选异常, 按非上行处理", symbol)
                result[symbol] = False
        n_up = sum(result.values())
        logger.info("右侧预筛选: %d/%d 只上行 (%.1f%%)",
                    n_up, len(result), 100.0 * n_up / max(len(result), 1))
        return result

    # ── 内部工具 ────────────────────────────────────────────────────

    @staticmethod
    def _last_amount(stock_df: pd.DataFrame) -> float:
        """最新成交额 (元); amount 列缺失时用 close*volume 近似."""
        if "amount" in stock_df.columns:
            return float(np.nan_to_num(stock_df["amount"].iloc[-1]))
        if {"close", "volume"} <= set(stock_df.columns):
            return float(np.nan_to_num(
                stock_df["close"].iloc[-1] * stock_df["volume"].iloc[-1]
            ))
        return None
