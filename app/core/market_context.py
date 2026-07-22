# -*- coding: utf-8 -*-
"""大盘指数上下文因子 — 将大盘走势注入每只股票的特征矩阵.

大盘走势对个股选股和交易至关重要。本模块负责：
1. 加载上证指数日线数据（通过 akshare 或本地缓存）
2. 计算 6 维大盘因子
3. 按日期 merge 到个股 DataFrame

因子清单 (6 列):
    - market_return_1d:    大盘1日收益率
    - market_return_5d:    大盘5日收益率
    - market_momentum:     大盘动量 (MA5 / MA20 - 1)
    - market_volatility:   大盘20日年化波动率
    - market_above_ma20:   大盘是否在MA20之上 (0/1)
    - market_trend:        大盘20日趋势斜率 (归一化)

使用方式:
    from app.core.market_context import MarketContext

    ctx = MarketContext()
    ctx.load("2023-01-01", "2025-01-01")  # 加载指数数据
    df_stock = ctx.merge_to_stock(df_stock)  # 注入大盘因子
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────
SH_INDEX_CODE = "000001"          # 上证指数
ANNUALIZATION = np.sqrt(252)      # 年化因子
MARKET_FACTOR_COLUMNS: List[str] = [
    "market_return_1d",
    "market_return_5d",
    "market_momentum",
    "market_volatility",
    "market_above_ma20",
    "market_trend",
]


class MarketContext:
    """大盘指数上下文管理器.

    负责加载大盘指数数据并计算 6 维因子，然后按日期注入个股 DataFrame。
    """

    def __init__(self, index_code: str = SH_INDEX_CODE):
        self.index_code = index_code
        self._index_df: Optional[pd.DataFrame] = None
        self._factors_df: Optional[pd.DataFrame] = None

    # ───────────────────────────────────────────────────────────────
    #  数据加载
    # ───────────────────────────────────────────────────────────────

    def load_from_akshare(self, start: str, end: str) -> pd.DataFrame:
        """通过 akshare 加载上证指数日线.

        Args:
            start: 起始日期 "YYYY-MM-DD"
            end:   结束日期 "YYYY-MM-DD"
        """
        import akshare as ak

        logger.info("通过 akshare 加载上证指数 %s ~ %s", start, end)
        try:
            raw = ak.stock_zh_index_daily(symbol=f"sh{self.index_code}")
        except Exception as e:
            logger.warning("akshare 加载失败, 尝试备用接口: %s", e)
            raw = ak.index_zh_a_hist(
                symbol=self.index_code,
                period="daily",
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
            )
            if "日期" in raw.columns:
                raw = raw.rename(columns={"日期": "date", "收盘": "close"})
                raw["date"] = pd.to_datetime(raw["date"])

        # 标准化列名
        if "date" not in raw.columns:
            # akshare stock_zh_index_daily 返回的列名可能是中文
            col_map = {"日期": "date", "收盘": "close", "最高": "high",
                       "最低": "low", "开盘": "open", "成交量": "volume"}
            raw = raw.rename(columns=col_map)

        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw[(raw["date"] >= start) & (raw["date"] <= end)].copy()
        raw = raw.sort_values("date").reset_index(drop=True)

        self._index_df = raw[["date", "open", "high", "low", "close", "volume"]].copy()
        logger.info("上证指数加载完成: %d 条", len(self._index_df))
        return self._index_df

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """从本地 Parquet 文件加载指数数据."""
        logger.info("从本地加载指数数据: %s", path)
        self._index_df = pd.read_parquet(path)
        self._index_df["date"] = pd.to_datetime(self._index_df["date"])
        self._index_df = self._index_df.sort_values("date").reset_index(drop=True)
        logger.info("指数数据加载完成: %d 条", len(self._index_df))
        return self._index_df

    def load_from_df(self, df: pd.DataFrame) -> None:
        """直接传入 DataFrame（用于测试或已有数据）.

        要求列: date, close (至少), 可选: open, high, low, volume
        """
        self._index_df = df.copy()
        self._index_df["date"] = pd.to_datetime(self._index_df["date"])
        self._index_df = self._index_df.sort_values("date").reset_index(drop=True)
        logger.info("指数数据(外部传入): %d 条", len(self._index_df))

    # ───────────────────────────────────────────────────────────────
    #  因子计算
    # ───────────────────────────────────────────────────────────────

    def compute_factors(self) -> pd.DataFrame:
        """计算 6 维大盘因子.

        Returns:
            DataFrame, 列: date + MARKET_FACTOR_COLUMNS
        """
        if self._index_df is None:
            raise RuntimeError("请先调用 load_from_*() 加载指数数据")

        df = self._index_df.copy()
        c = df["close"]

        # 1. 1日收益率
        df["market_return_1d"] = c.pct_change().fillna(0.0)

        # 2. 5日收益率
        df["market_return_5d"] = (c / c.shift(5) - 1.0).fillna(0.0)

        # 3. 动量 = MA5 / MA20 - 1
        ma5 = c.rolling(5, min_periods=1).mean()
        ma20 = c.rolling(20, min_periods=1).mean()
        momentum = (ma5 / ma20 - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df["market_momentum"] = momentum

        # 4. 20日年化波动率
        ret = c.pct_change()
        vol_20 = ret.rolling(20, min_periods=5).std() * ANNUALIZATION
        df["market_volatility"] = vol_20.fillna(0.0)

        # 5. 大盘是否在MA20之上
        df["market_above_ma20"] = (c > ma20).astype(float)

        # 6. 20日趋势斜率 (归一化)
        def _trend_slope(x):
            if len(x) < 5:
                return 0.0
            y = np.arange(len(x))
            slope = np.polyfit(y, x, 1)[0]
            # 归一化: 斜率 / 均值
            mean_val = np.mean(x)
            if abs(mean_val) < 1e-10:
                return 0.0
            return slope / mean_val

        df["market_trend"] = (
            c.rolling(20, min_periods=5)
            .apply(_trend_slope, raw=True)
            .fillna(0.0)
        )

        self._factors_df = df[["date"] + MARKET_FACTOR_COLUMNS].copy()

        # NaN → 0
        for col in MARKET_FACTOR_COLUMNS:
            self._factors_df[col] = self._factors_df[col].replace(
                [np.inf, -np.inf], 0.0
            ).fillna(0.0)

        logger.info("大盘因子计算完成: %d 列, %d 行", len(MARKET_FACTOR_COLUMNS), len(self._factors_df))
        return self._factors_df

    # ───────────────────────────────────────────────────────────────
    #  注入个股
    # ───────────────────────────────────────────────────────────────

    def merge_to_stock(self, stock_df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """将大盘因子按日期 merge 到个股 DataFrame.

        Args:
            stock_df: 个股日线, 必须包含 date 列
            date_col: 日期列名

        Returns:
            stock_df 附加 6 列 market_* 因子
        """
        if self._factors_df is None:
            raise RuntimeError("请先调用 compute_factors()")

        result = stock_df.copy()
        result[date_col] = pd.to_datetime(result[date_col])

        result = result.merge(
            self._factors_df,
            on=date_col,
            how="left",
        )

        # 缺失日期用前值填充，再补 0
        for col in MARKET_FACTOR_COLUMNS:
            result[col] = result[col].ffill().fillna(0.0)

        logger.debug("大盘因子注入完成: %d 行", len(result))
        return result

    # ───────────────────────────────────────────────────────────────
    #  辅助
    # ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_factor_columns() -> List[str]:
        """返回大盘因子列名列表."""
        return MARKET_FACTOR_COLUMNS.copy()

    def is_ready(self) -> bool:
        """是否已计算好因子."""
        return self._factors_df is not None
