# -*- coding: utf-8 -*-
"""模块3: DataLoader — 数据加载与对齐.

职责:
    1. 加载预测表、行情表、基准表
    2. 校验必要列
    3. 对齐日期和股票代码
    4. 预计算 20 日日均成交额 (avg_amount_20d)
    5. 构建交易日历
    6. 生成数据版本哈希

所有价格在 DataLoader 阶段保持原始精度 (元),
BacktestEngine 内部统一转换为分 (fen).
"""

import hashlib
import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from app.backtest.config_manager import BacktestConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载与对齐.

    加载预测表、行情表、基准表, 校验必要列, 对齐日期和股票代码.
    """

    REQUIRED_PRED_COLS = [
        "date",
        "stock",
        "score_h1",
        "prob_up_h1",
        "pred_ret_h1",
        "score_h2",
        "prob_up_h2",
        "pred_ret_h2",
        "score_h4",
        "prob_up_h4",
        "pred_ret_h4",
        "board",
    ]

    REQUIRED_PRICE_COLS = [
        "date",
        "stock",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "up_limit",
        "down_limit",
        "is_st",
        "is_halt",
        "pre_close",
        "circ_mv",
    ]

    REQUIRED_MARKET_COLS = [
        "date",
        "index_close",
    ]

    def __init__(
        self,
        pred_path: str,
        price_path: str,
        benchmark_path: str | None = None,
        market_path: str | None = None,
        config: BacktestConfig | None = None,
    ):
        """初始化数据加载器.

        Args:
            pred_path: 预测表路径 (CSV/Parquet).
            price_path: 行情表路径 (CSV/Parquet).
            benchmark_path: 基准表路径 (可选).
            market_path: 大盘指数表路径 (V5.0, 可选).
            config: 回测配置.
        """
        self.pred_path = pred_path
        self.price_path = price_path
        self.benchmark_path = benchmark_path
        self.market_path = market_path
        self.config = config or BacktestConfig()

        self.pred_df: pd.DataFrame | None = None
        self.price_df: pd.DataFrame | None = None
        self.benchmark_df: pd.DataFrame | None = None
        self.market_df: pd.DataFrame | None = None
        self.trade_dates: List[pd.Timestamp] = []
        self._data_version_hash: str = ""

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """加载并对齐数据.

        Returns:
            (pred_df, price_df, benchmark_df).

        Raises:
            FileNotFoundError: 文件不存在.
            ValueError: 必要列缺失.
        """
        self.pred_df = self._read_file(self.pred_path, "预测表")
        self.price_df = self._read_file(self.price_path, "行情表")
        if self.benchmark_path:
            self.benchmark_df = self._read_file(self.benchmark_path, "基准表")
        else:
            self.benchmark_df = pd.DataFrame()

        # V5.0: 加载大盘指数数据
        if self.market_path:
            self.market_df = self._read_file(self.market_path, "大盘指数表")
            self._validate(self.market_df, self.REQUIRED_MARKET_COLS, "大盘指数表")
            self.market_df["date"] = pd.to_datetime(self.market_df["date"])
        else:
            self.market_df = pd.DataFrame()

        self._validate(
            self.pred_df, self.REQUIRED_PRED_COLS, "预测表"
        )
        self._validate(
            self.price_df, self.REQUIRED_PRICE_COLS, "行情表"
        )

        self._align()
        self._calc_rolling_metrics()
        self._build_trade_calendar()
        self._data_version_hash = self.get_data_version_hash()

        return self.pred_df, self.price_df, self.benchmark_df, self.market_df

    @staticmethod
    def _read_file(path: str, name: str) -> pd.DataFrame:
        """根据文件扩展名读取 CSV 或 Parquet.

        Args:
            path: 文件路径.
            name: 文件名称 (日志用).

        Returns:
            DataFrame.

        Raises:
            FileNotFoundError: 文件不存在.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}文件不存在: {path}")

        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)
        logger.info("%s加载完成: %s, shape=%s", name, path, df.shape)
        return df

    def _validate(
        self, df: pd.DataFrame, required_cols: list, name: str
    ) -> None:
        """校验必要列是否存在.

        Args:
            df: 待校验 DataFrame.
            required_cols: 必要列名列表.
            name: 表名.

        Raises:
            ValueError: 缺失列.
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name}缺失必要列: {missing}")

    def _align(self) -> None:
        """对齐预测表和行情表.

        1. 统一日期格式为 pd.Timestamp
        2. 股票代码统一为字符串
        3. 删除预测表中行情缺失的 (date, stock)
        4. 按 (date, stock) 排序 (强制排序确保可重复)
        """
        assert self.pred_df is not None
        assert self.price_df is not None

        # 统一日期格式
        self.pred_df["date"] = pd.to_datetime(self.pred_df["date"])
        self.price_df["date"] = pd.to_datetime(self.price_df["date"])
        if self.benchmark_df is not None and not self.benchmark_df.empty:
            self.benchmark_df["date"] = pd.to_datetime(
                self.benchmark_df["date"]
            )

        # 股票代码统一为字符串
        self.pred_df["stock"] = self.pred_df["stock"].astype(str)
        self.price_df["stock"] = self.price_df["stock"].astype(str)

        # 创建行情表的 (date, stock) 索引集合
        price_keys = set(
            zip(self.price_df["date"], self.price_df["stock"])
        )

        # 删除预测表中行情缺失的行
        pred_keys = list(
            zip(self.pred_df["date"], self.pred_df["stock"])
        )
        mask = [k in price_keys for k in pred_keys]
        before = len(self.pred_df)
        self.pred_df = self.pred_df[mask].copy()
        after = len(self.pred_df)
        if before != after:
            logger.warning(
                "预测表对齐: 删除 %d 行缺失行情 (%d → %d)",
                before - after,
                before,
                after,
            )

        # 强制排序 (可重复性要求)
        self.pred_df = self.pred_df.sort_values(
            ["date", "stock"]
        ).reset_index(drop=True)
        self.price_df = self.price_df.sort_values(
            ["date", "stock"]
        ).reset_index(drop=True)

        logger.info(
            "数据对齐完成: pred=%d, price=%d, dates=%d",
            len(self.pred_df),
            len(self.price_df),
            self.price_df["date"].nunique(),
        )

    def _calc_rolling_metrics(self) -> None:
        """预计算滚动指标: 20日日均成交额 (avg_amount_20d).

        在行情表中新增 avg_amount_20d 列, 用于流动性过滤.
        """
        assert self.price_df is not None
        df = self.price_df

        if "avg_amount_20d" not in df.columns:
            df["avg_amount_20d"] = (
                df.groupby("stock")["amount"]
                .rolling(window=20, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            logger.info("预计算 avg_amount_20d 完成")

    def _build_trade_calendar(self) -> None:
        """构建交易日历 (使用 trade_dates 列表, 不用 pd.Timedelta)."""
        assert self.price_df is not None
        self.trade_dates = sorted(self.price_df["date"].unique())
        logger.info("交易日历构建: %d 个交易日", len(self.trade_dates))

    def get_trade_dates(self) -> List[pd.Timestamp]:
        """获取所有交易日列表 (按时间排序).

        Returns:
            交易日 Timestamp 列表.
        """
        return self.trade_dates

    def get_next_trade_date(
        self, date: pd.Timestamp, n: int = 1
    ) -> pd.Timestamp | None:
        """获取 date 后第 n 个交易日.

        使用交易日历列表查找, 不使用 pd.Timedelta (避免节假日错误).

        Args:
            date: 基准日期.
            n: 偏移量 (1 = 下一个交易日).

        Returns:
            交易日 Timestamp, 不存在时返回 None.
        """
        idx = np.searchsorted(self.trade_dates, date)
        target = idx + n
        if 0 <= target < len(self.trade_dates):
            return self.trade_dates[target]
        return None

    def get_data_version_hash(self) -> str:
        """计算数据版本哈希 (用于审计和复现).

        基于预测表和行情表的行数、日期范围、列名生成哈希.

        Returns:
            "sha256:" 前缀的哈希字符串.
        """
        assert self.pred_df is not None
        assert self.price_df is not None

        parts = [
            f"pred_rows={len(self.pred_df)}",
            f"price_rows={len(self.price_df)}",
            f"pred_dates={self.pred_df['date'].min()}_{self.pred_df['date'].max()}",
            f"price_dates={self.price_df['date'].min()}_{self.price_df['date'].max()}",
            f"pred_cols={','.join(sorted(self.pred_df.columns))}",
            f"price_cols={','.join(sorted(self.price_df.columns))}",
        ]
        raw = "|".join(parts).encode("utf-8")
        h = hashlib.sha256(raw).hexdigest()[:16]
        return f"sha256:{h}"
