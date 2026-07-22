# -*- coding: utf-8 -*-
"""分时数据加载器 (P10, ARCH §8.2.2).

为看板详情窗口提供分时/分钟 K 线数据:
实时分时 (akshare 1min, 内存缓存当日有效) + 历史分钟线 (Parquet 按需缓存)。

akshare 为可选重依赖 — 方法内惰性导入, 未安装时抛出带安装指引的
RuntimeError, 模块本身可独立导入与单元测试 (monkeypatch 假 akshare)。
"""

import datetime as _dt
import logging
import os
from typing import Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# akshare stock_zh_a_hist_min_em 中文列 → 统一英文列
_HIST_MIN_COLMAP = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "最新价": "close",
    "均价": "vwap",
}

# akshare stock_zh_a_minute 英文列 → 统一列
_MINUTE_COLMAP = {
    "day": "datetime",
}

_OUTPUT_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount"]


def _import_akshare():
    """惰性导入 akshare; 未安装时抛出带指引的 RuntimeError."""
    try:
        import akshare as ak  # noqa: WPS433 (刻意惰性导入)
    except ImportError as exc:
        raise RuntimeError(
            "akshare 未安装, 无法加载分时数据。请先执行: pip install akshare"
        ) from exc
    return ak


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """列名标准化 + 类型整理; 输出列齐 ``_OUTPUT_COLS`` (缺的补 NaN)."""
    df = df.rename(columns=_HIST_MIN_COLMAP).rename(columns=_MINUTE_COLMAP)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    for col in _OUTPUT_COLS:
        if col not in df.columns:
            df[col] = float("nan") if col != "datetime" else pd.NaT
    return df[_OUTPUT_COLS].sort_values("datetime").reset_index(drop=True)


class IntradayLoader:
    """分时数据加载器."""

    def __init__(self, cache_dir: str = "data/intraday/5min") -> None:
        """初始化.

        Args:
            cache_dir: 历史分钟线 Parquet 缓存目录 (按需创建)。
        """
        self.cache_dir = cache_dir
        # 实时分时内存缓存: (symbol, trade_date_str) -> DataFrame
        self._rt_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        os.makedirs(cache_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────
    #  实时分时
    # ────────────────────────────────────────────────────────────────

    def load_realtime(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """实时分时 (akshare stock_zh_a_minute, period=1, 内存缓存).

        缓存键为 (symbol, 当日日期) — 当日有效, 跨日自动失效。

        Args:
            symbol: 股票代码 (akshare 格式, 如 "sh600000")。
            use_cache: False 强制重新拉取。

        Returns:
            标准化 1 分钟分时 DataFrame。
        """
        today = _dt.date.today().isoformat()
        key = (symbol, today)
        if use_cache and key in self._rt_cache:
            logger.debug("实时分时命中缓存: %s", symbol)
            return self._rt_cache[key]

        ak = _import_akshare()
        logger.info("拉取实时分时: %s (akshare stock_zh_a_minute)", symbol)
        raw = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="qfq")
        df = _normalize(raw)
        self._rt_cache[key] = df
        return df

    def clear_realtime_cache(self, symbol: Optional[str] = None) -> None:
        """清空实时分时内存缓存; symbol=None 全清."""
        if symbol is None:
            self._rt_cache.clear()
        else:
            for key in [k for k in self._rt_cache if k[0] == symbol]:
                del self._rt_cache[key]

    # ────────────────────────────────────────────────────────────────
    #  历史分钟线
    # ────────────────────────────────────────────────────────────────

    def _cache_path(self, symbol: str, period: str) -> str:
        """Parquet 缓存路径."""
        safe = symbol.replace("/", "_")
        return os.path.join(self.cache_dir, f"{safe}_{period}min.parquet")

    def load_history_min(
        self, symbol: str, period: str = "5", start: str = None, end: str = None
    ) -> pd.DataFrame:
        """历史分钟线 (akshare stock_zh_a_hist_min_em, Parquet 缓存).

        缓存命中时按 start/end 过滤返回; 未命中则拉全量落盘。

        Args:
            symbol: 股票代码。
            period: 1/5/15/30/60 分钟。
            start: 开始日期 (含, "YYYY-MM-DD" 或 datetime 字符串)。
            end: 结束日期 (含)。

        Returns:
            标准化分钟 K 线 DataFrame。
        """
        path = self._cache_path(symbol, period)
        if os.path.exists(path):
            logger.debug("历史分钟线命中缓存: %s", path)
            df = pd.read_parquet(path)
        else:
            ak = _import_akshare()
            logger.info(
                "拉取历史分钟线: %s period=%s (akshare stock_zh_a_hist_min_em)",
                symbol,
                period,
            )
            raw = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                start_date="1979-09-01 09:32:00",
                end_date="2222-01-01 09:32:00",
                adjust="qfq",
            )
            df = _normalize(raw)
            df.to_parquet(path, index=False)
            logger.info("历史分钟线落盘: %s (%d 行)", path, len(df))

        df["datetime"] = pd.to_datetime(df["datetime"])
        if start is not None:
            df = df[df["datetime"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[
                df["datetime"]
                <= pd.to_datetime(end)
                + pd.Timedelta(days=1)
                - pd.Timedelta(microseconds=1)
            ]
        return df.reset_index(drop=True)
