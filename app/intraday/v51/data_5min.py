"""
5 分钟数据下载 (V5.1 P20.0 W1, akshare 适配层)
====================================================
薄适配层: 拉取 → 列名规范化 → 缓存 parquet (WORM, 不覆盖).
测试与回测用 mock fetcher 注入, 不在引擎内直接联网.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd

logger = logging.getLogger(__name__)

RETRY = 3
RETRY_SLEEP = 2.0


def normalize_5min(df: pd.DataFrame, symbol: str, trade_date: str) -> pd.DataFrame:
    """akshare 5min 列名 → 标准列 (t/open/high/low/close/volume/amount)."""
    col_map = {
        "时间": "datetime", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }
    out = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    out["symbol"] = symbol
    out["trade_date"] = str(trade_date)
    if "datetime" in out.columns:
        out["t"] = pd.to_datetime(out["datetime"]).dt.strftime("%H:%M")
    keep = ["symbol", "trade_date", "t", "open", "high", "low", "close",
            "volume", "amount"]
    return out[[c for c in keep if c in out.columns]]


class IntradayDataLoader:
    """5min 数据拉取 + parquet 缓存 (失败重试 3 次 + 告警)."""

    def __init__(self, cache_dir: str = "data/intraday_5min", fetcher=None):
        self.cache_dir = cache_dir
        self.fetcher = fetcher  # 注入: fn(symbol, trade_date) -> DataFrame
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, symbol: str, trade_date: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{trade_date}.parquet")

    def load(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """缓存优先, 缺失则拉取入库 (WORM: 已有文件不覆盖)."""
        path = self._cache_path(symbol, trade_date)
        if os.path.exists(path):
            return pd.read_parquet(path)
        df = self._fetch_with_retry(symbol, trade_date)
        df.to_parquet(path, index=False)
        return df

    def _fetch_with_retry(self, symbol: str, trade_date: str) -> pd.DataFrame:
        fetcher = self.fetcher or self._akshare_fetch
        last_err = None
        for attempt in range(1, RETRY + 1):
            try:
                raw = fetcher(symbol, trade_date)
                return normalize_5min(raw, symbol, trade_date)
            except Exception as exc:  # noqa: BLE001 — 重试后仍失败则告警上抛
                last_err = exc
                logger.warning("5min 拉取失败 (%s %s, 第%d次): %s",
                               symbol, trade_date, attempt, exc)
                time.sleep(RETRY_SLEEP)
        logger.error("5min 拉取连续 %d 次失败: %s %s", RETRY, symbol, trade_date)
        raise RuntimeError(f"5min 数据拉取失败: {symbol} {trade_date}") from last_err

    @staticmethod
    def _akshare_fetch(symbol: str, trade_date: str) -> pd.DataFrame:
        """生产数据源: akshare 5min (需联网, 仅在本适配层调用)."""
        import akshare as ak

        return ak.stock_zh_a_hist_min_em(
            symbol=symbol, period="5",
            start_date=f"{trade_date} 09:30:00",
            end_date=f"{trade_date} 15:00:00",
            adjust="",
        )
