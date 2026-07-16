# -*- coding: utf-8 -*-
"""DataAdapter interface and factory.

Every data source exposes the SAME canonical column schema after loading:
    date, open, close, high, low, volume, amount
This keeps factor_engine / models fully source-agnostic.
"""
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

# akshare/iFinD Chinese → canonical English
CANONICAL_COLUMNS = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
}


class DataAdapter(ABC):
    """Abstract data-source adapter."""

    @abstractmethod
    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily K-line for one symbol.

        Args:
            symbol: Stock code, e.g. '600519'.
            start: Start date 'YYYY-MM-DD'.
            end: End date 'YYYY-MM-DD'.

        Returns:
            DataFrame with canonical columns (date, open, close, high,
            low, volume, amount), sorted by date ascending.
        """

    @abstractmethod
    def fetch_intraday(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """Fetch intraday bars for M2 pattern learning."""

    def fetch_many(self, symbols: Iterable[str],
                   start: str, end: str) -> Dict[str, pd.DataFrame]:
        """Fetch many symbols (loop with anti-crawl sleep)."""
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.fetch_daily(sym, start, end)
                logger.info("Fetched %s: %d rows", sym, len(out[sym]))
            except Exception as exc:  # noqa: BLE001
                logger.error("Fetch failed for %s: %s", sym, exc)
            time.sleep(settings.DOWNLOAD_SLEEP_SEC)
        return out

    @staticmethod
    def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
        """Rename Chinese columns to canonical English schema."""
        return df.rename(columns=CANONICAL_COLUMNS)


def get_adapter(source: Optional[str] = None) -> DataAdapter:
    """Factory: pick adapter by name; iFinD falls back to akshare on failure.

    Args:
        source: 'ifind' or 'akshare'. Defaults to settings.DATA_SOURCE.

    Returns:
        A concrete DataAdapter instance.
    """
    source = source or settings.DATA_SOURCE
    if source == "ifind":
        try:
            from .ifind_adapter import IfindAdapter
            logger.info("Using iFinD adapter")
            return IfindAdapter()
        except Exception as exc:  # noqa: BLE001
            logger.warning("iFinD unavailable (%s); falling back to akshare", exc)
    from .akshare_adapter import AkshareAdapter
    logger.info("Using akshare adapter")
    return AkshareAdapter()
