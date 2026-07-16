# -*- coding: utf-8 -*-
"""akshare data adapter — free, no credentials. Default dev/fallback source."""
import logging

import pandas as pd

from .base import DataAdapter

logger = logging.getLogger(__name__)


class AkshareAdapter(DataAdapter):
    """Daily + intraday data via akshare."""

    def fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Fetch daily qfq K-line via ak.stock_zh_a_hist."""
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
        df = self._canonicalize(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        logger.info("akshare daily %s: %d rows", symbol, len(df))
        return df

    def fetch_intraday(self, symbol: str, trade_date: str) -> pd.DataFrame:
        """Fetch 1-minute bars via ak.stock_zh_a_hist_min_em (M2 source)."""
        import akshare as ak
        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="1", adjust="qfq")
        df = self._canonicalize(df)
        logger.info("akshare intraday %s: %d rows", symbol, len(df))
        return df
