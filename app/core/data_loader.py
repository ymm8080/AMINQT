# -*- coding: utf-8 -*-
"""Local CSV loader + column canonicalization.

Reads raw CSVs written by scripts/download_data.py and renames the Chinese
akshare/iFinD columns to the canonical English schema used everywhere
downstream (factor_engine, models). Per PROMPT_CONTENT §1, renaming MUST
happen here on read.
"""

import logging
import os
from typing import Dict, Iterable, Optional

import pandas as pd

from config import settings
from data.adapters.base import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


def load_csv(symbol: str) -> pd.DataFrame:
    """Load one symbol's raw daily CSV and canonicalize columns.

    Args:
        symbol: Stock code, e.g. '600519'.

    Returns:
        Canonicalized DataFrame sorted by date ascending.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = os.path.join(settings.RAW_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.rename(columns=CANONICAL_COLUMNS)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    logger.info("Loaded %s: %d rows, cols=%s", symbol, len(df), list(df.columns))
    return df


def load_all(symbols: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
    """Load the whole pool into memory (Phase 4 <5s response requirement).

    Args:
        symbols: Iterable of codes. Defaults to settings.STOCK_LIST.

    Returns:
        Dict mapping symbol → canonicalized DataFrame.
    """
    symbols = list(symbols or settings.STOCK_LIST)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_csv(sym)
        except FileNotFoundError as exc:
            logger.error("Missing CSV for %s: %s", sym, exc)
    logger.info("Loaded %d/%d symbols into memory", len(out), len(symbols))
    return out
