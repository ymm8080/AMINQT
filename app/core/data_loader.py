"""Local Parquet loader + column canonicalization + OHLCV validation.

Reads raw Parquet files written by scripts/download_data.py and renames any
remaining Chinese akshare/iFinD columns to the canonical English schema used
everywhere downstream (factor_engine, models). Per PROMPT_CONTENT §1, renaming
MUST happen here on read.
"""

import logging
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd

from config import settings
from data.adapters.base import CANONICAL_COLUMNS

logger = logging.getLogger(__name__)


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV invariants after load.

    Checks:
      - high >= low
      - high >= open and high >= close
      - low <= open and low <= close
      - volume >= 0

    Args:
        df: DataFrame with canonical columns.

    Returns:
        The same DataFrame.

    Raises:
        ValueError: If any invariant is violated (异常数据不得静默丢弃).
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)

    ok = (
        np.isfinite(o)
        & np.isfinite(h)
        & np.isfinite(low)
        & np.isfinite(c)
        & np.isfinite(v)
        & (h >= low)
        & (h >= o)
        & (h >= c)
        & (low <= o)
        & (low <= c)
        & (v >= 0)
    )

    bad = int((~ok).sum())
    if bad:
        bad_idx = df.index[~ok].tolist()[:5]
        raise ValueError(
            f"OHLCV validation failed for {bad} row(s); first bad indices: {bad_idx}"
        )
    logger.debug("OHLCV validation passed: %d rows", len(df))
    return df


def load_parquet(symbol: str) -> pd.DataFrame:
    """Load one symbol's raw daily Parquet and canonicalize columns.

    Args:
        symbol: Stock code, e.g. '600519'.

    Returns:
        Canonicalized DataFrame sorted by date ascending.

    Raises:
        FileNotFoundError: If the Parquet does not exist.
        ValueError: If OHLCV invariants are violated.
    """
    path = os.path.join(settings.RAW_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        legacy = os.path.join(settings.RAW_DIR, f"{symbol}.csv")
        if os.path.exists(legacy):
            raise FileNotFoundError(
                f"{path} not found; legacy CSV exists. "
                "Re-run scripts/download_data.py to migrate to Parquet."
            )
        raise FileNotFoundError(path)

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.error("Failed to read Parquet %s: %s", path, exc)
        raise

    df = df.rename(columns=CANONICAL_COLUMNS)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    try:
        validate_ohlcv(df)
    except ValueError as exc:
        logger.error("OHLCV validation failed for %s: %s", symbol, exc)
        raise
    logger.info("Loaded %s: %d rows, cols=%s", symbol, len(df), list(df.columns))
    return df


def load_all(symbols: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the whole pool into memory (Phase 4 <5s response requirement).

    Args:
        symbols: Iterable of codes. Defaults to settings.STOCK_LIST.

    Returns:
        Dict mapping symbol → canonicalized DataFrame.
    """
    symbols = list(symbols or settings.STOCK_LIST)
    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = load_parquet(sym)
        except FileNotFoundError as exc:
            logger.error("Missing Parquet for %s: %s", sym, exc)
    logger.info("Loaded %d/%d symbols into memory", len(out), len(symbols))
    return out
