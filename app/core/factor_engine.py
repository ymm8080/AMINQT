# -*- coding: utf-8 -*-
"""M1 — Factor engine: convert daily K-line into a numeric feature matrix.

Phase 2 will implement build_features(). This module fixes the contract and
the 防坑 rules it MUST obey (PROMPT_CONTENT §2):
  * Future-function prevention: indicator at day t uses only data up to t
    (pandas rolling/expanding; NEVER shift(-k) of future bars).
  * Divide-by-zero: (close - DIF) / DIF → 0 when DIF == 0 (use safe_divide).
  * NaN → 0 before model input: np.nan_to_num(X).
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Target shape: (samples, 20 days, >=25 features); y = future 5-day return
WINDOW_DAYS = 20
HORIZON_DAYS = 5


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Build windowed feature matrix X and future-return label y.

    TODO(Phase 2): implement.
      - Technical indicators: MACD (DIF/DEA/BAR), KDJ (K/D/J),
        BOLL (upper/mid/lower), RSI.
      - Derived: (close-DIF)/DIF deviation, close/MA5-1 bias,
        5-day linear slope of indicators.
      - Sliding window → (N, 20, F); y = future 5-day return.
      - np.nan_to_num(X) before return.

    Args:
        df: Canonicalized daily DataFrame (date, open, close, high,
            low, volume, amount).

    Returns:
        (X, y): X shape (N, WINDOW_DAYS, F>=25), y shape (N,).
    """
    raise NotImplementedError("factor_engine.build_features — implement in Phase 2")


def safe_divide(numerator, denominator) -> pd.Series:
    """Element-wise divide returning 0 where denominator is 0 (防 inf).

    Args:
        numerator: Array-like (Series or scalar).
        denominator: Array-like (Series or scalar).

    Returns:
        pd.Series with inf/NaN-from-zero-denominator replaced by 0.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return pd.Series(out, index=getattr(numerator, "index", None))
