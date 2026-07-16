# -*- coding: utf-8 -*-
"""Hard-constraint risk filter applied after model prediction (Phase 4)."""

import logging
from typing import List

from config import settings

logger = logging.getLogger(__name__)


def apply_filters(
    candidates: List[dict], account_drawdown_pct: float = 0.0
) -> List[dict]:
    """Apply hard constraints to model-selected candidates.

    TODO(Phase 4): implement.
      - Drop amount < MIN_AMOUNT (5000万).
      - Drop |涨跌幅| > PRICE_LIMIT_PCT (9.5%).
      - If account drawdown > MAX_ACCOUNT_DRAWDOWN_PCT (3%) → return [].

    Args:
        candidates: List of {symbol, score, amount, pct_change, ...}.
        account_drawdown_pct: Current account drawdown percentage.

    Returns:
        Filtered candidate list (Top-N by score).
    """
    if account_drawdown_pct > settings.MAX_ACCOUNT_DRAWDOWN_PCT:
        logger.warning(
            "Account drawdown %.2f%% > limit; returning empty", account_drawdown_pct
        )
        return []
    raise NotImplementedError("risk_filter.apply_filters — implement in Phase 4")
