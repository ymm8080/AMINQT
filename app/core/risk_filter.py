# -*- coding: utf-8 -*-
"""Hard-constraint risk filter applied after model prediction (Phase 4).

Rules (selection_config.yaml::risk_filter):
  - Drop amount < min_amount.
  - Drop |pct_change| > price_limit_pct.
  - Drop ST names when exclude_st is true.
  - Drop names listed fewer than exclude_new_days days.
  - If account drawdown > max_account_drawdown_pct → return [] (circuit breaker).
"""

import logging

from app.core.config_loader import load_config
from config import settings

logger = logging.getLogger(__name__)


def _risk_cfg() -> dict:
    """Load risk_filter block from selection_config.yaml with settings fallback."""
    cfg = load_config("selection_config").get("risk_filter", {})
    return {
        "min_amount": float(cfg.get("min_amount", settings.MIN_AMOUNT)),
        "price_limit_pct": float(cfg.get("price_limit_pct", settings.PRICE_LIMIT_PCT)),
        "max_account_drawdown_pct": float(
            cfg.get("max_account_drawdown_pct", settings.MAX_ACCOUNT_DRAWDOWN_PCT)
        ),
        "exclude_st": bool(cfg.get("exclude_st", True)),
        "exclude_new_days": int(cfg.get("exclude_new_days", 5)),
    }


def apply_filters(
    candidates: list[dict],
    account_drawdown_pct: float = 0.0,
    cfg: dict | None = None,
) -> list[dict]:
    """Apply hard constraints to model-selected candidates.

    Args:
        candidates: List of candidate dicts. Expected keys:
            symbol (str), score (float), amount (float), pct_change (float),
            is_st (bool, optional), list_days (int, optional).
        account_drawdown_pct: Current account drawdown percentage.
        cfg: Optional risk_filter config override.

    Returns:
        Filtered candidate list, sorted by score descending.
    """
    cfg = cfg or _risk_cfg()

    if account_drawdown_pct > cfg["max_account_drawdown_pct"]:
        logger.warning(
            "Account drawdown %.2f%% > limit %.2f%%; returning empty",
            account_drawdown_pct,
            cfg["max_account_drawdown_pct"],
        )
        return []

    passed: list[dict] = []
    for c in candidates:
        symbol = c.get("symbol", "?")
        amount = float(c.get("amount", 0.0) or 0.0)
        pct_change = float(c.get("pct_change", 0.0) or 0.0)

        if amount < cfg["min_amount"]:
            logger.debug(
                "Risk filter drop %s: amount %.0f < %.0f",
                symbol,
                amount,
                cfg["min_amount"],
            )
            continue

        if abs(pct_change) > cfg["price_limit_pct"]:
            logger.debug(
                "Risk filter drop %s: |pct_change| %.2f > %.2f",
                symbol,
                pct_change,
                cfg["price_limit_pct"],
            )
            continue

        if cfg["exclude_st"] and c.get("is_st"):
            logger.debug("Risk filter drop %s: ST excluded", symbol)
            continue

        list_days = c.get("list_days")
        if list_days is not None and int(list_days) < cfg["exclude_new_days"]:
            logger.debug(
                "Risk filter drop %s: list_days %d < %d",
                symbol,
                int(list_days),
                cfg["exclude_new_days"],
            )
            continue

        passed.append(c)

    passed.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    logger.info(
        "Risk filter: %d candidates → %d passed (drawdown=%.2f%%)",
        len(candidates),
        len(passed),
        account_drawdown_pct,
    )
    return passed
