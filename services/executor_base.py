# -*- coding: utf-8 -*-
"""M3 — Executor interface + execution-mode toggle.

Two modes (settings.ExecutionMode):
  AUTO   (granted): orders sent to broker directly.
  MANUAL (pop-up):  only emit a recommendation; user must confirm.

AUTO orders must carry market metadata (amount, pct_change) so the hard-constraint
risk filter can run. Missing metadata in AUTO mode is rejected (fail-safe).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from decimal import Decimal

from app.core import risk_filter
from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """A single buy/sell order."""

    symbol: str
    side: str  # "buy" | "sell"
    qty: int
    price: Decimal | None = None  # None → market price
    # Market metadata required for AUTO-mode risk-filter gate.
    amount: float | None = None
    pct_change: float | None = None
    is_st: bool = False
    list_days: int | None = None


class Executor(ABC):
    """Abstract broker executor with mode toggle and risk-filter gate."""

    mode: settings.ExecutionMode = settings.EXECUTION_MODE

    @abstractmethod
    def _place(self, order: Order) -> dict:
        """Broker-specific order placement."""

    def execute(self, order: Order) -> dict:
        """Execute or recommend per current mode.

        Args:
            order: The Order to act on.

        Returns:
            dict describing what happened (executed vs recommended vs rejected).
        """
        if self.mode is settings.ExecutionMode.MANUAL:
            price_str = str(order.price) if order.price is not None else "MKT"
            logger.info(
                "[MANUAL] recommend: %s %s %d @ %s",
                order.side,
                order.symbol,
                order.qty,
                price_str,
            )
            return {
                "mode": "manual",
                "recommendation": asdict(order),
                "executed": False,
            }

        # AUTO mode: hard-constraint risk filter gate.
        if order.amount is None or order.pct_change is None:
            logger.error(
                "[AUTO] reject %s %s: missing amount/pct_change metadata",
                order.side,
                order.symbol,
            )
            return {
                "mode": "auto",
                "executed": False,
                "reason": "missing_risk_metadata",
            }

        candidate = {
            "symbol": order.symbol,
            "score": 1.0,
            "amount": order.amount,
            "pct_change": order.pct_change,
            "is_st": order.is_st,
            "list_days": order.list_days,
        }
        if not risk_filter.apply_filters([candidate], account_drawdown_pct=0.0):
            logger.warning(
                "[AUTO] risk_filter reject %s %s (amount=%s pct_change=%s)",
                order.side,
                order.symbol,
                order.amount,
                order.pct_change,
            )
            return {
                "mode": "auto",
                "executed": False,
                "reason": "risk_filter_rejected",
            }

        logger.info("[AUTO] submit: %s %s %d", order.side, order.symbol, order.qty)
        return {"mode": "auto", "result": self._place(order), "executed": True}

    @abstractmethod
    def get_positions(self) -> dict:
        """Query current holdings."""

    @abstractmethod
    def sync_portfolio(self, target_holdings: dict) -> list:
        """Diff target vs actual → buy/sell list (A-share T+1 aware)."""


def get_executor() -> Executor:
    """Factory: SIM by default; miniQMT when AMINQT_BROKER=xt."""
    if settings.EXECUTION_BROKER == "xt":
        from services.xt_executor import XtExecutor

        return XtExecutor()
    from services.sim_executor import SimExecutor

    return SimExecutor()
