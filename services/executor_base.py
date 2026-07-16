# -*- coding: utf-8 -*-
"""M3 — Executor interface + execution-mode toggle.

Two modes (settings.ExecutionMode):
  AUTO   (granted): orders sent to broker directly.
  MANUAL (pop-up):  only emit a recommendation; user must confirm.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """A single buy/sell order."""
    symbol: str
    side: str            # "buy" | "sell"
    qty: int
    price: Optional[float] = None  # None → market price


class Executor(ABC):
    """Abstract broker executor with mode toggle."""

    mode: settings.ExecutionMode = settings.EXECUTION_MODE

    @abstractmethod
    def _place(self, order: Order) -> dict:
        """Broker-specific order placement."""

    def execute(self, order: Order) -> dict:
        """Execute or recommend per current mode.

        Args:
            order: The Order to act on.

        Returns:
            dict describing what happened (executed vs recommended).
        """
        if self.mode is settings.ExecutionMode.MANUAL:
            logger.info("[MANUAL] recommend: %s %s %d %s",
                        order.side, order.symbol, order.qty,
                        order.price or "MKT")
            return {"mode": "manual", "recommendation": asdict(order),
                    "executed": False}
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
