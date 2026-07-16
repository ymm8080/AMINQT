# -*- coding: utf-8 -*-
"""M3 — miniQMT (xtquant) real executor.

Requires the miniQMT client + xtquant (NOT pip-installable). Enforces
A-share T+1: shares bought today cannot be sold today. Module imports
safely without xtquant; construction raises until the client is available.
"""

import logging
from datetime import date

from services.executor_base import Executor, Order

logger = logging.getLogger(__name__)


class XtExecutor(Executor):
    """miniQMT executor via xtquant."""

    def __init__(self) -> None:
        try:
            from xtquant import xttrader  # noqa: F401

            self._xt = __import__("xtquant")
        except ImportError as exc:
            raise RuntimeError("xtquant not installed (needs miniQMT client).") from exc
        self._today_bought: set = set()  # T+1 enforcement
        self._today: date = date.today()

    def _place(self, order: Order) -> dict:
        """Submit order to miniQMT.

        TODO(M3): real xttrader.order_stock(...) call + connection check.
        """
        return {"status": "not_connected", "order": order.__dict__}

    def get_positions(self) -> dict:
        """Query current holdings.

        TODO(M3): xttrader.query_stock_positions(...).
        """
        return {}

    def sync_portfolio(self, target_holdings: dict) -> list:
        """Diff target vs actual → order list, honoring A-share T+1.

        Args:
            target_holdings: {symbol: target_qty}.

        Returns:
            List of Order objects to reach the target portfolio.
        """
        self._rollover_day()
        actions: list = []
        actual = self.get_positions()
        for sym, target_qty in target_holdings.items():
            actual_qty = actual.get(sym, 0)
            diff = target_qty - actual_qty
            if diff > 0:
                actions.append(Order(sym, "buy", diff))
                self._today_bought.add(sym)
            elif diff < 0 and sym not in self._today_bought:
                actions.append(Order(sym, "sell", -diff))
            elif diff < 0:
                logger.info("T+1 block: cannot sell %s bought today", sym)
        return actions

    def _rollover_day(self) -> None:
        """Reset T+1 tracking on a new trading day."""
        today = date.today()
        if today != self._today:
            self._today = today
            self._today_bought.clear()
            logger.info("New trading day; T+1 set cleared")
