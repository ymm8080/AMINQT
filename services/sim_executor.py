# -*- coding: utf-8 -*-
"""M3 — Simulator executor: prints orders, touches no broker.

Used for dev and as the MANUAL-mode recommendation sink.
"""

import logging

from services.executor_base import Executor, Order

logger = logging.getLogger(__name__)


class SimExecutor(Executor):
    """Print-only executor."""

    def _place(self, order: Order) -> dict:
        verb = "买入" if order.side == "buy" else "卖出"
        print(f"[SIM] {verb} {order.symbol} {order.qty}股 @ {order.price or '市价'}")
        return {
            "status": "sim_filled",
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
        }

    def get_positions(self) -> dict:
        return {}

    def sync_portfolio(self, target_holdings: dict) -> list:
        logger.info("[SIM] sync_portfolio target=%s", target_holdings)
        return []
