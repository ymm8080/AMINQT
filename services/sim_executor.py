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
        price_str = str(order.price) if order.price is not None else "市价"
        print(f"[SIM] {verb} {order.symbol} {order.qty}股 @ {price_str}")
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
