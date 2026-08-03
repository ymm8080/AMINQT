"""委托管理器 (P10, ARCH §9.4).

手动确认 / 自动执行 / 撤单; 与 TradingStateMachine 联动
(自动买/卖独立开关, 暂停/恢复/停止)。

幂等: client_order_id 唯一, 重复提交直接拒绝 (ValueError)。
T+1: 卖出数量不得超过可用持仓 (available_qty)。
审计日志: 每笔成交/拒绝追加写入日志文件, append-only, 不可变。
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from config import settings
from services.executor_base import Order

logger = logging.getLogger(__name__)

# 交易审计日志目录 (DATA_OTHERS_DIR/logs/trades/YYYY/MM/)
_TRADE_LOG_DIR = settings.DATA_OTHERS_DIR / "logs" / "trades"


def _trade_log_path() -> Path:
    """Return today's append-only trade log path (WORM: one file per day)."""
    now = datetime.now(timezone.utc)
    path = _TRADE_LOG_DIR / f"{now.year:04d}" / f"{now.month:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"trades_{now.strftime('%Y%m%d')}.jsonl"


def _append_trade_log(record: dict) -> None:
    """Append one immutable record to today's trade log.

    Args:
        record: Must be JSON-serializable.
    """
    log_path = _trade_log_path()
    line = json.dumps(record, ensure_ascii=False, default=str, sort_keys=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    logger.debug("[AUDIT] appended trade log: %s", log_path)


class OrderStatus(Enum):
    """委托状态."""

    PENDING_CONFIRM = "pending_confirm"  # 待手动确认
    SUBMITTED = "submitted"
    PARTIAL_FILLED = "partial_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# 允许撤单的状态
_CANCELLABLE = {OrderStatus.PENDING_CONFIRM, OrderStatus.SUBMITTED}


class OrderManager:
    """委托管理器."""

    def __init__(self, executor=None) -> None:
        """注入执行器 (XtExecutor / SimExecutor).

        Args:
            executor: Executor 实例; None 时回退到 get_executor() 工厂 (SIM)。
        """
        if executor is None:
            from services.executor_base import get_executor

            executor = get_executor()
        self.executor = executor
        self._orders: dict[str, dict] = {}  # order_id → 委托记录
        self._client_ids: dict[str, str] = {}  # client_order_id → order_id (幂等)
        self._fills: list[dict] = []  # 成交回报

    # ── 内部 helpers ──────────────────────────────────────────────

    def _snapshot(self, order_id: str) -> dict:
        """委托记录副本 (防外部篡改内部状态)."""
        return dict(self._orders[order_id])

    def _send_to_executor(self, rec: dict) -> bool:
        """调用执行器下单并更新状态/成交回报.

        Returns:
            True 执行器接受 (SUBMITTED), False 异常 (REJECTED)。
        """
        order = Order(
            symbol=rec["symbol"],
            side=rec["side"],
            qty=rec["qty"],
            price=rec["price"],
            amount=rec.get("amount"),
            pct_change=rec.get("pct_change"),
        )
        try:
            result = self.executor.execute(order)
        except Exception:
            logger.exception("[OM] 执行器下单异常: %s", rec["order_id"])
            rec["status"] = OrderStatus.REJECTED
            _append_trade_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": rec["order_id"],
                    "symbol": rec["symbol"],
                    "side": rec["side"],
                    "qty": rec["qty"],
                    "price": rec["price"],
                    "status": "rejected",
                    "reason": "executor_exception",
                }
            )
            return False
        rec["executor_result"] = result
        rec["status"] = OrderStatus.SUBMITTED
        logger.info(
            "[OM] 已报: %s %s %s %d @ %s",
            rec["order_id"],
            rec["side"],
            rec["symbol"],
            rec["qty"],
            rec["price"],
        )
        if result.get("executed"):
            fill = {
                "order_id": rec["order_id"],
                "symbol": rec["symbol"],
                "side": rec["side"],
                "qty": rec["qty"],
                "price": rec["price"],
                "result": result,
            }
            self._fills.append(fill)
            _append_trade_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    **fill,
                    "status": "filled",
                }
            )
        else:
            _append_trade_log(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": rec["order_id"],
                    "symbol": rec["symbol"],
                    "side": rec["side"],
                    "qty": rec["qty"],
                    "price": rec["price"],
                    "status": "rejected",
                    "reason": result.get("reason", "unknown"),
                }
            )
        return True

    # ── T+1 检查 ──────────────────────────────────────────────────

    def check_t1_sell(
        self, symbol: str, qty: int, positions: dict | None = None
    ) -> bool:
        """T+1 检查: 卖出数量 ≤ 可用持仓 (available_qty).

        Args:
            symbol: 股票代码。
            qty: 拟卖出数量。
            positions: 持仓 dict {symbol: {"available_qty": int, ...}};
                None 时通过执行器查询。

        Returns:
            True 允许卖出。
        """
        if positions is None:
            positions = self.executor.get_positions()
        pos = positions.get(symbol) or {}
        available = pos.get("available_qty", pos.get("qty", 0))
        ok = qty <= available
        if not ok:
            logger.warning(
                "[OM] T+1 拦截: %s 拟卖 %d > 可用 %d", symbol, qty, available
            )
        return ok

    # ── 委托操作 ──────────────────────────────────────────────────

    def submit(
        self,
        symbol: str,
        side: str,
        price: Decimal | float | int | None,
        qty: int,
        require_confirm: bool = True,
        client_order_id: str | None = None,
        amount: float | None = None,
        pct_change: float | None = None,
    ) -> str:
        """提交委托 (默认手动确认模式).

        Args:
            symbol: 股票代码。
            side: "buy" | "sell"。
            price: 委托价格 (Decimal/float/int/None for MKT)。
            qty: 数量。
            require_confirm: True → PENDING_CONFIRM 待手动确认;
                False → 直接报单 (SUBMITTED)。
            client_order_id: 客户端幂等键; 重复提交将被拒绝。
            amount: 最新日成交额 (AUTO 风控需要).
            pct_change: 最新日涨跌幅 %% (AUTO 风控需要).

        Returns:
            order_id。

        Raises:
            ValueError: client_order_id 重复 (幂等拒绝)。
        """
        client_order_id = client_order_id or uuid.uuid4().hex
        if client_order_id in self._client_ids:
            raise ValueError(
                f"重复 client_order_id 拒绝: {client_order_id} "
                f"(已存在委托 {self._client_ids[client_order_id]})"
            )
        order_id = uuid.uuid4().hex
        decimal_price = Decimal(str(price)) if price is not None else None
        rec = {
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "price": decimal_price,
            "qty": qty,
            "amount": amount,
            "pct_change": pct_change,
            "status": OrderStatus.PENDING_CONFIRM,
            "executor_result": None,
        }
        self._orders[order_id] = rec
        self._client_ids[client_order_id] = order_id
        logger.info(
            "[OM] 提交: %s %s %s %d @ %s (require_confirm=%s)",
            order_id,
            side,
            symbol,
            qty,
            price,
            require_confirm,
        )
        if not require_confirm:
            self._send_to_executor(rec)
        return order_id

    def manual_confirm(self, order_id: str) -> bool:
        """手动确认下单 (交易看板弹窗确认).

        Returns:
            True 确认并成功报单; False 委托不存在/状态不允许/执行器异常。
        """
        rec = self._orders.get(order_id)
        if rec is None:
            logger.warning("[OM] 确认失败, 委托不存在: %s", order_id)
            return False
        if rec["status"] is not OrderStatus.PENDING_CONFIRM:
            logger.warning(
                "[OM] 确认失败, 状态不允许: %s (%s)",
                order_id,
                rec["status"].value,
            )
            return False
        return self._send_to_executor(rec)

    def batch_confirm(self, order_ids: list[str]) -> int:
        """批量确认.

        Returns:
            成功确认的委托数。
        """
        return sum(1 for oid in order_ids if self.manual_confirm(oid))

    def cancel(self, order_id: str) -> bool:
        """撤单 (仅 PENDING_CONFIRM / SUBMITTED 可撤).

        Returns:
            True 撤单成功。
        """
        rec = self._orders.get(order_id)
        if rec is None:
            logger.warning("[OM] 撤单失败, 委托不存在: %s", order_id)
            return False
        if rec["status"] not in _CANCELLABLE:
            logger.warning(
                "[OM] 撤单失败, 状态不允许: %s (%s)",
                order_id,
                rec["status"].value,
            )
            return False
        rec["status"] = OrderStatus.CANCELLED
        logger.info("[OM] 已撤单: %s", order_id)
        return True

    def get_pending(self) -> list[dict]:
        """待确认委托队列 (看板展示)."""
        return [
            self._snapshot(oid)
            for oid, rec in self._orders.items()
            if rec["status"] is OrderStatus.PENDING_CONFIRM
        ]

    def get_fills(self, symbol: str | None = None) -> list[dict]:
        """成交回报 (可按 symbol 过滤)."""
        if symbol is None:
            return [dict(f) for f in self._fills]
        return [dict(f) for f in self._fills if f["symbol"] == symbol]
