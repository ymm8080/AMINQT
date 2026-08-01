"""Tests for services/order_manager.py — confirm flow, cancel, T+1, idempotency, audit log."""

import json
from decimal import Decimal

import pytest

import services.order_manager as om_module
from services.order_manager import OrderManager, OrderStatus


@pytest.fixture
def trade_log_dir(tmp_path, monkeypatch):
    """Redirect append-only trade log to a temp directory."""
    path = tmp_path / "logs" / "trades"
    monkeypatch.setattr(om_module, "_TRADE_LOG_DIR", path)
    return path


class FakeExecutor:
    """记录下单的假执行器 (duck-type Executor.execute)."""

    def __init__(self):
        self.placed = []
        self.positions = {}

    def execute(self, order):
        self.placed.append(order)
        return {"mode": "auto", "executed": True, "result": {"status": "filled"}}

    def get_positions(self):
        return self.positions


class FailingExecutor(FakeExecutor):
    """下单即抛异常."""

    def execute(self, order):
        raise ConnectionError("broker down")


@pytest.fixture
def executor():
    return FakeExecutor()


@pytest.fixture
def om(executor):
    return OrderManager(executor=executor)


class TestSubmitConfirmFlow:
    """提交 → 待确认 → 手动确认 → 已报."""

    def test_submit_default_pending(self, om, executor):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        assert executor.placed == []
        rec = om._orders[oid]
        assert rec["status"] is OrderStatus.PENDING_CONFIRM

    def test_get_pending_lists_order(self, om):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        pending = om.get_pending()
        assert len(pending) == 1
        assert pending[0]["order_id"] == oid
        assert pending[0]["status"] is OrderStatus.PENDING_CONFIRM

    def test_manual_confirm_sends_to_executor(self, om, executor):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        assert om.manual_confirm(oid) is True
        assert len(executor.placed) == 1
        assert executor.placed[0].symbol == "600519"
        assert om._orders[oid]["status"] is OrderStatus.SUBMITTED

    def test_manual_confirm_twice_rejected(self, om):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        assert om.manual_confirm(oid) is True
        assert om.manual_confirm(oid) is False

    def test_manual_confirm_unknown_id(self, om):
        assert om.manual_confirm("nonexistent") is False

    def test_direct_submit_no_confirm(self, om, executor):
        oid = om.submit("000001", "sell", Decimal("12.5"), 200, require_confirm=False)
        assert len(executor.placed) == 1
        assert om._orders[oid]["status"] is OrderStatus.SUBMITTED

    def test_batch_confirm(self, om, executor):
        ids = [
            om.submit("600519", "buy", Decimal("1685"), 100),
            om.submit("000001", "buy", Decimal("12"), 200),
            om.submit("300750", "buy", Decimal("180"), 100),
        ]
        ok = om.batch_confirm([ids[0], ids[1], "bad-id"])
        assert ok == 2
        assert len(executor.placed) == 2
        assert om._orders[ids[2]]["status"] is OrderStatus.PENDING_CONFIRM

    def test_executor_failure_marks_rejected(self):
        om = OrderManager(executor=FailingExecutor())
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        assert om.manual_confirm(oid) is False
        assert om._orders[oid]["status"] is OrderStatus.REJECTED

    def test_get_fills(self, om):
        oid1 = om.submit("600519", "buy", Decimal("1685"), 100)
        om.submit("000001", "buy", Decimal("12"), 200)
        om.manual_confirm(oid1)
        fills = om.get_fills()
        assert len(fills) == 1
        assert fills[0]["order_id"] == oid1
        assert om.get_fills(symbol="000001") == []
        assert len(om.get_fills(symbol="600519")) == 1


class TestCancel:
    """撤单."""

    def test_cancel_pending(self, om):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        assert om.cancel(oid) is True
        assert om._orders[oid]["status"] is OrderStatus.CANCELLED
        assert om.get_pending() == []

    def test_cancel_submitted(self, om):
        oid = om.submit("600519", "buy", Decimal("1685"), 100, require_confirm=False)
        assert om.cancel(oid) is True

    def test_cancel_twice_rejected(self, om):
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        om.cancel(oid)
        assert om.cancel(oid) is False

    def test_cancel_unknown_id(self, om):
        assert om.cancel("nonexistent") is False


class TestT1Check:
    """T+1: 卖出数量 ≤ 可用持仓."""

    def test_sell_within_available(self, om):
        positions = {"600519": {"available_qty": 300, "qty": 500}}
        assert om.check_t1_sell("600519", 300, positions) is True

    def test_sell_above_available_blocked(self, om):
        positions = {"600519": {"available_qty": 100, "qty": 500}}
        assert om.check_t1_sell("600519", 200, positions) is False

    def test_sell_without_position_blocked(self, om):
        assert om.check_t1_sell("600519", 100, {}) is False

    def test_positions_from_executor(self, om, executor):
        executor.positions = {"000001": {"available_qty": 1000}}
        assert om.check_t1_sell("000001", 500) is True
        assert om.check_t1_sell("000001", 1500) is False

    def test_fallback_to_qty_field(self, om):
        positions = {"600519": {"qty": 400}}
        assert om.check_t1_sell("600519", 400, positions) is True


class TestIdempotency:
    """client_order_id 幂等."""

    def test_duplicate_client_order_id_rejected(self, om):
        om.submit("600519", "buy", Decimal("1685"), 100, client_order_id="cid-1")
        with pytest.raises(ValueError, match="重复 client_order_id"):
            om.submit("600519", "buy", Decimal("1685"), 100, client_order_id="cid-1")

    def test_distinct_client_ids_ok(self, om):
        om.submit("600519", "buy", Decimal("1685"), 100, client_order_id="cid-1")
        om.submit("600519", "buy", Decimal("1685"), 100, client_order_id="cid-2")
        assert len(om.get_pending()) == 2

    def test_auto_generated_ids_unique(self, om):
        om.submit("600519", "buy", Decimal("1685"), 100)
        om.submit("600519", "buy", Decimal("1685"), 100)
        assert len(om.get_pending()) == 2


class TestAuditLog:
    """交易审计日志 append-only."""

    def _read_log_lines(self, trade_log_dir):
        files = list(trade_log_dir.rglob("*.jsonl"))
        if not files:
            return []
        return files[0].read_text(encoding="utf-8").strip().split("\n")

    def test_filled_order_appended_to_log(self, om, trade_log_dir):
        oid = om.submit("600519", "buy", Decimal("1685"), 100, require_confirm=False)
        lines = self._read_log_lines(trade_log_dir)
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["order_id"] == oid
        assert rec["symbol"] == "600519"
        assert rec["side"] == "buy"
        assert rec["status"] == "filled"

    def test_rejected_order_appended_to_log(self, trade_log_dir):
        om = OrderManager(executor=FailingExecutor())
        oid = om.submit("600519", "buy", Decimal("1685"), 100)
        om.manual_confirm(oid)
        lines = self._read_log_lines(trade_log_dir)
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["order_id"] == oid
        assert rec["status"] == "rejected"
        assert rec["reason"] == "executor_exception"

    def test_log_is_append_only(self, om, trade_log_dir):
        om.submit("600519", "buy", Decimal("1685"), 100, require_confirm=False)
        om.submit("000001", "sell", Decimal("12"), 200, require_confirm=False)
        lines = self._read_log_lines(trade_log_dir)
        assert len(lines) == 2
        assert {json.loads(line)["symbol"] for line in lines} == {"600519", "000001"}
