# -*- coding: utf-8 -*-
"""Tests for services/executor — mode toggle + risk-filter gate + T+1 (M3)."""

from decimal import Decimal

import pytest

from config import settings
from services.executor_base import Executor, Order
from services.sim_executor import SimExecutor
import services.executor_base as executor_base


class _StubExecutor(Executor):
    """Test executor that records placed orders."""

    def __init__(self):
        self.placed: list[Order] = []

    def _place(self, order: Order) -> dict:
        self.placed.append(order)
        return {"status": "filled", "order": order.symbol}

    def get_positions(self) -> dict:
        return {}

    def sync_portfolio(self, target_holdings: dict) -> list:
        return []


@pytest.fixture
def manual_executor(monkeypatch):
    monkeypatch.setattr(executor_base.Executor, "mode", settings.ExecutionMode.MANUAL)
    return _StubExecutor()


@pytest.fixture
def auto_executor(monkeypatch):
    monkeypatch.setattr(executor_base.Executor, "mode", settings.ExecutionMode.AUTO)
    return _StubExecutor()


def test_manual_mode_returns_recommendation(manual_executor):
    order = Order(
        symbol="000001",
        side="buy",
        qty=100,
        price=Decimal("10"),
        amount=1e8,
        pct_change=1.0,
    )
    result = manual_executor.execute(order)
    assert result["mode"] == "manual"
    assert result["executed"] is False
    assert result["recommendation"]["symbol"] == "000001"
    assert not manual_executor.placed


def test_auto_mode_accepts_passing_risk_filter(auto_executor):
    order = Order(
        symbol="000001",
        side="buy",
        qty=100,
        price=Decimal("10"),
        amount=1e8,
        pct_change=1.0,
    )
    result = auto_executor.execute(order)
    assert result["mode"] == "auto"
    assert result["executed"] is True
    assert len(auto_executor.placed) == 1


def test_auto_mode_rejects_missing_metadata(auto_executor):
    order = Order(symbol="000001", side="buy", qty=100, price=10.0)
    result = auto_executor.execute(order)
    assert result["executed"] is False
    assert result["reason"] == "missing_risk_metadata"
    assert not auto_executor.placed


def test_auto_mode_rejects_amount_too_low(auto_executor):
    order = Order(
        symbol="000001", side="buy", qty=100, price=10.0, amount=1e6, pct_change=1.0
    )
    result = auto_executor.execute(order)
    assert result["executed"] is False
    assert result["reason"] == "risk_filter_rejected"
    assert not auto_executor.placed


def test_auto_mode_rejects_price_limit(auto_executor):
    order = Order(
        symbol="000001", side="buy", qty=100, price=10.0, amount=1e8, pct_change=10.0
    )
    result = auto_executor.execute(order)
    assert result["executed"] is False
    assert result["reason"] == "risk_filter_rejected"


def test_sim_executor_prints_order(capfd):
    order = Order(symbol="000001", side="buy", qty=100, price=10.0)
    ex = SimExecutor()
    result = ex._place(order)
    assert result["status"] == "sim_filled"
    out, _ = capfd.readouterr()
    assert "[SIM]" in out
    assert "000001" in out
