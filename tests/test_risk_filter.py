"""Tests for app/core/risk_filter — hard constraints (Phase 4)."""

import pytest

from app.core import risk_filter


@pytest.fixture
def cfg():
    return {
        "min_amount": 50_000_000.0,
        "price_limit_pct": 9.5,
        "max_account_drawdown_pct": 3.0,
    }


def _c(symbol, score, amount, pct_change):
    return {
        "symbol": symbol,
        "score": score,
        "amount": amount,
        "pct_change": pct_change,
    }


def test_amount_filter_drops_low_amount(cfg):
    candidates = [
        _c("A", score=0.9, amount=60_000_000.0, pct_change=2.0),
        _c("B", score=0.8, amount=10_000_000.0, pct_change=2.0),
    ]
    out = risk_filter.apply_filters(candidates, cfg=cfg)
    assert [c["symbol"] for c in out] == ["A"]


def test_price_limit_filter_drops_limit_up(cfg):
    candidates = [
        _c("A", score=0.9, amount=60_000_000.0, pct_change=9.6),
        _c("B", score=0.8, amount=60_000_000.0, pct_change=-9.6),
        _c("C", score=0.7, amount=60_000_000.0, pct_change=9.4),
    ]
    out = risk_filter.apply_filters(candidates, cfg=cfg)
    assert [c["symbol"] for c in out] == ["C"]


def test_drawdown_circuit_breaker_returns_empty(cfg):
    candidates = [_c("A", score=0.9, amount=60_000_000.0, pct_change=2.0)]
    out = risk_filter.apply_filters(candidates, account_drawdown_pct=3.1, cfg=cfg)
    assert out == []


def test_output_sorted_by_score_desc(cfg):
    candidates = [
        _c("A", score=0.5, amount=60_000_000.0, pct_change=2.0),
        _c("B", score=0.9, amount=60_000_000.0, pct_change=2.0),
        _c("C", score=0.7, amount=60_000_000.0, pct_change=2.0),
    ]
    out = risk_filter.apply_filters(candidates, cfg=cfg)
    assert [c["symbol"] for c in out] == ["B", "C", "A"]
