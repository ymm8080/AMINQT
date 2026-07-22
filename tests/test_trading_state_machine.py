# -*- coding: utf-8 -*-
"""Tests for services/trading_state_machine.py — transitions, flags, can_execute."""

import logging

import pytest

from services.trading_state_machine import TradingState, TradingStateMachine


@pytest.fixture
def tsm():
    return TradingStateMachine()


class TestTransitions:
    """状态迁移矩阵."""

    def test_initial_state(self, tsm):
        assert tsm.state is TradingState.STOPPED
        assert tsm.auto_buy_enabled is False
        assert tsm.auto_sell_enabled is False

    def test_start_from_stopped(self, tsm):
        tsm.start()
        assert tsm.state is TradingState.RUNNING

    def test_start_from_running_is_noop(self, tsm, caplog):
        tsm.start()
        with caplog.at_level(logging.WARNING):
            tsm.start()
        assert tsm.state is TradingState.RUNNING
        assert "非法迁移" in caplog.text

    def test_pause_from_running(self, tsm):
        tsm.start()
        tsm.pause()
        assert tsm.state is TradingState.PAUSED

    def test_pause_from_stopped_is_noop(self, tsm, caplog):
        with caplog.at_level(logging.WARNING):
            tsm.pause()
        assert tsm.state is TradingState.STOPPED
        assert "非法迁移" in caplog.text

    def test_resume_from_paused(self, tsm):
        tsm.start()
        tsm.pause()
        tsm.resume()
        assert tsm.state is TradingState.RUNNING

    def test_resume_from_running_is_noop(self, tsm, caplog):
        tsm.start()
        with caplog.at_level(logging.WARNING):
            tsm.resume()
        assert tsm.state is TradingState.RUNNING

    def test_start_from_paused(self, tsm):
        tsm.start()
        tsm.pause()
        tsm.start()
        assert tsm.state is TradingState.RUNNING

    def test_stop_all_from_running(self, tsm):
        tsm.start()
        tsm.stop_all()
        assert tsm.state is TradingState.STOPPED

    def test_stop_all_from_paused(self, tsm):
        tsm.start()
        tsm.pause()
        tsm.stop_all()
        assert tsm.state is TradingState.STOPPED

    def test_stop_all_from_stopped_is_noop(self, tsm, caplog):
        with caplog.at_level(logging.WARNING):
            tsm.stop_all()
        assert tsm.state is TradingState.STOPPED

    def test_stop_all_invokes_callback(self):
        calls = []
        tsm = TradingStateMachine(on_stop_all=lambda: calls.append(1))
        tsm.start()
        tsm.stop_all()
        assert calls == [1]

    def test_stop_all_callback_not_called_on_invalid(self):
        calls = []
        tsm = TradingStateMachine(on_stop_all=lambda: calls.append(1))
        tsm.stop_all()  # STOPPED → no-op
        assert calls == []


class TestIndependentFlags:
    """自动买/卖独立开关."""

    def test_set_auto_buy_does_not_touch_sell(self, tsm):
        tsm.set_auto_buy(True)
        assert tsm.auto_buy_enabled is True
        assert tsm.auto_sell_enabled is False

    def test_set_auto_sell_does_not_touch_buy(self, tsm):
        tsm.set_auto_sell(True)
        assert tsm.auto_sell_enabled is True
        assert tsm.auto_buy_enabled is False

    def test_toggle_off(self, tsm):
        tsm.set_auto_buy(True)
        tsm.set_auto_buy(False)
        assert tsm.auto_buy_enabled is False


class TestCanExecute:
    """can_execute(side) = RUNNING 且对应开关."""

    def test_stopped_never_executes(self, tsm):
        tsm.set_auto_buy(True)
        tsm.set_auto_sell(True)
        assert tsm.can_execute("buy") is False
        assert tsm.can_execute("sell") is False

    def test_paused_never_executes(self, tsm):
        tsm.set_auto_buy(True)
        tsm.start()
        tsm.pause()
        assert tsm.can_execute("buy") is False

    def test_running_buy_only(self, tsm):
        tsm.set_auto_buy(True)
        tsm.start()
        assert tsm.can_execute("buy") is True
        assert tsm.can_execute("sell") is False

    def test_running_sell_only(self, tsm):
        tsm.set_auto_sell(True)
        tsm.start()
        assert tsm.can_execute("sell") is True
        assert tsm.can_execute("buy") is False

    def test_running_both(self, tsm):
        tsm.set_auto_buy(True)
        tsm.set_auto_sell(True)
        tsm.start()
        assert tsm.can_execute("buy") is True
        assert tsm.can_execute("sell") is True

    def test_unknown_side(self, tsm):
        tsm.start()
        assert tsm.can_execute("hold") is False
