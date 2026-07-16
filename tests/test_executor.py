# -*- coding: utf-8 -*-
"""Tests for services/executor — mode toggle + T+1 (M3 / Phase 6)."""
import pytest

pytest.skip("executor tests — implement in M3 / Phase 6",
            allow_module_level=True)

# TODO: test MANUAL mode execute() returns executed=False (recommendation only).
# TODO: test AUTO mode calls _place().
# TODO: test XtExecutor.sync_portfolio blocks selling today-bought symbols (T+1).
# TODO: test SimExecutor._place prints [SIM] order.
