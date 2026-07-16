# -*- coding: utf-8 -*-
"""Tests for app/core/factor_engine — build_features + safe_divide (Phase 2)."""

import pytest

pytest.skip("factor_engine tests — implement in Phase 2", allow_module_level=True)

# TODO: test safe_divide returns 0 where denominator is 0 (no inf).
# TODO: test build_features output X.shape[1]==20, X.shape[2]>=25.
# TODO: test no future-function leakage (indicator at t independent of t+k).
