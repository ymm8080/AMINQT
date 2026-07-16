# -*- coding: utf-8 -*-
"""Tests for data/adapters — canonicalize, factory fallback (Phase 1+)."""

import pytest

pytest.skip(
    "data adapter tests — implement after deps installed", allow_module_level=True
)

# TODO: test DataAdapter._canonicalize renames 日期→date etc.
# TODO: test get_adapter('akshare') returns AkshareAdapter.
# TODO: test get_adapter('ifind') falls back to akshare when iFinDPy missing.
