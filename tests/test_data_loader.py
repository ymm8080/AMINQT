# -*- coding: utf-8 -*-
"""Tests for app/core/data_loader — CSV load + canonicalize (Phase 1+)."""

import pytest

pytest.skip(
    "data_loader tests — implement after download_data.py run", allow_module_level=True
)

# TODO: write a sample CSV with Chinese cols, assert load_csv renames + sorts.
# TODO: test load_all skips missing symbols without crashing.
