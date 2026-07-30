# -*- coding: utf-8 -*-
"""Tests for FeatureRegistry (P19 auto-adoption)."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.feature_registry import FeatureRegistry


class TestFeatureRegistryCore:
    """Load / save / basic queries."""

    def test_init_empty(self):
        """New registry starts with empty features, adoption disabled."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        assert reg.features == {}
        assert reg.summary()["total_features"] == 0
        assert not reg.is_adoption_enabled()

    def test_save_load_roundtrip(self):
        """Save and reload preserves all data."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            reg = FeatureRegistry(path=path)
            reg.register_new("test_feature", {
                "dim_group": "dim01_price_volume",
                "active": True,
                "grade": "strong",
                "icir": 0.25,
                "ic_abs": 0.04,
                "source_cols": ["close_hfq"],
                "transform": "macd",
            })
            reg.save()

            # Reload
            reg2 = FeatureRegistry(path=path)
            assert "test_feature" in reg2.features
            meta = reg2.get_meta("test_feature")
            assert meta["dim_group"] == "dim01_price_volume"
            assert meta["grade"] == "strong"
            assert meta["icir"] == 0.25

    def test_register_new_upsert(self):
        """register_new() preserves created date on update."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f1", {"dim_group": "dim01", "grade": "strong"})
        created = reg.get_meta("f1")["created"]
        # Update
        reg.register_new("f1", {"grade": "weak", "icir": 0.15})
        assert reg.get_meta("f1")["grade"] == "weak"
        assert reg.get_meta("f1")["created"] == created  # Preserved
        assert reg.get_meta("f1")["icir"] == 0.15

    def test_get_active_all(self):
        """get_active() returns only active features."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f_active", {"active": True, "dim_group": "dim01"})
        reg.register_new("f_dead", {"active": False, "dim_group": "dim01"})
        reg.register_new("f_active2", {"active": True, "dim_group": "dim02"})

        all_active = reg.get_active()
        assert sorted(all_active) == ["f_active", "f_active2"]

        dim01_active = reg.get_active("dim01")
        assert dim01_active == ["f_active"]

        dim02_active = reg.get_active("dim02")
        assert dim02_active == ["f_active2"]

        dim03_active = reg.get_active("dim03")
        assert dim03_active == []

    def test_has_dim_group(self):
        """has_dim_group() returns True iff dim has >=1 active feature."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f1", {"active": True, "dim_group": "dim01"})
        reg.register_new("f2", {"active": False, "dim_group": "dim02"})

        assert reg.has_dim_group("dim01") is True
        assert reg.has_dim_group("dim02") is False  # All inactive
        assert reg.has_dim_group("dim03") is False  # Not registered

    def test_deactivate_activate(self):
        """deactivate/activate toggles active flag."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f1", {"active": True, "dim_group": "dim01"})
        assert reg.is_active("f1")

        reg.deactivate(["f1"])
        assert not reg.is_active("f1")
        assert not reg.has_dim_group("dim01")

        reg.activate(["f1"])
        assert reg.is_active("f1")
        assert reg.has_dim_group("dim01")

    def test_is_active_unknown_returns_true(self):
        """Unknown features default to active (safe default)."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        assert reg.is_active("nonexistent") is True

    def test_prune_stale(self):
        """prune_stale deactivates dead features."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f_good", {"active": True, "grade": "strong"})
        reg.register_new("f_bad", {"active": True, "grade": "dead"})
        reg.register_new("f_bad2", {"active": True, "grade": "dead"})

        n = reg.prune_stale()
        assert n == 2
        assert not reg.is_active("f_bad")
        assert not reg.is_active("f_bad2")
        assert reg.is_active("f_good")

    def test_summary_counts(self):
        """summary() returns correct grade distribution."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        for i in range(5):
            reg.register_new(f"strong_{i}", {"grade": "strong", "active": True})
        for i in range(3):
            reg.register_new(f"weak_{i}", {"grade": "weak", "active": True})
        for i in range(2):
            reg.register_new(f"dead_{i}", {"grade": "dead", "active": False})

        s = reg.summary()
        assert s["total_features"] == 10
        assert s["active"] == 8
        assert s["inactive"] == 2
        assert s["by_grade"]["strong"] == 5
        assert s["by_grade"]["weak"] == 3
        assert s["by_grade"]["dead"] == 2


class TestFeatureRegistryUpdateFromScreen:
    """ICScreener → Registry sync."""

    def test_update_from_screen_basic(self):
        """Standard screen result updates grades and active flags."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("MACD", {"dim_group": "dim01", "active": True, "grade": "unknown"})
        reg.register_new("RSI", {"dim_group": "dim01", "active": True, "grade": "unknown"})
        reg.register_new("dead_factor", {"dim_group": "dim02", "active": True, "grade": "unknown"})

        screen_result = {
            "window_id": "main_2026W31",
            "factors": ["MACD", "RSI"],
            "detail": {
                "MACD": {"ic_1d": 0.045, "ic_3d": 0.038, "ic_5d": 0.032,
                         "icir": 0.28, "grade": "strong",
                         "rolling_mean": 0.035, "rolling_pos_ratio": 0.72},
                "RSI": {"ic_1d": 0.018, "ic_3d": 0.022, "ic_5d": 0.015,
                        "icir": 0.08, "grade": "weak",
                        "rolling_mean": 0.012, "rolling_pos_ratio": 0.55},
                "dead_factor": {"ic_1d": 0.003, "ic_3d": 0.002, "ic_5d": 0.001,
                                "icir": 0.01, "grade": "dead",
                                "rolling_mean": 0.002, "rolling_pos_ratio": 0.30},
            },
        }
        reg.update_from_screen(screen_result, "main_2026W31")

        assert reg.get_meta("MACD")["grade"] == "strong"
        assert reg.get_meta("MACD")["active"] is True
        assert reg.get_meta("MACD")["icir"] == 0.28
        assert reg.get_meta("MACD")["ic_abs"] == pytest.approx(0.045)
        assert reg.get_meta("MACD")["window_birth"] == "main_2026W31"

        assert reg.get_meta("RSI")["grade"] == "weak"
        assert reg.get_meta("RSI")["active"] is True

        assert reg.get_meta("dead_factor")["grade"] == "dead"
        assert reg.get_meta("dead_factor")["active"] is False

    def test_update_from_screen_ic_abs_takes_max(self):
        """ic_abs = max(|ic_1d|, |ic_3d|, |ic_5d|)."""
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.register_new("f1", {"dim_group": "dim01"})

        screen_result = {
            "window_id": "test",
            "factors": ["f1"],
            "detail": {"f1": {"ic_1d": -0.010, "ic_3d": 0.055, "ic_5d": -0.020,
                              "icir": 0.15, "grade": "strong",
                              "rolling_mean": 0.02, "rolling_pos_ratio": 0.60}},
        }
        reg.update_from_screen(screen_result, "test")
        assert reg.get_meta("f1")["ic_abs"] == pytest.approx(0.055)


class TestFeatureRegistryAdoption:
    """Auto-adoption enable/disable and source col tracking."""

    def test_enable_disable(self):
        reg = FeatureRegistry(path="/nonexistent/test.json")
        assert not reg.is_adoption_enabled()

        reg.enable_adoption()
        assert reg.is_adoption_enabled()

        reg.disable_adoption()
        assert not reg.is_adoption_enabled()

    def test_mark_source_cols(self):
        reg = FeatureRegistry(path="/nonexistent/test.json")
        reg.mark_source_cols_registered(["eps", "ocfps"])
        assert reg.get_registered_source_cols() == {"eps", "ocfps"}

        # Duplicate add
        reg.mark_source_cols_registered(["eps", "bps"])
        assert reg.get_registered_source_cols() == {"eps", "ocfps", "bps"}


class TestFeatureRegistrySeed:
    """_seed() auto-discovers features from FeatureEngine."""

    def test_seed_from_minimal_panel(self):
        """Seed on a small synthetic panel produces valid registry."""
        import numpy as np
        np.random.seed(42)
        n_symbols = 3
        n_dates = 50
        symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
        dates = pd.bdate_range("2025-01-01", periods=n_dates)
        rows = []
        for sym in symbols:
            for d in dates:
                base_price = 10 + hash(sym) % 50
                rows.append({
                    "symbol": sym, "date": d,
                    "open": base_price, "high": base_price * 1.02,
                    "low": base_price * 0.98, "close": base_price * 1.01,
                    "open_hfq": base_price, "high_hfq": base_price * 1.02,
                    "low_hfq": base_price * 0.98, "close_hfq": base_price * 1.01,
                    "volume": 1e6, "amount": 1e7, "turnover_rate": 2.0,
                    "pre_close": base_price * 0.99,
                    "board": "main", "industry": "银行", "is_st": 0,
                    "is_suspended": 0, "list_days": 500,
                })
        panel = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            reg = FeatureRegistry(path=path)
            n = reg._seed(panel)

            assert n > 100  # Should discover many features
            assert reg.has_dim_group("dim01_price_volume")
            assert reg.has_dim_group("dim12_ma_system")
            s = reg.summary()
            assert s["total_features"] == n
            assert s["dim_groups"] >= 20
            # All features active after seed
            assert s["active"] == n

    def test_seed_produces_valid_registry(self):
        """Seed result can be loaded back."""
        import numpy as np
        np.random.seed(99)
        symbols = ["000001", "000002"]
        dates = pd.bdate_range("2025-01-01", periods=30)
        rows = []
        for sym in symbols:
            for d in dates:
                base_price = 10 + hash(sym) % 50
                rows.append({
                    "symbol": sym, "date": d,
                    "open": base_price, "high": base_price * 1.02,
                    "low": base_price * 0.98, "close": base_price * 1.01,
                    "open_hfq": base_price, "high_hfq": base_price * 1.02,
                    "low_hfq": base_price * 0.98, "close_hfq": base_price * 1.01,
                    "volume": 1e6, "amount": 1e7, "turnover_rate": 2.0,
                    "pre_close": base_price * 0.99,
                    "board": "main", "industry": "银行", "is_st": 0,
                    "is_suspended": 0, "list_days": 500,
                })
        panel = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "registry.json")
            reg = FeatureRegistry(path=path)
            reg._seed(panel)
            reg.save()

            # Reload
            reg2 = FeatureRegistry(path=path)
            assert reg2.summary()["total_features"] == reg.summary()["total_features"]
            # All features loaded
            for name in reg.features:
                assert name in reg2.features
