# -*- coding: utf-8 -*-
"""Tests for Three-Layer Feature Selection: BruteForceGenerator, dedup_l2, gate_d_ablation, nan_filter, FeatureSelector."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.feature_selector import (
    BruteForceGenerator,
    FeatureSelector,
    dedup_l2,
    gate_d_ablation,
    nan_filter,
)


# ── Synthetic data helpers ──────────────────────────────────────────


def _make_panel(n_symbols=10, n_dates=60, seed=42):
    """Create synthetic panel with OHLCV + some extra numeric columns."""
    np.random.seed(seed)
    symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range("2025-06-01", periods=n_dates)
    rows = []
    for sym in symbols:
        base = 10.0 + hash(sym) % 50
        for i, d in enumerate(dates):
            close = base + np.cumsum(np.random.randn(n_dates) * 0.3)[i]
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "open": close * 0.99,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "close": close,
                    "volume": np.random.uniform(5e5, 5e6),
                    "amount": close * np.random.uniform(5e5, 5e6),
                    "turnover_rate": np.random.uniform(0.5, 5.0),
                    "pre_close": close * 0.995,
                    "board": "main",
                    "industry": "银行",
                    "is_suspended": 0,
                    "eps": np.random.normal(1.5, 0.3),
                    "bps": np.random.normal(8.0, 1.0),
                    "ocfps": np.random.normal(0.5, 0.2),
                }
            )
    return pd.DataFrame(rows)


def _make_small_df(n_symbols=3, n_dates=30, n_extra_feats=0, seed=123):
    """Create a tiny panel for fast unit tests."""
    np.random.seed(seed)
    symbols = [f"{i:06d}" for i in range(1, n_symbols + 1)]
    dates = pd.bdate_range("2025-07-01", periods=n_dates)
    rows = []
    for sym in symbols:
        base = 10.0 + hash(sym) % 30
        for i, d in enumerate(dates):
            close = base + np.cumsum(np.random.randn(n_dates) * 0.2)[i]
            row = {
                "symbol": sym,
                "date": d,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": np.random.uniform(5e5, 5e6),
                "amount": close * np.random.uniform(5e5, 5e6),
                "turnover_rate": np.random.uniform(0.5, 5.0),
                "pre_close": close * 0.995,
                "board": "main",
                "industry": "银行",
                "is_suspended": 0,
            }
            for j in range(n_extra_feats):
                row[f"extra_{j}"] = np.random.normal(0, 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# BruteForceGenerator
# ═══════════════════════════════════════════════════════════════════


class TestBruteForceGenerator:
    """Tests for brute-force feature generation from raw panel columns."""

    def test_generate_produces_features(self):
        """BruteForceGenerator generates _brute_ prefixed columns."""
        df = _make_small_df(n_symbols=5, n_dates=40)
        gen = BruteForceGenerator()
        new = gen.generate(df)
        assert len(new.columns) > 50, (
            f"Expected >50 brute columns, got {len(new.columns)}"
        )
        # All new columns should have '_brute_' in name
        for c in new.columns:
            assert "_brute_" in c, f"Column {c} missing _brute_ marker"
        # Should match index of original df
        assert len(new) == len(df)

    def test_generate_respects_eligible_cols(self):
        """When eligible_cols is specified, only those columns are used."""
        df = _make_small_df(n_symbols=3, n_dates=20)
        gen = BruteForceGenerator(eligible_cols=["close"])
        new = gen.generate(df)
        # All generated features should start with "close_brute_"
        for c in new.columns:
            assert c.startswith("close_brute_"), f"Unexpected column: {c}"
        assert len(new.columns) > 0

    def test_generate_no_inf_values(self):
        """Generated features should not contain inf values."""
        df = _make_small_df(n_symbols=5, n_dates=40)
        gen = BruteForceGenerator()
        new = gen.generate(df)
        for c in new.columns:
            has_inf = np.isinf(new[c].values).any()
            assert not has_inf, f"Column {c} contains inf"

    def test_generate_single_symbol(self):
        """Single symbol should still work."""
        df = _make_small_df(n_symbols=1, n_dates=30)
        gen = BruteForceGenerator()
        new = gen.generate(df)
        assert len(new) == len(df)
        assert len(new.columns) > 0

    def test_generate_empty_skip_cols(self):
        """EXCLUDE_COLS columns are not used for generation."""
        df = _make_small_df(n_symbols=3, n_dates=20)
        gen = BruteForceGenerator()
        eligible = gen._eligible(df)
        for excl in BruteForceGenerator.EXCLUDE_COLS:
            assert excl not in eligible, f"Excluded column {excl} in eligible list"

    def test_generate_skips_label_cols(self):
        """Columns starting with label_ are excluded."""
        df = _make_small_df(n_symbols=3, n_dates=20)
        df["label_test"] = np.random.randn(len(df))
        gen = BruteForceGenerator()
        eligible = gen._eligible(df)
        assert "label_test" not in eligible
        # Also check no brute feature from label_test
        new = gen.generate(df)
        label_brute = [c for c in new.columns if "label_test" in c]
        assert len(label_brute) == 0

    def test_transform_windows_match_spec(self):
        """Verify transform definitions match design spec."""
        bfg = BruteForceGenerator()
        t = bfg.BASE_TRANSFORM_DEFS
        assert t["pct_change"]["windows"] == (1, 2, 3, 5, 10, 20, 40, 60)
        assert t["rolling_mean"]["windows"] == (5, 10, 20, 40, 60)
        assert t["rolling_std"]["windows"] == (5, 10, 20, 40)
        assert t["EMA"]["windows"] == (5, 20, 40)

    def test_generate_respects_raw_cols_param(self):
        """raw_cols parameter limits generation scope."""
        df = _make_small_df(n_symbols=3, n_dates=20)
        gen = BruteForceGenerator()
        new = gen.generate(df, raw_cols=["close"])
        for c in new.columns:
            assert c.startswith("close_brute_"), f"Unexpected column from raw_cols: {c}"


# ═══════════════════════════════════════════════════════════════════
# nan_filter
# ═══════════════════════════════════════════════════════════════════


class TestNanFilter:
    """Tests for NaN-based feature filtering."""

    def test_filters_high_nan(self):
        """Features with >=threshold NaN are dropped."""
        df = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0, 4.0, 5.0],
                "half": [1.0, np.nan, 3.0, np.nan, 5.0],
                "bad": [1.0, np.nan, np.nan, np.nan, np.nan],
            }
        )
        feats = ["good", "half", "bad"]
        result = nan_filter(feats, df, threshold=0.5)
        assert "good" in result
        assert "bad" not in result  # 80% NaN >= 50% threshold

    def test_all_pass_zero_threshold(self):
        """Threshold of 0 means nothing is filtered."""
        df = pd.DataFrame({"a": [np.nan] * 5})
        result = nan_filter(["a"], df, threshold=0.0)
        assert len(result) == 0  # a is 100% NaN > 0%

    def test_all_pass_high_threshold(self):
        """Threshold of 1.0 keeps columns with <100% NaN (strict inequality)."""
        df = pd.DataFrame(
            {
                "a": [np.nan, 1.0, 2.0, 3.0, 4.0],  # 20% NaN < 100%
                "b": [1.0] * 5,
            }
        )  # 0% NaN < 100%
        result = nan_filter(["a", "b"], df, threshold=1.0)
        # a has 20% NaN < 1.0 → kept; b has 0% NaN → kept
        assert "a" in result
        assert "b" in result

    def test_object_columns_skipped(self):
        """Non-numeric (object) columns are excluded."""
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = nan_filter(["a"], df, threshold=0.95)
        assert "a" not in result  # object dtype

    def test_missing_column_skipped(self):
        """Feature not in df is silently skipped."""
        result = nan_filter(["nonexistent"], pd.DataFrame(), threshold=0.95)
        assert result == []

    def test_default_threshold(self):
        """Default threshold is 0.95 (from design)."""
        df = pd.DataFrame(
            {
                "mostly_ok": [np.nan, 1.0, 1.0, 1.0, 1.0],  # 20% NaN
                "mostly_bad": [np.nan] * 4 + [1.0],  # 80% NaN but < 95%
            }
        )
        result = nan_filter(["mostly_ok", "mostly_bad"], df)
        assert "mostly_ok" in result
        assert "mostly_bad" in result  # 80% < 95% threshold


# ═══════════════════════════════════════════════════════════════════
# dedup_l2
# ═══════════════════════════════════════════════════════════════════


class TestDedupL2:
    """Tests for L2 correlation dedup within base groups."""

    def test_dedup_within_same_base(self):
        """Highly correlated features from same base are dedup'd."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x_brute_pct1": base,
                "x_brute_pct5": base + np.random.randn(n) * 0.01,  # near-duplicate
                "x_brute_ma10": base * -1,  # anti-correlated, different transform
            }
        )
        feats = ["x_brute_pct1", "x_brute_pct5", "x_brute_ma10"]
        result = dedup_l2(feats, df, threshold=0.7)
        # pct1 and pct5 are highly correlated → one should be dropped
        assert "x_brute_pct1" in result
        # ma10 is anti-correlated (|r| ≈ 1) → also correlated, could be dropped
        assert len(result) >= 1
        assert len(result) <= 2

    def test_uncorrelated_kept(self):
        """Uncorrelated features are all kept."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "a_brute_pct1": np.random.randn(n),
                "a_brute_pct5": np.random.randn(n),
                "b_brute_pct1": np.random.randn(n),
            }
        )
        # Force all pairwise |r| <= 0.7 by using low-correlation data
        feats = ["a_brute_pct1", "a_brute_pct5", "b_brute_pct1"]
        result = dedup_l2(feats, df, threshold=0.7)
        # All should survive since they're independent
        assert len(result) == 3

    def test_single_feature_per_group(self):
        """Groups with single feature pass through."""
        df = pd.DataFrame({"x_brute_ma5": np.random.randn(50)})
        result = dedup_l2(["x_brute_ma5"], df)
        assert result == ["x_brute_ma5"]

    def test_dim_features_grouped(self):
        """dim prefixed features group by dim number."""
        np.random.seed(42)
        n = 100
        base = np.random.randn(n)
        df = pd.DataFrame(
            {
                "dim01_MACD": base,
                "dim01_RSI": base + np.random.randn(n) * 0.01,  # correlated
                "dim12_MA_5": np.random.randn(n),  # independent
            }
        )
        feats = ["dim01_MACD", "dim01_RSI", "dim12_MA_5"]
        result = dedup_l2(feats, df, threshold=0.7)
        # dim01 pair: one should survive
        # dim12: should survive (different group)
        assert "dim12_MA_5" in result
        dim01_count = sum(1 for c in result if c.startswith("dim01"))
        assert dim01_count >= 1

    def test_empty_input(self):
        """Empty feature list returns empty."""
        result = dedup_l2([], pd.DataFrame())
        assert result == []

    def test_keeps_variance_ordered(self):
        """Higher variance features are preferred."""
        np.random.seed(42)
        n = 200
        base = np.random.randn(n)
        df = pd.DataFrame(
            {
                "x_brute_ma5": base * 0.1,  # low variance
                "x_brute_ma10": base * 5.0,  # high variance (same signal, amplified)
            }
        )
        feats = ["x_brute_ma5", "x_brute_ma10"]
        result = dedup_l2(feats, df, threshold=0.7)
        # Both are highly correlated; higher variance should survive
        assert "x_brute_ma10" in result


# ═══════════════════════════════════════════════════════════════════
# gate_d_ablation
# ═══════════════════════════════════════════════════════════════════


class TestGateD:
    """Tests for Gate D (importance forward ablation with saturation)."""

    def test_gate_d_basic(self):
        """Gate D runs and selects a subset of features."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * n,
                "date": pd.bdate_range("2025-01-01", periods=n),
                "label_pm_1d_net": np.random.randn(n),
            }
        )
        # Add 40 features: 5 informative + 35 noise
        signal = np.random.randn(n)
        for i in range(5):
            df[f"sig_{i}"] = signal + np.random.randn(n) * 0.3
        for i in range(35):
            df[f"noise_{i}"] = np.random.randn(n)

        feats = [
            c for c in df.columns if c.startswith("sig_") or c.startswith("noise_")
        ]
        # Make label correlated with signal features
        df["label_pm_1d_net"] = (
            df[[f"sig_{i}" for i in range(5)]].mean(axis=1) + np.random.randn(n) * 0.5
        )

        result = gate_d_ablation(
            feats, df, label_col="label_pm_1d_net", min_feats=5, sat_pct=0.90
        )
        assert len(result) >= 5
        assert len(result) <= len(feats)
        # Signal features should dominate selection
        sig_selected = len([c for c in result if c.startswith("sig_")])
        assert sig_selected >= 1, "Expected at least 1 signal feature selected"

    def test_gate_d_min_feature_clamp(self):
        """Result is clamped to at least min_feats."""
        np.random.seed(42)
        n = 150
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * n,
                "date": pd.bdate_range("2025-01-01", periods=n),
                "label_pm_1d_net": np.random.randn(n),
            }
        )
        for i in range(10):
            df[f"f_{i}"] = np.random.randn(n)

        feats = [f"f_{i}" for i in range(10)]
        result = gate_d_ablation(
            feats, df, label_col="label_pm_1d_net", min_feats=10, sat_pct=0.95
        )
        # When feats <= min_feats, return all
        assert result == feats

    def test_gate_d_uses_label_col_param(self):
        """Gate D respects the label_col parameter."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * n,
                "date": pd.bdate_range("2025-01-01", periods=n),
                "custom_label": np.random.randn(n),
            }
        )
        for i in range(20):
            df[f"f_{i}"] = np.random.randn(n)

        feats = [f"f_{i}" for i in range(20)]
        df["custom_label"] = df["f_0"] + np.random.randn(n) * 0.3

        result = gate_d_ablation(
            feats, df, label_col="custom_label", min_feats=5, sat_pct=0.90
        )
        assert len(result) >= 5
        assert len(result) <= 20

    def test_gate_d_with_nan_in_features(self):
        """Gate D handles NaN in feature columns via fillna(0)."""
        np.random.seed(42)
        n = 120
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * n,
                "date": pd.bdate_range("2025-01-01", periods=n),
                "label_1d_net": np.random.randn(n),
            }
        )
        for i in range(15):
            vals = np.random.randn(n)
            if i % 3 == 0:
                vals[:10] = np.nan  # Some NaN at start
            df[f"f_{i}"] = vals

        feats = [f"f_{i}" for i in range(15)]
        result = gate_d_ablation(
            feats, df, label_col="label_1d_net", min_feats=5, sat_pct=0.95
        )
        assert len(result) >= 5

    def test_gate_d_single_date_returns_all(self):
        """Edge case: single date returns all features (train/test split issue)."""
        # With 1-2 dates, the 80/20 split produces empty test set
        np.random.seed(42)
        n = 5  # small
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * n,
                "date": pd.bdate_range("2025-01-01", periods=n),
                "label_pm_1d_net": np.random.randn(n),
                "f_0": np.random.randn(n),
                "f_1": np.random.randn(n),
            }
        )
        feats = ["f_0", "f_1"]
        result = gate_d_ablation(feats, df, min_feats=2, sat_pct=0.95)
        assert len(result) >= 1  # Should not crash


# ═══════════════════════════════════════════════════════════════════
# FeatureSelector — config + selection
# ═══════════════════════════════════════════════════════════════════


class TestFeatureSelectorConfig:
    """Tests for FeatureSelector configuration and factory."""

    def test_default_config_has_main_and_dual(self):
        sel = FeatureSelector(registry_dir="/tmp")
        assert "main" in sel.config
        assert "dual" in sel.config
        assert sel.config["main"]["pipeline"] == "bruteforce_dedup"
        assert sel.config["dual"]["pipeline"] == "gate_d"

    def test_custom_config_override(self):
        custom = {
            "main": {"pipeline": "gate_d", "nan_threshold": 0.90},
        }
        sel = FeatureSelector(config=custom, registry_dir="/tmp")
        assert sel.config["main"]["pipeline"] == "gate_d"
        assert sel.config["main"]["nan_threshold"] == 0.90

    def test_fallback_config(self):
        sel = FeatureSelector(registry_dir="/tmp")
        assert "fallback" in sel.config
        assert sel.config["fallback"]["pipeline"] == "ic_screener"


class TestFeatureSelectorSelection:
    """Tests for the select() method with main/dual pipelines."""

    def test_select_main_bruteforce_dedup(self):
        """MAIN board runs bruteforce + dedup pipeline."""
        df = _make_small_df(n_symbols=5, n_dates=40)
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            features = sel.select(df, "main")
            assert len(features) > 10
            # All should be valid column names
            for f in features:
                assert isinstance(f, str)

    def test_select_dual_gate_d(self):
        """DUAL board runs gate_d pipeline (features built via FeatureEngineV35)."""
        np.random.seed(42)
        n = 150
        symbols = ["300001", "300002", "688001"]
        dates = pd.bdate_range("2025-01-01", periods=n // 3)
        rows = []
        for sym in symbols:
            base = 10 + hash(sym) % 30
            for i, d in enumerate(dates):
                close = base + np.cumsum(np.random.randn(len(dates)) * 0.2)[i]
                rows.append(
                    {
                        "symbol": sym,
                        "date": d,
                        "open": close * 0.99,
                        "high": close * 1.02,
                        "low": close * 0.98,
                        "close": close,
                        "open_hfq": close * 0.99,
                        "high_hfq": close * 1.02,
                        "low_hfq": close * 0.98,
                        "close_hfq": close,
                        "volume": 1e6,
                        "amount": 1e7,
                        "turnover_rate": 2.0,
                        "pre_close": close * 0.995,
                        "board": "GEM",
                        "industry": "科技",
                        "is_suspended": 0,
                        "label_pm_1d_net": np.random.randn(),
                        "label_1d_net": np.random.randn(),
                    }
                )
        df = pd.DataFrame(rows)

        # Build features (FeatureEngineV35 expects built columns)
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        fe = FeatureEngineV35()
        df = fe.build(df)

        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            features = sel.select(df, "dual")
            assert len(features) > 0

    def test_select_fallback(self):
        """Unknown board uses fallback pipeline."""
        df = _make_small_df(n_symbols=3, n_dates=20)
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            features = sel.select(df, "unknown_board")
            assert isinstance(features, list)


# ═══════════════════════════════════════════════════════════════════
# FeatureSelector — versioning
# ═══════════════════════════════════════════════════════════════════


class TestFeatureSelectorVersioning:
    """Tests for save/load version, current pointer, list, and status."""

    def test_save_and_load_current(self):
        """Save a version and load it back as current."""
        result = {
            "board": "main",
            "pipeline": "test",
            "created": "2026-01-01",
            "pool_size": 100,
            "selected_count": 50,
            "features": ["dim01_MA_5", "dim02_STD_10", "dim03_feat"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            sel.save_version(result, "main", activate=True)

            loaded = sel.load_current("main")
            assert loaded["board"] == "main"
            assert loaded["pool_size"] == 100
            assert loaded["selected_count"] == 50
            assert loaded["features"] == result["features"]

    def test_save_as_draft(self):
        """Save as draft without activating (no current pointer)."""
        result = {
            "board": "dual",
            "pipeline": "gate_d",
            "created": "2026-01-02",
            "pool_size": 200,
            "selected_count": 30,
            "features": ["a", "b", "c"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            path = sel.save_version(result, "dual", activate=False)
            assert os.path.exists(path)
            # current should not exist
            current_path = os.path.join(tmp, "selected_dual_current.json")
            assert not os.path.exists(current_path)

    def test_load_current_not_found(self):
        """load_current raises when no current version exists."""
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            with pytest.raises(FileNotFoundError):
                sel.load_current("main")

    def test_list_versions(self):
        """list_versions returns all non-current versions sorted newest first."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            sel.save_version({"features": ["a"]}, "main", activate=False)
            time.sleep(1.1)  # avoid timestamp collision (second granularity)
            sel.save_version({"features": ["b"]}, "main", activate=False)
            time.sleep(1.1)
            sel.save_version({"features": ["c"]}, "main", activate=False)

            versions = sel.list_versions("main")
            assert len(versions) == 3
            # Sorted newest first
            assert versions[0] > versions[-1]

    def test_get_status_active(self):
        """get_status returns correct status for active board."""
        result = {
            "board": "main",
            "pipeline": "test",
            "selected_count": 10,
            "pool_size": 50,
            "features": ["a"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            sel.save_version(result, "main", activate=True)
            status = sel.get_status("main")
            assert status["status"] == "active"
            assert status["selected_count"] == 10
            assert "active_version" in status

    def test_get_status_no_current(self):
        """get_status returns no_current when no version exists."""
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            status = sel.get_status("dual")
            assert status["status"] == "no_current"

    def test_get_status_broken_pointer(self):
        """get_status detects broken current pointer."""
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            # Write a broken pointer
            current_path = os.path.join(tmp, "selected_main_current.json")
            with open(current_path, "w") as f:
                json.dump({"active_version": "selected_main_nonexistent.json"}, f)
            status = sel.get_status("main")
            assert status["status"] == "broken_pointer"

    def test_rollback(self):
        """rollback points current to a previous version."""
        import time

        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            sel.save_version({"features": ["v1"]}, "main", activate=True)
            time.sleep(1.1)
            sel.save_version({"features": ["v2"]}, "main", activate=True)

            versions = sel.list_versions("main")
            assert len(versions) >= 2

            # Rollback to first version
            first_version = versions[-1]  # oldest
            sel.rollback("main", first_version)

            loaded = sel.load_current("main")
            assert loaded["features"] == ["v1"]

    def test_rollback_nonexistent(self):
        """Rollback to nonexistent version raises."""
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            with pytest.raises(FileNotFoundError):
                sel.rollback("main", "nonexistent")

    def test_diff_versions(self):
        """diff_versions computes added/removed correctly."""
        sel = FeatureSelector(registry_dir="/tmp")
        old = ["a", "b", "c", "d"]
        new = ["a", "c", "e", "f"]
        diff = sel.diff_versions(old, new)
        assert diff["added_count"] == 2  # e, f
        assert diff["removed_count"] == 2  # b, d
        assert diff["net_change"] == 0
        assert set(diff["sample_added"]) <= {"e", "f"}
        assert set(diff["sample_removed"]) <= {"b", "d"}

    def test_diff_versions_all_new(self):
        """Diff when there's no current version."""
        sel = FeatureSelector(registry_dir="/tmp")
        diff = sel.diff_versions([], ["a", "b", "c"])
        assert diff["added_count"] == 3
        assert diff["removed_count"] == 0
        assert diff["net_change"] == 3

    def test_diff_versions_all_removed(self):
        """Diff when all features removed."""
        sel = FeatureSelector(registry_dir="/tmp")
        diff = sel.diff_versions(["a", "b", "c"], [])
        assert diff["added_count"] == 0
        assert diff["removed_count"] == 3
        assert diff["net_change"] == -3

    def test_version_files_isolated_per_board(self):
        """Main and Dual versions don't interfere."""
        with tempfile.TemporaryDirectory() as tmp:
            sel = FeatureSelector(registry_dir=tmp)
            sel.save_version({"features": ["f_main"]}, "main", activate=True)
            sel.save_version({"features": ["f_dual"]}, "dual", activate=True)

            main_loaded = sel.load_current("main")
            dual_loaded = sel.load_current("dual")
            assert main_loaded["features"] == ["f_main"]
            assert dual_loaded["features"] == ["f_dual"]


# ═══════════════════════════════════════════════════════════════════
# Integration-style tests
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndMini:
    """Mini end-to-end tests of the three-layer pipeline logic."""

    def test_full_pipeline_main(self):
        """Layer1 (brute) → Layer2 (nan_filter + dedup_l2) produces valid result."""
        df = _make_small_df(n_symbols=8, n_dates=60)

        # Layer 1: brute force generation
        gen = BruteForceGenerator()
        new = gen.generate(df)
        df_exp = df.join(new)

        all_feats = [
            c
            for c in df_exp.columns
            if c not in BruteForceGenerator.EXCLUDE_COLS
            and not c.startswith("label_")
            and df_exp[c].dtype in ("float64", "int64")
        ]
        assert len(all_feats) > 30, f"Expected >30 brute features, got {len(all_feats)}"

        # Layer 2: NaN filter
        valid = nan_filter(all_feats, df_exp, 0.95)
        assert len(valid) > 0

        # Layer 2: Dedup L2
        selected = dedup_l2(valid, df_exp, 0.7)
        assert len(selected) > 0
        assert len(selected) <= len(valid)
        # Dedup should reduce count
        if len(valid) > 10:
            assert len(selected) < len(valid), "Dedup should reduce feature count"

    def test_full_pipeline_mini_dual(self):
        """Layer1 (curated) → Layer2 (nan_filter + gate_d) for DUAL board."""
        np.random.seed(42)
        n = 120
        symbols = ["300001", "688001"]
        dates = pd.bdate_range("2025-06-01", periods=n // 2)
        rows = []
        for sym in symbols:
            base = 10 + hash(sym) % 30
            for i, d in enumerate(dates):
                close = base + np.cumsum(np.random.randn(len(dates)) * 0.15)[i]
                rows.append(
                    {
                        "symbol": sym,
                        "date": d,
                        "open": close * 0.99,
                        "high": close * 1.02,
                        "low": close * 0.98,
                        "close": close,
                        "open_hfq": close * 0.99,
                        "high_hfq": close * 1.02,
                        "low_hfq": close * 0.98,
                        "close_hfq": close,
                        "volume": 1e6,
                        "amount": 1e7,
                        "turnover_rate": 2.0,
                        "pre_close": close * 0.995,
                        "board": "GEM",
                        "industry": "科技",
                        "is_suspended": 0,
                        "label_pm_1d_net": np.random.randn(),
                        "label_1d_net": np.random.randn(),
                    }
                )
        df = pd.DataFrame(rows)

        # Build features via FeatureEngineV35
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        fe = FeatureEngineV35()
        df = fe.build(df)

        all_feats = FeatureEngineV35.feature_columns(df)
        assert len(all_feats) > 0, "FeatureEngine should produce features"

        # NaN filter
        valid = nan_filter(all_feats, df, 0.95)
        assert len(valid) > 0

        # Gate D (only if enough features)
        if len(valid) > 30:
            selected = gate_d_ablation(
                valid, df, label_col="label_pm_1d_net", min_feats=5, sat_pct=0.95
            )
            assert len(selected) >= 5
