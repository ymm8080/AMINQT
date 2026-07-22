# -*- coding: utf-8 -*-
"""factor_discovery 单元测试 (ARCH §5.5).

环境: lightgbm / shap 均缺失 → 走 sklearn GBM + permutation_importance 兜底。
"""

import numpy as np
import pandas as pd
import pytest

from app.core.factor_discovery import (
    FORCE_TOP_FACTORS,
    FactorDiscovery,
)


@pytest.fixture
def dataset():
    """构造 5 因子数据集: f_good 与 y 强相关, 其余纯噪声.

    含 FORCE_TOP_FACTORS 之一的 tech_ths_ctrl_ratio (故意做成弱因子)。
    """
    rng = np.random.default_rng(42)
    n = 240
    dates = pd.bdate_range("2025-07-01", periods=n)
    signal = rng.normal(0, 1, n)
    y = pd.Series(0.8 * signal + 0.2 * rng.normal(0, 1, n), index=dates)
    X = pd.DataFrame(
        {
            "f_good": signal + 0.05 * rng.normal(0, 1, n),
            "f_noise1": rng.normal(0, 1, n),
            "f_noise2": rng.normal(0, 1, n),
            "f_noise3": rng.normal(0, 1, n),
            "tech_ths_ctrl_ratio": rng.normal(0, 1, n),  # 弱因子, 测试强制保留
        },
        index=dates,
    )
    return X, y


class TestIC:
    def test_perfect_correlation(self):
        fd = FactorDiscovery()
        x = pd.Series(np.arange(50, dtype=float))
        y = pd.Series(np.arange(50, dtype=float) * 2 + 1)
        assert fd.compute_ic(x, y) == pytest.approx(1.0, abs=1e-9)

    def test_perfect_negative(self):
        fd = FactorDiscovery()
        x = pd.Series(np.arange(50, dtype=float))
        y = pd.Series(-np.arange(50, dtype=float))
        assert fd.compute_ic(x, y) == pytest.approx(-1.0, abs=1e-9)

    def test_constant_factor_zero(self):
        fd = FactorDiscovery()
        x = pd.Series(np.ones(50))
        y = pd.Series(np.arange(50, dtype=float))
        assert fd.compute_ic(x, y) == 0.0

    def test_too_few_samples(self):
        fd = FactorDiscovery()
        assert fd.compute_ic(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])) == 0.0

    def test_known_signal_highest_ic(self, dataset):
        X, y = dataset
        fd = FactorDiscovery()
        ics = {c: fd.compute_ic(X[c], y) for c in X.columns}
        assert max(ics, key=ics.get) == "f_good"
        assert ics["f_good"] > 0.5


class TestICIR:
    def test_strong_factor_icir_positive(self, dataset):
        X, y = dataset
        fd = FactorDiscovery()
        icir_good = fd.compute_icir(X["f_good"], y)
        icir_noise = fd.compute_icir(X["f_noise1"], y)
        assert icir_good > 0
        assert icir_good > abs(icir_noise)

    def test_short_series_zero(self):
        fd = FactorDiscovery()
        x = pd.Series(np.arange(5, dtype=float))
        y = pd.Series(np.arange(5, dtype=float))
        assert fd.compute_icir(x, y) == 0.0


class TestRun:
    def test_report_schema_and_order(self, dataset):
        X, y = dataset
        fd = FactorDiscovery()
        report = fd.run(X, y)
        assert list(report.columns) == [
            "factor",
            "lgbm_gain",
            "shap_mean",
            "ic",
            "icir",
            "composite_score",
            "rank",
        ]
        assert len(report) == X.shape[1]
        # 按 composite_score 降序, rank 从 1 连续
        assert report["composite_score"].is_monotonic_decreasing
        assert list(report["rank"]) == list(range(1, len(report) + 1))
        assert np.isfinite(
            report[
                ["lgbm_gain", "shap_mean", "ic", "icir", "composite_score"]
            ].to_numpy()
        ).all()

    def test_good_factor_ranks_top(self, dataset):
        X, y = dataset
        report = FactorDiscovery().run(X, y)
        row = report.set_index("factor").loc["f_good"]
        assert row["ic"] == pytest.approx(report["ic"].max(), rel=1e-9)
        # 强信号因子应进 Top-2
        assert row["rank"] <= 2

    def test_fallback_without_lightgbm_and_shap(self, dataset):
        """lightgbm/shap 缺失环境下 run() 不抛异常 (走兜底路径)."""
        X, y = dataset
        fd = FactorDiscovery()
        report = fd.run(X, y)  # 不抛 ImportError 即通过
        assert not report.empty

    def test_empty_raises(self):
        fd = FactorDiscovery()
        with pytest.raises(ValueError):
            fd.run(pd.DataFrame(), pd.Series(dtype=float))


class TestTopFactors:
    def test_top_k_and_force_include(self, dataset):
        X, y = dataset
        fd = FactorDiscovery()
        report = fd.run(X, y)
        top = fd.get_top_factors(report, top_k=3)
        assert len(top) >= 3
        # FORCE_TOP_FACTORS 强制保留 (即使它是弱因子)
        for forced in FORCE_TOP_FACTORS:
            assert forced in top
        # 强因子应在 Top-K 内
        assert "f_good" in top[:3]

    def test_force_factor_absent_from_report(self, dataset):
        X, y = dataset
        fd = FactorDiscovery()
        report = fd.run(X, y)
        report = report[report["factor"] != "tech_ths_ctrl_ratio"]
        top = fd.get_top_factors(report, top_k=3)
        # 报告中不存在该因子时不强行加入
        assert "tech_ths_ctrl_ratio" not in top

    def test_empty_report(self):
        fd = FactorDiscovery()
        assert fd.get_top_factors(pd.DataFrame()) == FORCE_TOP_FACTORS
