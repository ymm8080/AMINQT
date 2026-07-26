"""Tests for app/utils/daily_rank_ic.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.utils.daily_rank_ic import (
    cross_sectional_rank_ic,
    daily_rank_ic_series,
    icir,
    mean_rank_ic,
)


class TestCrossSectionalRankIC:
    """单截面 Rank IC."""

    def test_perfect_rank_ic(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 2, 3, 4, 5])
        assert cross_sectional_rank_ic(x, y) == pytest.approx(1.0)

    def test_inverse_rank_ic(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([5, 4, 3, 2, 1])
        assert cross_sectional_rank_ic(x, y) == pytest.approx(-1.0)

    def test_insufficient_unique_x_returns_nan(self):
        x = pd.Series([1, 1, 1, 1, 1])
        y = pd.Series([1, 2, 3, 4, 5])
        assert np.isnan(cross_sectional_rank_ic(x, y))

    def test_insufficient_unique_y_returns_nan(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([1, 1, 1, 1, 1])
        assert np.isnan(cross_sectional_rank_ic(x, y))

    def test_empty_returns_nan(self):
        assert np.isnan(cross_sectional_rank_ic(pd.Series([], dtype=float), pd.Series([], dtype=float)))


class TestDailyRankICSeries:
    """日度 Rank IC 序列."""

    @pytest.fixture
    def df(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range("2025-01-01", periods=10)
        frames = []
        for d in dates:
            n = 20
            score = rng.uniform(0, 1, n)
            ret = score + rng.normal(0, 0.1, n)  # 强相关
            frames.append(
                pd.DataFrame(
                    {"date": d, "score": score, "ret": ret}
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_series_length_equals_unique_dates(self, df):
        ics = daily_rank_ic_series(df, "score", "ret")
        assert len(ics) == df["date"].nunique()
        assert ics.index.name == "date"

    def test_positive_mean_ic(self, df):
        ics = daily_rank_ic_series(df, "score", "ret")
        assert ics.mean() > 0.5

    def test_uses_custom_date_col(self):
        df = pd.DataFrame(
            {
                "trade_date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 2, 1, 2],
                "ret": [1, 2, 1, 2],
            }
        )
        ics = daily_rank_ic_series(
            df, "score", "ret", date_col="trade_date", min_x_unique=2
        )
        assert len(ics) == 2

    def test_drop_invalid_dates(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 1, 1, 2],  # 第一日 x 唯一值不足
                "ret": [1, 2, 1, 2],
            }
        )
        ics = daily_rank_ic_series(df, "score", "ret", min_x_unique=2)
        assert len(ics) == 1


class TestMeanRankIC:
    """日度 IC 均值."""

    def test_mean_rank_ic_positive(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 2, 1, 2],
                "ret": [1, 2, 1, 2],
            }
        )
        assert mean_rank_ic(df, "score", "ret", min_x_unique=2) == pytest.approx(1.0)

    def test_abs_mean(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 2, 1, 2],
                "ret": [2, 1, 1, 2],  # 第一日 -1, 第二日 +1
            }
        )
        assert mean_rank_ic(df, "score", "ret", abs_mean=True, min_x_unique=2) == pytest.approx(1.0)
        assert mean_rank_ic(df, "score", "ret", abs_mean=False, min_x_unique=2) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        df = pd.DataFrame({"date": [], "score": [], "ret": []})
        assert mean_rank_ic(df, "score", "ret") == 0.0


class TestICIR:
    """ICIR 计算."""

    def test_icir_positive(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 2, 1, 2],
                "ret": [1, 2, 1, 2],
            }
        )
        # 两日期 IC 相同, std=0 → icir 返回 0.0
        assert icir(df, "score", "ret", min_x_unique=2) == 0.0

    def test_icir_with_variation(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2025-01-01", periods=20)
        frames = []
        for d in dates:
            n = 50
            score = rng.uniform(0, 1, n)
            ret = score + rng.normal(0, 0.3, n)
            frames.append(pd.DataFrame({"date": d, "score": score, "ret": ret}))
        df = pd.concat(frames, ignore_index=True)
        result = icir(df, "score", "ret")
        assert result > 0

    def test_insufficient_dates_returns_zero(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"],
                "score": [1, 2, 1, 2],
                "ret": [1, 2, 1, 2],
            }
        )
        assert icir(df, "score", "ret", min_x_unique=2) == 0.0
