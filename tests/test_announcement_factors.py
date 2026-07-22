# -*- coding: utf-8 -*-
"""announcement_factors 单元测试 (ARCH §5.11)."""

import logging

import numpy as np
import pandas as pd
import pytest

from app.core.announcement_factors import (
    ANNOUNCEMENT_FACTOR_COLUMNS,
    compute_announcement_factors,
)

SYMBOL = "600000"


@pytest.fixture
def trade_df():
    dates = pd.bdate_range("2026-03-02", "2026-04-10")
    return pd.DataFrame({"date": dates, "close": np.linspace(10, 11, len(dates))})


@pytest.fixture
def ann_dir(tmp_path):
    """写入公告 parquet: major/earnings/hold_change/risk_warning 各一条."""
    ann = pd.DataFrame(
        {
            "date": [
                "2026-03-05",
                "2026-03-05",
                "2026-03-10",
                "2026-03-20",
                "2026-03-25",
            ],
            "type": ["major", "earnings", "hold_change", "risk_warning", "major"],
        }
    )
    ann.to_parquet(tmp_path / f"{SYMBOL}_ann.parquet", index=False)
    return str(tmp_path)


class TestWithAnnouncements:
    def test_output_columns(self, trade_df, ann_dir):
        out = compute_announcement_factors(trade_df, SYMBOL, ann_dir)
        for col in ANNOUNCEMENT_FACTOR_COLUMNS:
            assert col in out.columns
        assert np.isfinite(out[ANNOUNCEMENT_FACTOR_COLUMNS].to_numpy()).all()

    def test_ann_count_5d(self, trade_df, ann_dir):
        out = compute_announcement_factors(trade_df, SYMBOL, ann_dir)
        by_date = out.set_index("date")
        # 03-05 当日 2 条公告
        assert by_date.loc["2026-03-05", "ann_count_5d"] == 2.0
        # 03-10: 近 5 交易日 = 03-04,05,06,09,10 → 03-05×2 + 03-10×1 = 3
        assert by_date.loc["2026-03-10", "ann_count_5d"] == 3.0
        # 03-03: 无公告
        assert by_date.loc["2026-03-03", "ann_count_5d"] == 0.0

    def test_major_flag_decay(self, trade_df, ann_dir):
        out = compute_announcement_factors(trade_df, SYMBOL, ann_dir)
        by_date = out.set_index("date")
        # 事件日附近 flag > 0
        assert by_date.loc["2026-03-05", "ann_major_flag"] > 0
        assert by_date.loc["2026-03-06", "ann_major_flag"] > 0
        # 事件前为 0 (无未来函数)
        assert by_date.loc["2026-03-04", "ann_major_flag"] == 0.0
        # 第二次 major (03-25) 后 flag 重新抬升
        assert by_date.loc["2026-03-26", "ann_major_flag"] > 0

    def test_earnings_and_hold_change(self, trade_df, ann_dir):
        out = compute_announcement_factors(trade_df, SYMBOL, ann_dir)
        by_date = out.set_index("date")
        assert by_date.loc["2026-03-05", "ann_earnings_flag"] > 0
        assert by_date.loc["2026-03-04", "ann_earnings_flag"] == 0.0
        assert by_date.loc["2026-03-10", "ann_hold_change_flag"] > 0
        assert by_date.loc["2026-03-09", "ann_hold_change_flag"] == 0.0

    def test_risk_warning_flag(self, trade_df, ann_dir):
        out = compute_announcement_factors(trade_df, SYMBOL, ann_dir)
        by_date = out.set_index("date")
        # 03-20 风险警示 → 当日及之后 10 个交易日内 = 1
        assert by_date.loc["2026-03-20", "ann_risk_warning_flag"] == 1.0
        assert by_date.loc["2026-03-27", "ann_risk_warning_flag"] == 1.0
        # 事件前 = 0
        assert by_date.loc["2026-03-19", "ann_risk_warning_flag"] == 0.0


class TestMissingFile:
    def test_all_zeros(self, trade_df, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            out = compute_announcement_factors(trade_df, "999999", str(tmp_path))
        for col in ANNOUNCEMENT_FACTOR_COLUMNS:
            assert (out[col] == 0.0).all(), col
        assert any("缺失" in r.message for r in caplog.records)

    def test_broken_parquet(self, trade_df, tmp_path):
        (tmp_path / f"{SYMBOL}_ann.parquet").write_bytes(b"not a parquet")
        out = compute_announcement_factors(trade_df, SYMBOL, str(tmp_path))
        for col in ANNOUNCEMENT_FACTOR_COLUMNS:
            assert (out[col] == 0.0).all(), col

    def test_missing_type_column(self, trade_df, tmp_path):
        pd.DataFrame({"date": ["2026-03-05"]}).to_parquet(
            tmp_path / f"{SYMBOL}_ann.parquet", index=False
        )
        out = compute_announcement_factors(trade_df, SYMBOL, str(tmp_path))
        for col in ANNOUNCEMENT_FACTOR_COLUMNS:
            assert (out[col] == 0.0).all(), col
