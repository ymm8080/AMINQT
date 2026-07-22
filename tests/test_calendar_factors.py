# -*- coding: utf-8 -*-
"""calendar_factors 单元测试 (ARCH §5.11)."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from app.core.calendar_factors import (
    CALENDAR_FACTOR_COLUMNS,
    compute_calendar_factors,
    load_holidays,
)


@pytest.fixture
def trade_df():
    """2026-02-23 (Mon) ~ 2026-04-10 的工作日作为交易日."""
    dates = pd.bdate_range("2026-02-23", "2026-04-10")
    return pd.DataFrame({"date": dates, "close": np.linspace(10, 12, len(dates))})


@pytest.fixture
def holidays_file(tmp_path):
    """写入 holidays.json: 2026-03-09 (周一) 为节假日."""
    path = tmp_path / "holidays.json"
    path.write_text(json.dumps({"holidays": ["2026-03-09"]}), encoding="utf-8")
    return str(path)


class TestLoadHolidays:
    def test_missing_file_returns_empty(self, tmp_path):
        holidays = load_holidays(str(tmp_path / "nope.json"))
        assert holidays == set()

    def test_parses_nested_format(self, tmp_path):
        path = tmp_path / "h.json"
        path.write_text(
            json.dumps({"2026": ["2026-01-01", "2026-10-01"]}), encoding="utf-8"
        )
        holidays = load_holidays(str(path))
        assert pd.Timestamp("2026-01-01") in holidays
        assert pd.Timestamp("2026-10-01") in holidays

    def test_broken_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_holidays(str(path)) == set()


class TestComputeCalendarFactors:
    def test_output_columns(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        for col in CALENDAR_FACTOR_COLUMNS:
            assert col in out.columns
        assert np.isfinite(out[CALENDAR_FACTOR_COLUMNS].to_numpy()).all()

    def test_day_of_week(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        row = out.loc[out["date"] == pd.Timestamp("2026-02-23")].iloc[0]
        assert row["cal_day_of_week"] == 0.0  # Monday
        row = out.loc[out["date"] == pd.Timestamp("2026-02-27")].iloc[0]
        assert row["cal_day_of_week"] == 4.0  # Friday

    def test_days_to_holiday_with_file(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        by_date = out.set_index("date")
        # 03-06 (Fri) → 03-09 节假日: 中间 0 个交易日
        assert by_date.loc["2026-03-06", "cal_days_to_holiday"] == 0.0
        # 03-05 (Thu): 中间只有 03-06 → 1
        assert by_date.loc["2026-03-05", "cal_days_to_holiday"] == 1.0
        # 02-23 (Mon): 02-24..03-06 共 9 个交易日
        assert by_date.loc["2026-02-23", "cal_days_to_holiday"] == 9.0

    def test_pre_holiday_flag(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        by_date = out.set_index("date")
        assert by_date.loc["2026-03-06", "cal_pre_holiday_flag"] == 1.0
        assert by_date.loc["2026-03-03", "cal_pre_holiday_flag"] == 1.0  # 节前 3 日
        assert by_date.loc["2026-02-23", "cal_pre_holiday_flag"] == 0.0  # 节前 9 日

    def test_month_start_end(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        by_date = out.set_index("date")
        # 2 月前 3 个交易日: 23/24/25
        assert by_date.loc["2026-02-23", "cal_is_month_start"] == 1.0
        assert by_date.loc["2026-02-25", "cal_is_month_start"] == 1.0
        assert by_date.loc["2026-02-26", "cal_is_month_start"] == 0.0
        # 2 月最后 3 个交易日: 25/26/27
        assert by_date.loc["2026-02-27", "cal_is_month_end"] == 1.0
        assert by_date.loc["2026-02-25", "cal_is_month_end"] == 1.0
        assert by_date.loc["2026-02-24", "cal_is_month_end"] == 0.0

    def test_quarter_end(self, trade_df, holidays_file):
        out = compute_calendar_factors(trade_df, holidays_file)
        by_date = out.set_index("date")
        # 3 月末 = 季末
        assert by_date.loc["2026-03-31", "cal_is_quarter_end"] == 1.0
        assert by_date.loc["2026-03-30", "cal_is_quarter_end"] == 1.0
        # 2 月末非季末
        assert by_date.loc["2026-02-27", "cal_is_quarter_end"] == 0.0

    def test_missing_file_weekend_fallback(self, trade_df, tmp_path, caplog):
        """无节假日文件 → 周末降级 + 警告日志."""
        with caplog.at_level(logging.WARNING):
            out = compute_calendar_factors(trade_df, str(tmp_path / "missing.json"))
        assert any(
            "weekend" in r.message or "缺失" in r.message for r in caplog.records
        )
        by_date = out.set_index("date")
        # 2026-03-02 (Mon) → 下一个周六: 中间 Tue-Fri 4 个交易日
        assert by_date.loc["2026-03-02", "cal_days_to_holiday"] == 4.0
        # 2026-03-06 (Fri) → 0
        assert by_date.loc["2026-03-06", "cal_days_to_holiday"] == 0.0
        # 周一距周末 ≤5 → 节前 flag 也置 1 (降级口径)
        assert by_date.loc["2026-03-02", "cal_pre_holiday_flag"] == 1.0

    def test_no_future_leakage(self, holidays_file):
        """只用前半段数据时, 前半段的因子值不应依赖后半段日期
        (月末/月初除外 — 它们按定义由当月交易日集合决定, 此处验证
        days_to_holiday 在节假日前的部分不受尾部数据影响)."""
        dates = pd.bdate_range("2026-02-23", "2026-04-10")
        full = compute_calendar_factors(pd.DataFrame({"date": dates}), holidays_file)
        head = compute_calendar_factors(
            pd.DataFrame({"date": dates[:10]}), holidays_file
        )
        col = "cal_days_to_holiday"
        # 尾部数据缺失时 holidays 仍在视野内 → 前 10 行应一致
        assert np.allclose(full[col].to_numpy()[:10], head[col].to_numpy())
