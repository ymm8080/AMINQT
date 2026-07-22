# -*- coding: utf-8 -*-
"""BaseLiquidityFilter 单元测试 (P6, DESIGN_V1 §4 STEP1 第一步)."""

import numpy as np
import pandas as pd
import pytest

from app.core.base_liquidity_filter import BaseLiquidityFilter


def _make_daily_df(n: int = 250, turnover: float = 0.08,
                   amount: float = 6e8, amplitude: float = 0.06,
                   with_limit_up: bool = True) -> pd.DataFrame:
    """构造满足/不满足过滤条件的合成近一年日线."""
    rng = np.random.default_rng(42)
    close = 10.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, n))
    high = close * (1.0 + amplitude / 2.0)
    low = close * (1.0 - amplitude / 2.0)
    volume = amount / close
    df = pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": np.full(n, amount),
        "turnover": np.full(n, turnover),
    })
    if with_limit_up and n > 101:
        # 第 100 天制造一个涨停 (涨幅 >= 9.8%)
        df.loc[100, "close"] = df.loc[99, "close"] * 1.10
        df.loc[100, "high"] = df.loc[100, "close"] * 1.01
        df.loc[100, "low"] = df.loc[99, "close"] * 1.02
        df.loc[100, "volume"] = df.loc[100, "amount"] / df.loc[100, "close"]
    return df


@pytest.fixture()
def filt() -> BaseLiquidityFilter:
    return BaseLiquidityFilter()


class TestCheckLiquidity:
    def test_active_stock_passes(self, filt):
        df = _make_daily_df()
        assert filt.check_liquidity(df) is True

    def test_low_turnover_fails(self, filt):
        df = _make_daily_df(turnover=0.01)
        assert filt.check_liquidity(df) is False

    def test_low_amount_fails(self, filt):
        df = _make_daily_df(amount=1e8)
        assert filt.check_liquidity(df) is False

    def test_low_amplitude_fails(self, filt):
        df = _make_daily_df(amplitude=0.01)
        assert filt.check_liquidity(df) is False

    def test_no_limit_up_fails(self, filt):
        df = _make_daily_df(with_limit_up=False)
        assert filt.check_liquidity(df) is False

    def test_insufficient_data_fails(self, filt):
        df = _make_daily_df(n=10)
        assert filt.check_liquidity(df) is False

    def test_limit_up_by_price_match(self, filt):
        """close == round(prev_close*1.1, 2) 也计为涨停."""
        df = _make_daily_df(with_limit_up=False)
        df.loc[150, "close"] = round(df.loc[149, "close"] * 1.1, 2)
        assert filt._limit_up_count(df) >= 1

    def test_turnover_percent_scale(self, filt):
        """换手率百分数量纲 (8 = 8%) 自动识别."""
        df = _make_daily_df()
        df["turnover"] = 8.0  # 百分数
        assert filt.check_liquidity(df) is True


class TestCheckExclusions:
    def test_st_stock_excluded(self, filt):
        df = _make_daily_df()
        df["name"] = "ST测试"
        assert filt.check_exclusions("000001", df) is True

    def test_delisting_stock_excluded(self, filt):
        df = _make_daily_df()
        df["name"] = "退市测试"
        assert filt.check_exclusions("000002", df) is True

    def test_risk_warning_excluded(self, filt):
        df = _make_daily_df()
        df["ann_risk_warning_flag"] = 0
        df.loc[df.index[-1], "ann_risk_warning_flag"] = 1
        assert filt.check_exclusions("000003", df) is True

    def test_consecutive_loss_excluded(self, filt):
        df = _make_daily_df()
        df["net_profit"] = 1e8
        df.loc[df.index[-2:], "net_profit"] = -1e8  # 最近两期连亏
        assert filt.check_exclusions("000004", df) is True

    def test_large_cap_low_vol_excluded(self, filt):
        df = _make_daily_df()
        # 构造低波动: 固定涨幅极小
        n = len(df)
        df["close"] = 100.0 * np.cumprod(np.full(n, 1.0001))
        df["mktcap"] = 2e11  # 2000 亿
        assert filt.check_exclusions("600519", df) is True

    def test_normal_stock_not_excluded(self, filt):
        df = _make_daily_df()
        df["name"] = "正常股份"
        assert filt.check_exclusions("000005", df) is False

    def test_missing_optional_cols_skipped(self, filt):
        """无 name/net_profit/mktcap 列时不误剔除."""
        df = _make_daily_df()
        assert filt.check_exclusions("000006", df) is False


class TestApply:
    def test_apply_mixed_pool(self, filt):
        good = _make_daily_df()
        st = _make_daily_df()
        st["name"] = "ST坏股"
        illiquid = _make_daily_df(turnover=0.01)
        pool = {"GOOD": good, "ST": st, "ILLIQ": illiquid}
        assert filt.apply(pool) == ["GOOD"]

    def test_apply_empty_df_skipped(self, filt):
        pool = {"EMPTY": pd.DataFrame(), "GOOD": _make_daily_df()}
        assert filt.apply(pool) == ["GOOD"]

    def test_config_with_initial_structure(self):
        """{initial, bounds} 结构配置正确解析."""
        cfg = {
            "min_turnover": {"initial": 0.05, "bounds": [0.02, 0.10]},
            "min_amount": {"initial": 500000000, "bounds": [2e8, 1e9]},
            "min_amplitude": {"initial": 0.05, "bounds": [0.03, 0.10]},
            "min_limit_up_count": {"initial": 1, "bounds": [1, 5]},
        }
        f = BaseLiquidityFilter(cfg)
        assert f.min_turnover == pytest.approx(0.05)
        assert f.min_amount == pytest.approx(5e8)
        assert f.check_liquidity(_make_daily_df()) is True

    def test_config_override_threshold(self):
        """提高换手率阈值后原本通过的票被过滤."""
        f = BaseLiquidityFilter({"min_turnover": 0.20})
        assert f.check_liquidity(_make_daily_df(turnover=0.08)) is False
