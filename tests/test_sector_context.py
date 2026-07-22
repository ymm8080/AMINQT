# -*- coding: utf-8 -*-
"""Tests for app/core/sector_context — 板块平均效应 4 因子."""

import numpy as np
import pandas as pd
import pytest

from app.core.sector_context import SECTOR_FACTOR_COLUMNS, SectorContext


def _df(closes, opens=None, volumes=None):
    """构建合成日线 DataFrame."""
    n = len(closes)
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "open": opens if opens is not None else closes,
            "close": closes,
            "volume": volumes if volumes is not None else [1000.0] * n,
        }
    )


# 板块: TECH = A(600xxx→map 覆盖) + B + C; BANK = D
SECTOR_MAP = {"600001": "TECH", "000001": "TECH", "300001": "TECH", "600004": "BANK"}


def _stocks():
    # 最新日涨跌幅: A +2%, B -2%, C +6%, D +1%
    return {
        "600001": _df([100.0, 102.0], opens=[100.0, 101.0], volumes=[1000.0, 2000.0]),
        "000001": _df([100.0, 98.0], opens=[100.0, 99.0], volumes=[1000.0, 1000.0]),
        "300001": _df([100.0, 106.0], opens=[100.0, 105.0], volumes=[1000.0, 3000.0]),
        "600004": _df([100.0, 101.0], opens=[100.0, 100.5], volumes=[1000.0, 1000.0]),
    }


class TestCompute:
    def test_four_factor_values(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("600001", _stocks())

        assert set(out.keys()) == set(SECTOR_FACTOR_COLUMNS)
        # 板块均值 = (0.02 - 0.02 + 0.06) / 3 = 0.02
        assert out["sector_avg_pct_change"] == pytest.approx(0.02)
        # 相对强度 = 0.02 - 0.02 = 0
        assert out["sector_relative_strength"] == pytest.approx(0.0)
        # 排名分位: pct<=0.02 的成员 = A(0.02),B(-0.02) → 2/3
        assert out["sector_rank_pct"] == pytest.approx(2.0 / 3.0)
        # 净流入 = (102-101)*2000 + (98-99)*1000 + (106-105)*3000 = 4000
        assert out["sector_net_flow"] == pytest.approx(4000.0)

    def test_strongest_member_rank_is_one(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("300001", _stocks())  # +6% 为板块最强
        assert out["sector_rank_pct"] == pytest.approx(1.0)
        assert out["sector_relative_strength"] == pytest.approx(0.04)

    def test_other_sector_isolated(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("600004", _stocks())  # BANK 只有自身
        assert out["sector_avg_pct_change"] == pytest.approx(0.01)
        assert out["sector_relative_strength"] == pytest.approx(0.0)
        assert out["sector_rank_pct"] == pytest.approx(1.0)
        assert out["sector_net_flow"] == pytest.approx(500.0)

    def test_output_has_no_nan(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("600001", _stocks())
        for v in out.values():
            assert np.isfinite(v)


class TestEdgeCases:
    def test_empty_all_stocks(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("600001", {})
        assert out == {c: 0.0 for c in SECTOR_FACTOR_COLUMNS}

    def test_symbol_missing(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        out = ctx.compute("999999", _stocks())
        assert out == {c: 0.0 for c in SECTOR_FACTOR_COLUMNS}

    def test_insufficient_rows(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        stocks = {"600001": _df([100.0])}  # 仅 1 行, 无法算涨跌幅
        out = ctx.compute("600001", stocks)
        assert out == {c: 0.0 for c in SECTOR_FACTOR_COLUMNS}

    def test_member_with_bad_data_skipped(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        stocks = _stocks()
        stocks["000001"] = pd.DataFrame()  # 空 DataFrame → 跳过
        out = ctx.compute("600001", stocks)
        # TECH 有效成员: A(0.02), C(0.06) → 均值 0.04
        assert out["sector_avg_pct_change"] == pytest.approx(0.04)

    def test_nan_close_handled(self):
        ctx = SectorContext(sector_map=SECTOR_MAP)
        stocks = _stocks()
        bad = _df([100.0, np.nan])
        stocks["000001"] = bad
        out = ctx.compute("600001", stocks)
        for v in out.values():
            assert np.isfinite(v)


class TestSectorMapping:
    def test_prefix_heuristic(self):
        ctx = SectorContext()  # 无 map
        assert ctx.sector_for("600519") == "SSE_MAIN"
        assert ctx.sector_for("000001") == "SZSE_MAIN"
        assert ctx.sector_for("300750") == "CHINEXT"
        assert ctx.sector_for("688981") == "STAR"
        assert ctx.sector_for("830799") == "OTHER"

    def test_map_overrides_heuristic(self):
        ctx = SectorContext(sector_map={"600519": "白酒"})
        assert ctx.sector_for("600519") == "白酒"

    def test_prefix_heuristic_groups_same_board(self):
        ctx = SectorContext()  # 无 map: 600001/600004 同属 SSE_MAIN
        stocks = {
            "600001": _df([100.0, 102.0]),
            "600004": _df([100.0, 104.0]),
        }
        out = ctx.compute("600001", stocks)
        assert out["sector_avg_pct_change"] == pytest.approx(0.03)

    def test_get_factor_columns(self):
        cols = SectorContext.get_factor_columns()
        assert cols == SECTOR_FACTOR_COLUMNS
        assert cols is not SECTOR_FACTOR_COLUMNS  # 返回副本
