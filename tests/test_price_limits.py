# -*- coding: utf-8 -*-
"""Tests for app/core/price_limits — 涨跌停幅度与判定."""

from app.core.price_limits import (
    get_limit_pct,
    is_limit_down,
    is_limit_up,
)
from app.core.universe_manager import Universe


class TestGetLimitPct:
    def test_main_board(self):
        assert get_limit_pct("600519", Universe.MAIN_BOARD) == 0.10

    def test_growth_boards(self):
        assert get_limit_pct("300750", Universe.GROWTH_BOARDS) == 0.20

    def test_st_overrides_board(self):
        assert get_limit_pct("600001", Universe.MAIN_BOARD, is_st=True) == 0.05
        assert get_limit_pct("300750", Universe.GROWTH_BOARDS, is_st=True) == 0.05

    def test_universe_all_routes_by_prefix(self):
        assert get_limit_pct("600519", Universe.ALL) == 0.10
        assert get_limit_pct("000001", Universe.ALL) == 0.10
        assert get_limit_pct("300750", Universe.ALL) == 0.20
        assert get_limit_pct("688981", Universe.ALL) == 0.20


class TestIsLimitUp:
    def test_main_board_limit_up(self):
        # prev 10.00 → 涨停价 11.00
        assert is_limit_up("600519", 11.00, 10.00, Universe.MAIN_BOARD) is True
        assert is_limit_up("600519", 10.99, 10.00, Universe.MAIN_BOARD) is False

    def test_growth_board_limit_up(self):
        # prev 10.00 → 涨停价 12.00
        assert is_limit_up("300750", 12.00, 10.00, Universe.GROWTH_BOARDS) is True
        assert is_limit_up("300750", 11.50, 10.00, Universe.GROWTH_BOARDS) is False

    def test_st_limit_up(self):
        # prev 10.00 → ST 涨停价 10.50
        assert (
            is_limit_up("600001", 10.50, 10.00, Universe.MAIN_BOARD, is_st=True) is True
        )
        assert (
            is_limit_up("600001", 10.49, 10.00, Universe.MAIN_BOARD, is_st=True)
            is False
        )

    def test_rounding(self):
        # prev 3.33 → round(3.663, 2) = 3.66
        assert is_limit_up("600519", 3.66, 3.33, Universe.MAIN_BOARD) is True
        assert is_limit_up("600519", 3.65, 3.33, Universe.MAIN_BOARD) is False

    def test_above_limit_still_true(self):
        assert is_limit_up("600519", 15.00, 10.00, Universe.MAIN_BOARD) is True

    def test_bad_prev_close_returns_false(self):
        assert is_limit_up("600519", 11.00, 0.0, Universe.MAIN_BOARD) is False
        assert is_limit_up("600519", 11.00, -1.0, Universe.MAIN_BOARD) is False
        assert is_limit_up("600519", 11.00, float("nan"), Universe.MAIN_BOARD) is False
        assert is_limit_up("600519", 11.00, None, Universe.MAIN_BOARD) is False


class TestIsLimitDown:
    def test_main_board_limit_down(self):
        # prev 10.00 → 跌停价 9.00
        assert is_limit_down("600519", 9.00, 10.00, Universe.MAIN_BOARD) is True
        assert is_limit_down("600519", 9.01, 10.00, Universe.MAIN_BOARD) is False

    def test_growth_board_limit_down(self):
        assert is_limit_down("300750", 8.00, 10.00, Universe.GROWTH_BOARDS) is True
        assert is_limit_down("300750", 8.50, 10.00, Universe.GROWTH_BOARDS) is False

    def test_st_limit_down(self):
        assert (
            is_limit_down("600001", 9.50, 10.00, Universe.MAIN_BOARD, is_st=True)
            is True
        )
        assert (
            is_limit_down("600001", 9.51, 10.00, Universe.MAIN_BOARD, is_st=True)
            is False
        )

    def test_rounding(self):
        # prev 3.33 → round(2.997, 2) = 3.00
        assert is_limit_down("600519", 3.00, 3.33, Universe.MAIN_BOARD) is True
        assert is_limit_down("600519", 3.01, 3.33, Universe.MAIN_BOARD) is False

    def test_bad_prev_close_returns_false(self):
        assert is_limit_down("600519", 9.00, 0.0, Universe.MAIN_BOARD) is False
        assert is_limit_down("600519", 9.00, float("nan"), Universe.MAIN_BOARD) is False
