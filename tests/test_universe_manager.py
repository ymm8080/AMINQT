# -*- coding: utf-8 -*-
"""Tests for app/core/universe_manager — Universe 分池 + ST 标记."""

from app.core.universe_manager import (
    Universe,
    UniverseManager,
    classify_symbol,
    name_is_st,
)


class TestClassify:
    """代码前缀 → Universe 路由."""

    def test_main_board_prefixes(self):
        for code in ("600519", "601318", "000001", "000858", "001234", "002594"):
            assert classify_symbol(code) == Universe.MAIN_BOARD, code

    def test_growth_board_prefixes(self):
        for code in ("300750", "301234", "688981", "688001"):
            assert classify_symbol(code) == Universe.GROWTH_BOARDS, code

    def test_unknown_prefix_defaults_main_board(self):
        # 北交所等未覆盖前缀 → 保守默认 MAIN_BOARD
        assert classify_symbol("830799") == Universe.MAIN_BOARD

    def test_whitespace_stripped(self):
        assert classify_symbol(" 600519 ") == Universe.MAIN_BOARD

    def test_manager_classify_delegates(self):
        mgr = UniverseManager()
        assert mgr.classify("300750") == Universe.GROWTH_BOARDS


class TestGetUniverseStocks:
    STOCKS = ["600519", "000001", "300750", "688981"]

    def test_all_returns_everything(self):
        mgr = UniverseManager(stocks=self.STOCKS)
        assert mgr.get_universe_stocks(Universe.ALL) == self.STOCKS

    def test_main_board_filter(self):
        mgr = UniverseManager(stocks=self.STOCKS)
        assert mgr.get_universe_stocks(Universe.MAIN_BOARD) == ["600519", "000001"]

    def test_growth_boards_filter(self):
        mgr = UniverseManager(stocks=self.STOCKS)
        assert mgr.get_universe_stocks(Universe.GROWTH_BOARDS) == ["300750", "688981"]

    def test_empty_stock_list(self):
        mgr = UniverseManager()
        assert mgr.get_universe_stocks(Universe.ALL) == []
        assert mgr.get_universe_stocks(Universe.MAIN_BOARD) == []

    def test_set_stocks_updates(self):
        mgr = UniverseManager()
        mgr.set_stocks(["300001"])
        assert mgr.get_universe_stocks(Universe.GROWTH_BOARDS) == ["300001"]


class TestIsSt:
    NAMES = {
        "600001": "ST 钢铁",
        "600002": "*ST 化工",
        "600003": "退市XX",
        "600519": "贵州茅台",
        "600004": "st科技",  # 小写 st
    }

    def _mgr(self):
        return UniverseManager(name_map=self.NAMES)

    def test_st_prefix(self):
        assert self._mgr().is_st("600001") is True

    def test_star_st_prefix(self):
        assert self._mgr().is_st("600002") is True

    def test_delist_prefix(self):
        assert self._mgr().is_st("600003") is True

    def test_normal_name(self):
        assert self._mgr().is_st("600519") is False

    def test_case_insensitive(self):
        assert self._mgr().is_st("600004") is True

    def test_unknown_symbol_returns_false(self):
        assert self._mgr().is_st("999999") is False

    def test_name_is_st_edge_cases(self):
        assert name_is_st("") is False
        assert name_is_st(None) is False
        assert name_is_st("  ST 药业  ") is True
