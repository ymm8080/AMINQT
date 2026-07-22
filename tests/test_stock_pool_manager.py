# -*- coding: utf-8 -*-
"""StockPoolManager 测试 (P10.11, ARCH §5.16)."""

import json

import pytest

from app.core.stock_pool_manager import StockPoolManager, TICK_FIELDS


@pytest.fixture
def pool_file(tmp_path):
    return str(tmp_path / "stock_pool.json")


@pytest.fixture
def mgr(pool_file):
    m = StockPoolManager(pool_path=pool_file)
    m.add_stock("600519", name="贵州茅台")
    m.add_stock("000858", name="五粮液")
    m.add_stock("300750", name="宁德时代")
    return m


class TestPersistence:
    def test_save_and_reload(self, mgr, pool_file):
        mgr.set_tick("600519", "is_watchlist", True)
        m2 = StockPoolManager(pool_path=pool_file)
        pool = m2.get_pool()
        assert len(pool) == 3
        rec = [r for r in pool if r["symbol"] == "600519"][0]
        assert rec["ticks"]["is_watchlist"] is True

    def test_file_is_valid_json(self, mgr, pool_file):
        with open(pool_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "pool" in data and "last_updated" in data
        assert len(data["pool"]) == 3

    def test_load_missing_file(self, tmp_path):
        m = StockPoolManager(pool_path=str(tmp_path / "nonexistent.json"))
        assert m.get_pool() == []


class TestTicks:
    def test_set_tick_and_filter(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True)
        mgr.set_tick("000858", "is_daily_buy", True)
        buys = mgr.get_pool(filter_by_tick="is_daily_buy")
        assert {r["symbol"] for r in buys} == {"600519", "000858"}
        assert len(mgr.get_pool(filter_by_tick="is_intraday_sell")) == 0

    def test_invalid_tick_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.set_tick("600519", "is_magic", True)
        with pytest.raises(ValueError):
            mgr.get_pool(filter_by_tick="is_magic")

    def test_tick_on_missing_stock_raises(self, mgr):
        with pytest.raises(ValueError):
            mgr.set_tick("999999", "is_watchlist", True)

    def test_daily_buy_auto_fixes(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True, source="manual")
        rec = [r for r in mgr.get_pool() if r["symbol"] == "600519"][0]
        assert rec["is_fixed"] is True
        # 撤标自动取消固定 (ARCH §5.16.4)
        mgr.set_tick("600519", "is_daily_buy", False)
        rec = [r for r in mgr.get_pool() if r["symbol"] == "600519"][0]
        assert rec["is_fixed"] is False

    def test_daily_sell_auto_fixes(self, mgr):
        mgr.set_tick("000858", "is_daily_sell", True)
        assert "000858" in mgr.get_fixed_stocks()

    def test_other_tick_does_not_fix(self, mgr):
        mgr.set_tick("600519", "is_intraday_buy", True)
        assert "600519" not in mgr.get_fixed_stocks()

    def test_ticks_history_recorded(self, mgr):
        mgr.set_tick("600519", "is_watchlist", True, source="manual")
        mgr.set_tick("600519", "is_daily_buy", True, source="pipeline1")
        rec = [r for r in mgr.get_pool() if r["symbol"] == "600519"][0]
        hist = rec["ticks_history"]
        assert len(hist) == 2
        assert hist[0]["tick"] == "is_watchlist" and hist[0]["source"] == "manual"
        assert hist[1]["tick"] == "is_daily_buy" and hist[1]["source"] == "pipeline1"
        assert all("timestamp" in h for h in hist)


class TestAddRemove:
    def test_add_duplicate_noop(self, mgr):
        mgr.add_stock("600519")
        assert len(mgr.get_pool()) == 3

    def test_remove(self, mgr):
        mgr.remove_stock("300750")
        assert {r["symbol"] for r in mgr.get_pool()} == {"600519", "000858"}

    def test_remove_missing_noop(self, mgr):
        mgr.remove_stock("999999")
        assert len(mgr.get_pool()) == 3


class TestAllocation:
    def test_allocation_requires_daily_buy(self, mgr):
        with pytest.raises(ValueError):
            mgr.set_allocation("600519", 20.0)

    def test_allocation_and_total(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True)
        mgr.set_tick("000858", "is_daily_buy", True)
        mgr.set_allocation("600519", 30.0)
        mgr.set_allocation("000858", 25.0)
        assert mgr.get_buy_allocation_total() == pytest.approx(55.0)

    def test_allocation_100_pct_cap(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True)
        mgr.set_tick("000858", "is_daily_buy", True)
        mgr.set_tick("300750", "is_daily_buy", True)
        mgr.set_allocation("600519", 60.0)
        mgr.set_allocation("000858", 40.0)
        with pytest.raises(ValueError):
            mgr.set_allocation("300750", 0.5)  # 60+40+0.5 > 100
        # 修改已有股票的分配不超限
        mgr.set_allocation("600519", 50.0)
        mgr.set_allocation("300750", 10.0)
        assert mgr.get_buy_allocation_total() == pytest.approx(100.0)

    def test_allocation_bounds(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True)
        with pytest.raises(ValueError):
            mgr.set_allocation("600519", -1.0)
        with pytest.raises(ValueError):
            mgr.set_allocation("600519", 100.1)


class TestPipelineUpdate:
    def test_fixed_preserved_across_update(self, mgr):
        # 600519 固定 (日线买 + 分配), 000858 非固定
        mgr.set_tick("600519", "is_daily_buy", True)
        mgr.set_allocation("600519", 30.0)
        mgr.set_tick("000858", "is_watchlist", True)

        new_pool = [
            {
                "symbol": "600519",
                "name": "贵州茅台",
                "score": 0.99,
                "prob_up": 0.9,
                "pct_up": 0.2,
            },  # 试图覆盖固定股
            {
                "symbol": "601318",
                "name": "中国平安",
                "score": 0.80,
                "prob_up": 0.7,
                "pct_up": 0.1,
            },  # 新增
        ]
        mgr.update_from_pipeline1(new_pool)
        pool = {r["symbol"]: r for r in mgr.get_pool()}

        # 固定股: score/ticks/allocation 均不被覆盖
        fixed = pool["600519"]
        assert fixed["is_fixed"] is True
        assert fixed["score"] == 0.0
        assert fixed["percentage_allocation"] == 30.0
        assert fixed["ticks"]["is_daily_buy"] is True

        # 非固定且不在新池 → 移除; 但 000858 是自选 → 保留 (ARCH §5.16.3)
        assert "000858" in pool
        # 非固定非自选不在新池 → 移除
        assert "300750" not in pool
        # 新增股票 ticks 全 False
        assert pool["601318"]["ticks"] == {t: False for t in TICK_FIELDS}
        assert pool["601318"]["added_by"] == "pipeline1"
        assert pool["601318"]["score"] == 0.80

    def test_nonfixed_score_updated_ticks_kept(self, mgr):
        mgr.set_tick("000858", "is_watchlist", True, source="manual")
        mgr.update_from_pipeline1(
            [
                {"symbol": "000858", "score": 0.77, "prob_up": 0.6, "pct_up": 0.05},
            ]
        )
        pool = {r["symbol"]: r for r in mgr.get_pool()}
        assert pool["000858"]["score"] == 0.77
        assert pool["000858"]["ticks"]["is_watchlist"] is True
        assert "600519" not in pool  # 非固定非自选被移除

    def test_fixed_not_in_new_pool_kept(self, mgr):
        mgr.set_tick("600519", "is_daily_buy", True)
        mgr.update_from_pipeline1(
            [
                {"symbol": "601318", "score": 0.5, "prob_up": 0.5, "pct_up": 0.01},
            ]
        )
        symbols = {r["symbol"] for r in mgr.get_pool()}
        assert symbols == {"600519", "601318"}
        assert mgr.get_fixed_stocks() == ["600519"]
