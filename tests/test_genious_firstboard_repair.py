# -*- coding: utf-8 -*-
"""首板/板前哨两页面板补跑接线测试 (2026-10-28 用户令: 两页不许因面板滞后静默消失).

只验 _build_firstboard_sheets 的修复 wiring (stale → 补跑 → 重载 → 出页),
页内容语义由 tests/test_firstboard_pages.py 覆盖。
"""

import sys

import pandas as pd
import pytest

sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")

from scripts import _genious_excel as gen

TARGET = "20261028"


def _frame(max_date: str) -> pd.DataFrame:
    """最小主板框: 两行, date 上限 = max_date (load_mainboard 形状: datetime 列)."""
    d = pd.to_datetime(max_date)
    return pd.DataFrame(
        {
            "date": [d - timedelta(days=1), d],
            "symbol": ["600001", "600002"],
            "close": [10.0, 11.0],
        }
    )


from datetime import timedelta  # noqa: E402  (scripts/* 允许; 测试文件保持顶部可读)


@pytest.fixture()
def _fake_pages(monkeypatch):
    """serve_* 换成标记帧, 屏蔽真实页面计算; banner 停训哨兵读数抹平."""
    monkeypatch.setattr(
        "scripts._firstboard_pages.serve_firstboard",
        lambda pdf: pd.DataFrame({"page": ["fb"], "rows": [len(pdf)]}),
    )
    monkeypatch.setattr(
        "scripts._firstboard_pages.serve_preboard",
        lambda pdf, record_csv=None: pd.DataFrame({"page": ["pb"], "rows": [len(pdf)]}),
    )
    monkeypatch.setattr(
        "scripts._firstboard_pages.model_stale_trading_days",
        lambda *a, **k: None,
    )


def test_stale_panel_repaired_then_pages_built(monkeypatch, _fake_pages):
    """面板滞后 + 补跑成功 → 重载后面板含当日 → 两页照常交付."""
    monkeypatch.setattr(gen, "_DRY_RUN", False)
    monkeypatch.setattr(gen, "_avail_commit_gb", lambda: 100.0)
    calls = {"load": 0, "repair": 0}

    def fake_load(panel_path=None, tail_dates=450):
        calls["load"] += 1
        return _frame("20261027" if calls["load"] == 1 else TARGET)

    monkeypatch.setattr("scripts._firstboard_pages.load_mainboard", fake_load)

    def fake_repair(target):
        calls["repair"] += 1
        assert target == TARGET
        return True

    monkeypatch.setattr(gen, "_repair_panel_via_fetch", fake_repair)

    fb, pb, banner = gen._build_firstboard_sheets(TARGET, dry_run=False)
    assert calls == {"load": 2, "repair": 1}
    assert list(fb["page"]) == ["fb"] and fb["rows"].iloc[0] == 2
    assert list(pb["page"]) == ["pb"] and pb["rows"].iloc[0] == 2


def test_repair_failed_keeps_skip(monkeypatch, _fake_pages):
    """补跑失败 (数据源仍缺) → 维持宁缺勿错: (None, None, banner), 不再重载."""
    monkeypatch.setattr(gen, "_DRY_RUN", False)
    monkeypatch.setattr(gen, "_avail_commit_gb", lambda: 100.0)
    calls = {"load": 0}

    def fake_load(panel_path=None, tail_dates=450):
        calls["load"] += 1
        return _frame("20261027")

    monkeypatch.setattr("scripts._firstboard_pages.load_mainboard", fake_load)
    monkeypatch.setattr(gen, "_repair_panel_via_fetch", lambda target: False)

    fb, pb, banner = gen._build_firstboard_sheets(TARGET, dry_run=False)
    assert calls["load"] == 1  # 修复失败不重载
    assert fb is None and pb is None and banner


def test_dry_run_and_low_memory_skip_repair(monkeypatch, _fake_pages):
    """dry-run 契约 (只打印不落文件) 与低内存 (<3.5GB) 都不得触发补跑子进程."""
    calls = {"repair": 0}

    def fake_repair(target):
        calls["repair"] += 1
        return True

    monkeypatch.setattr(
        "scripts._firstboard_pages.load_mainboard",
        lambda panel_path=None, tail_dates=450: _frame("20261027"),
    )
    monkeypatch.setattr(gen, "_repair_panel_via_fetch", fake_repair)

    monkeypatch.setattr(gen, "_DRY_RUN", True)
    monkeypatch.setattr(gen, "_avail_commit_gb", lambda: 100.0)
    fb, pb, _ = gen._build_firstboard_sheets(TARGET, dry_run=True)
    assert fb is None and pb is None and calls["repair"] == 0

    monkeypatch.setattr(gen, "_DRY_RUN", False)
    monkeypatch.setattr(gen, "_avail_commit_gb", lambda: 0.5)
    fb, pb, _ = gen._build_firstboard_sheets(TARGET, dry_run=False)
    assert fb is None and pb is None and calls["repair"] == 0
