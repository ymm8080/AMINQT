"""LEGACY_SELECTION mode="prob10_pull" 接线测试 (2026-09-07).

依据 (tmp_t/_rankkey_prob_vs_pred_0907.py, 125d 同窗同口径): E7 闸内换排名键 =
空操作 (B1≡A1 逐字节); 胜出来自撤闸 — 纯 prob10+回撤闸 13.8只/日 44.2%/+6.42pp
全轴胜旧栈 (E7+pred键+wr5) 11.0只/日 33.2%/+3.77pp。wr5 在该臂毁值 (被切票赢率
47% > 留守 42.8%) 一并摘除 (_deliver_legacy_list 同开关); 密度/PARALLEL 两线不动。
语义锁: 排序 → 板内 top_n → 回撤闸 (真删不补齐, 不 refill)。
"""

from __future__ import annotations

import importlib
import inspect

import pandas as pd
import pytest

from app.pipeline1 import list_generator as lg
from app.pipeline1 import prob_head
from app.pipeline1.list_generator import ListGenerator

GATE_OFF = {"entry_prob": 0.0, "entry_ret_mult": 0.0}


@pytest.fixture
def sel_on(monkeypatch):
    from config.settings import LEGACY_SELECTION

    monkeypatch.setitem(LEGACY_SELECTION, "enable", True)
    monkeypatch.setitem(LEGACY_SELECTION, "mode", "prob10_pull")
    monkeypatch.setitem(LEGACY_SELECTION, "pull_min", -0.10)
    monkeypatch.setitem(LEGACY_SELECTION, "board_top_n", 10)


def _panel_fp(tmp_path, frame):
    fp = tmp_path / "panel_v3.parquet"
    frame.to_parquet(fp, index=False)
    return fp


def _make_panel(
    cut_symbol, end, syms=("600001", "600002", "600003", "300001", "688001")
):
    """12 个交易日面板: cut_symbol 末日暴跌 → pull=-20%; 其余恒价 → pull=0."""
    dates = pd.bdate_range(end=end, periods=12)
    rows = []
    for sym in syms:
        for i, d in enumerate(dates):
            px = 8.0 if (sym == cut_symbol and i == len(dates) - 1) else 10.0
            rows.append({"symbol": sym, "date": d, "close_hfq": px})
    return pd.DataFrame(rows)


def _candidates() -> pd.DataFrame:
    """prob 降序 ≠ pred 降序 (600002 prob 最高但 pred 最低) — 用于证明换栈生效."""
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600003", "300001", "688001"],
            "board": ["main", "main", "main", "GEM", "STAR"],
            "date": pd.Timestamp("2026-06-20"),
            "pred_ret_3d": [0.01] * 5,
            "pred_ret_5d": [0.01] * 5,
            "pred_ret_10d": [0.030, 0.020, 0.010, 0.005, 0.004],
            "prob_up": [0.6] * 5,
            "prob_up_10d": [0.60, 0.90, 0.75, 0.50, 0.40],
        }
    )


def test_select_pull_cuts_no_refill(tmp_path, monkeypatch, sel_on):
    """排序 → 板内 top_n=2 → 回撤闸; 被切位不 refill.

    main prob 降序 = [600002(0.90), 600003(0.75), 600001(0.60)]; head(2) 截掉
    600001 后回撤切 600002 → main 只剩 600003 (600001 不得补进).
    """
    from config.settings import LEGACY_SELECTION

    monkeypatch.setitem(LEGACY_SELECTION, "board_top_n", 2)
    fp = _panel_fp(tmp_path, _make_panel("600002", pd.Timestamp("2026-06-20")))
    monkeypatch.setattr(lg, "PANEL_V3_PATH", fp)
    out = ListGenerator._select_prob10_pull(_candidates())
    assert out["symbol"].tolist() == ["600003", "300001", "688001"]
    assert "600002" not in out["symbol"].tolist()  # 回撤闸真删
    assert "600001" not in out["symbol"].tolist()  # top_n 截断在闸前 → 不 refill


def test_select_pull_failopen_without_panel(tmp_path, monkeypatch, sel_on):
    """面板读不到 → 回撤闸整体跳过 (fail-open), prob 排序 + 板内截断照常."""
    monkeypatch.setattr(lg, "PANEL_V3_PATH", tmp_path / "missing.parquet")
    out = ListGenerator._select_prob10_pull(_candidates())
    assert out["symbol"].tolist() == [
        "600002",
        "600003",
        "600001",
        "300001",
        "688001",
    ]


def test_select_pull_fallback_to_mag_key_when_prob_missing(sel_on):
    """prob_up_10d 缺列 (旧 bundle) → 级联回退幅度键, 不炸."""
    cands = _candidates().drop(columns=["prob_up_10d"])
    out = ListGenerator._select_prob10_pull(cands)
    assert out["symbol"].tolist()[:3] == ["600001", "600002", "600003"]


def test_emit_new_mode_skips_prob_gate_and_mag_rank(tmp_path, monkeypatch, sel_on):
    """新栈: prob_gate 不调用; 幅度键不再重排; 回撤闸生效; board 原样保留."""
    calls: list[int] = []
    monkeypatch.setattr(
        prob_head, "gate_probabilities", lambda *a, **k: calls.append(1)
    )
    fp = _panel_fp(tmp_path, _make_panel("600002", pd.Timestamp("2026-06-20")))
    monkeypatch.setattr(lg, "PANEL_V3_PATH", fp)
    out = ListGenerator(**GATE_OFF).emit(_candidates())
    assert not out["empty"]
    # prob 降序 [600002(切), 600003, 600001, ...] → 交付 [600003, 600001, ...];
    # 幅度序首位会是 600001 (0.030) — 首位 600003 证明幅度键未重排
    lst = out["list"]
    assert lst["symbol"].tolist() == ["600003", "600001", "300001", "688001"]
    assert calls == []  # prob_gate 未被调用


def test_mode_e7_pred_restores_old_stack(monkeypatch):
    """回退开关: mode="e7_pred" → 旧栈 (幅度键排名) 原样."""
    from config.settings import LEGACY_SELECTION

    monkeypatch.setitem(LEGACY_SELECTION, "enable", True)
    monkeypatch.setitem(LEGACY_SELECTION, "mode", "e7_pred")
    out = ListGenerator(**GATE_OFF).emit(_candidates())
    assert out["list"]["symbol"].tolist() == [
        "600001",
        "600002",
        "600003",
        "300001",
        "688001",
    ]


def test_deliver_wr5_gate_guarded_by_mode():
    """交付端 wr5 派发闸被 LEGACY_SELECTION 开关守卫 (新栈摘除, 旧栈保留)."""
    legacy = importlib.import_module("scripts._deliver_legacy_list")
    src = inspect.getsource(legacy.main)
    assert "prob10_pull" in src  # 守卫存在
    assert "apply_chip_gate(df" in src  # 调用保留 (旧栈路径)
