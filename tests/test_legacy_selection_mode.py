"""LEGACY_SELECTION mode="prob10_pull" 接线测试 (2026-09-07).

依据 (tmp_t/_rankkey_prob_vs_pred_0907.py, 125d 同窗同口径): E7 闸内换排名键 =
空操作 (B1≡A1 逐字节); 胜出来自撤闸 — 纯 prob10+回撤闸 13.8只/日 44.2%/+6.42pp
全轴胜旧栈 (E7+pred键+wr5) 11.0只/日 33.2%/+3.77pp。wr5 在该臂毁值 (被切票赢率
47% > 留守 42.8%) 一并摘除 (_deliver_legacy_list 同开关); 密度/PARALLEL 两线不动。
语义锁: 排序 → 趋势闸 (MA10↑ 先滤后截, 补位) → 板内 top_n → 回撤闸 (真删不补齐)。
[0913 趋势闸] 数据驱动定案 (tmp_t/_trend_gate_sweep_0913.py, 22 夜存档): MA10↑ 闸
riser5 54.1% vs 无闸 53.6%, fail_trend 17.7%→10%; 5D/益盟做键与组合键全劣。
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
    monkeypatch.setitem(LEGACY_SELECTION, "trend_gate", "ma10_up")
    monkeypatch.setitem(LEGACY_SELECTION, "board_top_n", 10)


def _panel_fp(tmp_path, frame):
    fp = tmp_path / "panel_v3.parquet"
    frame.to_parquet(fp, index=False)
    return fp


def _make_panel(
    cut_symbol, end, final_px=8.9, syms=("600001", "600002", "600003", "300001", "688001")
):
    """12 交易日斜坡面板 (5.0→10.5, MA10 恒升); cut_symbol 末日 = final_px:
    8.9 → MA10 仍升 (6.5..10 均值 8.0 → 换入 8.9 后 8.09) 但 pull=−11%;
    ≤5.0 → MA10 拐头向下 (趋势闸切)."""
    dates = pd.bdate_range(end=end, periods=12)
    rows = []
    for sym in syms:
        for i, d in enumerate(dates):
            px = 5.0 + 0.5 * i
            if sym == cut_symbol and i == len(dates) - 1:
                px = final_px
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
    """排序 → 趋势闸全过 → 板内 top_n=2 → 回撤闸; 被切位不 refill.

    main prob 降序 = [600002(0.90), 600003(0.75), 600001(0.60)]; 600002 末日 8.9
    (MA10 仍升过趋势闸, pull=−11%) → head(2) 截掉 600001 后回撤切 600002 →
    main 只剩 600003 (600001 不得补进).
    """
    from config.settings import LEGACY_SELECTION

    monkeypatch.setitem(LEGACY_SELECTION, "board_top_n", 2)
    fp = _panel_fp(tmp_path, _make_panel("600002", pd.Timestamp("2026-06-20")))
    monkeypatch.setattr(lg, "PANEL_V3_PATH", fp)
    out = ListGenerator._select_prob10_pull(_candidates())
    assert out["symbol"].tolist() == ["600003", "300001", "688001"]
    assert "600002" not in out["symbol"].tolist()  # 回撤闸真删
    assert "600001" not in out["symbol"].tolist()  # top_n 截断在闸前 → 不 refill


def test_select_trend_gate_filters_and_refills(tmp_path, monkeypatch, sel_on):
    """[0913 趋势闸] MA10↓ 最高 prob 票先滤 (600002 末日 5.0 拐头), 板内 top_n
    截断时低位票补位 — 闸在截断前 = refill 语义 (与回撤闸真删相反)."""
    fp = _panel_fp(
        tmp_path, _make_panel("600002", pd.Timestamp("2026-06-20"), final_px=5.0)
    )
    monkeypatch.setattr(lg, "PANEL_V3_PATH", fp)
    out = ListGenerator._select_prob10_pull(_candidates())
    assert "600002" not in out["symbol"].tolist()  # 趋势闸切 (MA10↓)
    assert out["symbol"].tolist() == ["600003", "600001", "300001", "688001"]
    assert (out["pull_flag"] == "").all()  # 补位票无深回撤 → 无标注


def test_select_trend_gate_off_restores_ungated(tmp_path, monkeypatch, sel_on):
    """旋钮 trend_gate="off" → 趋势闸关: 同面板下 600002 过闸进 head(2),
    再被回撤闸真删 → main 只剩 600003 (与闸开时的补位 [600003, 600001] 区分)."""
    from config.settings import LEGACY_SELECTION

    monkeypatch.setitem(LEGACY_SELECTION, "trend_gate", "off")
    monkeypatch.setitem(LEGACY_SELECTION, "board_top_n", 2)
    fp = _panel_fp(
        tmp_path, _make_panel("600002", pd.Timestamp("2026-06-20"), final_px=5.0)
    )
    monkeypatch.setattr(lg, "PANEL_V3_PATH", fp)
    out = ListGenerator._select_prob10_pull(_candidates())
    assert out["symbol"].tolist() == ["600003", "300001", "688001"]


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


def test_select_pull_fallback_to_mag_key_when_prob_missing(
    tmp_path, monkeypatch, sel_on
):
    """prob_up_10d 缺列 (旧 bundle) → 级联回退幅度键, 不炸.

    面板指到缺失路径 (fail-open): 真实面板 MA10 涨跌不定, 会让排序断言随机红."""
    monkeypatch.setattr(lg, "PANEL_V3_PATH", tmp_path / "missing.parquet")
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


def test_deliver_chip_mark_unconditional():
    """交付端派发标注不分模式 (09-09 改标注后 prob10_pull 守卫已废):
    两种 LEGACY_SELECTION 模式都带 chip_flag 标注."""
    legacy = importlib.import_module("scripts._deliver_legacy_list")
    src = inspect.getsource(legacy.main)
    assert "    df = apply_chip_gate(df, pd.Timestamp(trade_date))" in src
    # 4空格缩进 = main 顶层恒调用, 旧 prob10_pull 模式守卫不得回归
