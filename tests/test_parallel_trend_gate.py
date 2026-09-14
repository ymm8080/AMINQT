"""PARALLEL 趋势闸接线测试 (2026-09-14 用户令 "起码基本MA10必须上升").

依据 (tmp_t/_trend_gate_parallel_sweep_0914.py, 21 夜 0806-0904, WORM
diag/trend_gate_parallel_sweep_0914_*.json): 0913 夜清单 15/16 派发标注且
MA10↑ 仅 1/16 (300808 且为单日 −18.3% 见顶崩盘票) → MA10↑ 闸 + 单日大跌
守卫, rank_and_truncate 前先滤后截 (补位语义, 同 LEGACY 0913 趋势闸)。
crash8 守卫 main 净5 +0.06pp 全变体最优; MA10↑ 风格代价披露见
settings.PARALLEL_TREND_GATE 注释。
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import _shortlist_t5_t10 as mod

GATE_ON = {"enable": True, "mode": "ma10_up", "crash_max_1d": -0.08}


def _panel_fp(tmp_path, frame):
    fp = tmp_path / "panel_v3.parquet"
    frame.to_parquet(fp, index=False)
    return fp


def _make_panel(crash_symbol=None, syms=("600001", "600002", "600003")):
    """12 交易日斜坡面板 (末 2026-09-11, 5.0→10.5, MA10 恒升); crash_symbol
    末日砸到 8.0: MA10 仍升 (换入 8.0 后均值上升) 但单日 −24% → crash 守卫切
    (300808 型)."""
    dates = pd.bdate_range(end=pd.Timestamp("2026-09-11"), periods=12)
    rows = []
    for sym in syms:
        for i, d in enumerate(dates):
            px = 5.0 + 0.5 * i
            if sym == crash_symbol and i == len(dates) - 1:
                px = 8.0
            rows.append({"symbol": sym, "date": d, "close_hfq": px})
    return pd.DataFrame(rows)


def _make_panel_falling(fall_symbol="600002", syms=("600001", "600002", "600003")):
    """fall_symbol 缓降 −2%/日 (末 2026-09-11, MA10 恒降, 无单日大跌 → 过
    crash 守卫); 其余斜坡上升."""
    dates = pd.bdate_range(end=pd.Timestamp("2026-09-11"), periods=12)
    rows = []
    for sym in syms:
        for i, d in enumerate(dates):
            px = 5.0 + 0.5 * i
            if sym == fall_symbol:
                px = 10.5 * 0.98**i
            rows.append({"symbol": sym, "date": d, "close_hfq": px})
    return pd.DataFrame(rows)


def _make_panel_spike_fade(spike_symbol="600002", syms=("600001", "600002", "600003")):
    """spike_symbol 冲高回落 (603798 型): 7 日平 10 → 3 日冲高 14/13/12 →
    回落 10.4/9.8。MA5=11.84 > MA10=10.92 (过 MA5>MA10), 单日 −5.8% (过
    crash 守卫), 但近 5 日净跌 9.8<10 (ret5<0) → 仅 ma5gt10_up5 杀,
    ma5_gt_10 放行 (up5 子句的存在依据)."""
    dates = pd.bdate_range(end=pd.Timestamp("2026-09-11"), periods=12)
    spike = [10.0] * 7 + [14.0, 13.0, 12.0, 10.4, 9.8]
    rows = []
    for sym in syms:
        for i, d in enumerate(dates):
            px = 5.0 + 0.5 * i
            if sym == spike_symbol:
                px = spike[i]
            rows.append({"symbol": sym, "date": d, "close_hfq": px})
    return pd.DataFrame(rows)


def _res(extra_syms=()):
    syms = ["600001", "600002", "600003"] + list(extra_syms)
    return pd.DataFrame(
        {
            "symbol": syms,
            "board": ["main"] * len(syms),
            "score": [0.9, 0.8, 0.7] + [0.5] * len(extra_syms),
        }
    )


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(mod, "PARALLEL_TREND_GATE", dict(GATE_ON))


def test_trend_gate_cuts_falling_ma10(tmp_path, monkeypatch, gate_on):
    """MA10↓ 票 (600002 持续下跌) 被切, 其余过闸."""
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_falling())
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert "600002" not in out["symbol"].tolist()
    assert set(out["symbol"]) == {"600001", "600003"}


def test_trend_gate_crash_guard_cuts_top_crash(tmp_path, monkeypatch, gate_on):
    """300808 型: MA10 仍升但评分日单日 −24% → crash 守卫切 (MA10↑ 单闸漏)."""
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel(crash_symbol="600002"))
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert "600002" not in out["symbol"].tolist()
    assert set(out["symbol"]) == {"600001", "600003"}


def test_trend_gate_missing_symbol_passes(tmp_path, monkeypatch, gate_on):
    """面板缺行个股过闸 (fail-open, 同 LEGACY 语义): .BJ 后缀票不在面板 → 留."""
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_falling())
    )
    out = mod.apply_trend_gate(_res(["920367.BJ"]), pd.Timestamp("2026-09-11"))
    assert "920367.BJ" in out["symbol"].tolist()


def test_trend_gate_disabled_noop(tmp_path, monkeypatch):
    """enable=False → 原样返回 (含 MA10↓ 票)."""
    monkeypatch.setattr(mod, "PARALLEL_TREND_GATE", {"enable": False})
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_falling())
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert len(out) == 3


def test_trend_gate_crash_only_mode(tmp_path, monkeypatch):
    """mode="off" + crash 保留 → 只切崩盘票, MA10↓ 票留守 (旋钮粒度)."""
    monkeypatch.setattr(
        mod,
        "PARALLEL_TREND_GATE",
        {"enable": True, "mode": "off", "crash_max_1d": -0.08},
    )
    monkeypatch.setattr(
        mod,
        "PANEL_V3_PATH",
        _panel_fp(tmp_path, _make_panel_falling()),  # 600002 MA10↓ 无崩盘日
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert set(out["symbol"]) == {"600001", "600002", "600003"}


def test_trend_gate_failopen_on_panel_error(tmp_path, monkeypatch, gate_on):
    """面板读不到 → 整体跳过 (fail-open), 清单照出."""
    monkeypatch.setattr(mod, "PANEL_V3_PATH", tmp_path / "missing.parquet")
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert len(out) == 3


def test_trend_gate_ma5gt10_up5_cuts_falling(tmp_path, monkeypatch):
    """[已测旋钮 mode] MA5>MA10 & 近5日净涨: 缓跌票 (MA5<MA10) 被切.
    生产 mode=ma10_up (0914 用户终令), MA5 族为回放已测回退旋钮."""
    monkeypatch.setattr(
        mod,
        "PARALLEL_TREND_GATE",
        {"enable": True, "mode": "ma5gt10_up5", "crash_max_1d": -0.08},
    )
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_falling())
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert set(out["symbol"]) == {"600001", "600003"}


def test_trend_gate_up5_cuts_spike_fade(tmp_path, monkeypatch):
    """冲高回落票 (MA5>MA10 成立 + 单日跌幅过守卫, 但近5日净跌) → up5 子句杀
    (603798 型: 近5日 −10.2% 仍过单 MA5>MA10 闸)."""
    monkeypatch.setattr(
        mod,
        "PARALLEL_TREND_GATE",
        {"enable": True, "mode": "ma5gt10_up5", "crash_max_1d": -0.08},
    )
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_spike_fade())
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert "600002" not in out["symbol"].tolist()
    assert set(out["symbol"]) == {"600001", "600003"}


def test_trend_gate_ma5gt10_only_passes_spike_fade(tmp_path, monkeypatch):
    """mode=ma5_gt_10 (无 up5 子句): 同一冲高回落票放行 — up5 子句的存在依据
    (回放: 单 MA5>MA10 仍放行 603798, 近5日 −10.2%)."""
    monkeypatch.setattr(
        mod,
        "PARALLEL_TREND_GATE",
        {"enable": True, "mode": "ma5_gt_10", "crash_max_1d": -0.08},
    )
    monkeypatch.setattr(
        mod, "PANEL_V3_PATH", _panel_fp(tmp_path, _make_panel_spike_fade())
    )
    out = mod.apply_trend_gate(_res(), pd.Timestamp("2026-09-11"))
    assert set(out["symbol"]) == {"600001", "600002", "600003"}


def test_trend_gate_per_board_mode(tmp_path, monkeypatch):
    """[0914 按板拍板] mode 分板 dict: main=ma10_up 切连跌票, dual=off 只留
    crash → 同样连跌的 dual 票留守 (dual 闸双视界全亏, 见 settings 注释)."""
    monkeypatch.setattr(
        mod,
        "PARALLEL_TREND_GATE",
        {
            "enable": True,
            "mode": {"main": "ma10_up", "dual": "off"},
            "crash_max_1d": -0.08,
        },
    )
    dates = pd.bdate_range(end=pd.Timestamp("2026-09-11"), periods=12)
    paths = {
        "600001": [5.0 + 0.5 * i for i in range(12)],  # main 升 (过闸)
        "600002": [10.5 * 0.98**i for i in range(12)],  # main 缓跌 MA10↓ → 切
        "300001": [5.0 + 0.5 * i for i in range(12)],  # dual 升 (过)
        "300002": [10.5 * 0.98**i for i in range(12)],  # dual 缓跌 MA10↓ → off 留守
    }
    rows = [
        {"symbol": s, "date": d, "close_hfq": px}
        for s, p in paths.items()
        for d, px in zip(dates, p)
    ]
    monkeypatch.setattr(mod, "PANEL_V3_PATH", _panel_fp(tmp_path, pd.DataFrame(rows)))
    res = pd.DataFrame(
        {
            "symbol": ["600001", "600002", "300001", "300002"],
            "board": ["main", "main", "dual", "dual"],
            "score": [0.9, 0.8, 0.7, 0.6],
        }
    )
    out = mod.apply_trend_gate(res, pd.Timestamp("2026-09-11"))
    assert set(out["symbol"]) == {"600001", "300001", "300002"}
    assert "600002" not in out["symbol"].tolist()


def test_trend_gate_wired_before_rank_and_truncate():
    """闸位锁: apply_trend_gate 调用在 rank_and_truncate 之前 (先滤后截补位),
    且在 full_res 赋值前 (滞留候选同过闸)."""
    src = open(mod.__file__, encoding="utf-8").read()
    assert src.index("res = apply_trend_gate(res, sel_date)") < src.index(
        "full_res = res"
    )
    assert src.index("full_res = res") < src.index("res = rank_and_truncate(res)")
