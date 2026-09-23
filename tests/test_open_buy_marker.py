"""Tests for scripts/_open_buy_marker 明日开盘勿买 (legacy+parallel+genious 三交付).

规则 (config OPEN_BUY_RISK): T 日涨幅 ≥ pct_rise (7%) 或
(amplitude_5d 当日截面分位 ≥ 0.8 且 winner_ratio ≥ 0.8) → 勿买·防高开低走。
删票闸 apply_open_buy_kill: flagged 且未过涨闸 → 删 (分线开关 kill)。
"""

import pandas as pd

from scripts._open_buy_marker import apply_open_buy_kill, open_buy_marker

TRADE_DATE = "20260921"


def _panel(tmp_path, rows):
    """rows: [{symbol, pctChg, amplitude_5d, winner_ratio}] → 当日单截面 parquet."""
    df = pd.DataFrame(
        [
            {
                "symbol": r["symbol"],
                "date": pd.Timestamp(TRADE_DATE),
                "pctChg": r["pctChg"],
                "amplitude_5d": r["amplitude_5d"],
                "winner_ratio": r["winner_ratio"],
            }
            for r in rows
        ]
    )
    fp = tmp_path / "panel.parquet"
    df.to_parquet(fp, index=False)
    return fp


def _picks(symbols):
    return pd.DataFrame({"symbol": symbols})


def _many(n, pct=0.0, amp=1.0, wr=0.5):
    """n 只背景票 (不命中任一触发) — 撑截面分位."""
    return [
        {"symbol": f"6{i:05d}", "pctChg": pct, "amplitude_5d": amp, "winner_ratio": wr}
        for i in range(n)
    ]


def test_big_rise_flagged(tmp_path):
    # 触发1: T 日涨幅 9.9% ≥ 7% (即便获利浅/波幅低)
    rows = _many(20) + [{"symbol": "300001", "pctChg": 9.9, "amplitude_5d": 1.0, "winner_ratio": 0.2}]
    out = open_buy_marker(_picks(["300001", "600001"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == "勿买·防高开低走"
    assert out.loc[1, "明日开盘"] == ""


def test_hot_deep_chips_flagged(tmp_path):
    # 触发2: 涨幅 2% (<7) 但波幅截面最高 (rank≥0.8) 且 获利盘 0.85 ≥ 0.8
    rows = _many(20) + [{"symbol": "300002", "pctChg": 2.0, "amplitude_5d": 9.9, "winner_ratio": 0.85}]
    out = open_buy_marker(_picks(["300002"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == "勿买·防高开低走"


def test_hot_shallow_chips_not_flagged(tmp_path):
    # 波幅高但获利盘 0.5 < 0.8 → 不标 (深获利是必要成分)
    rows = _many(20) + [{"symbol": "300003", "pctChg": 2.0, "amplitude_5d": 9.9, "winner_ratio": 0.5}]
    out = open_buy_marker(_picks(["300003"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == ""


def test_calm_not_flagged(tmp_path):
    rows = _many(20) + [{"symbol": "300004", "pctChg": 1.5, "amplitude_5d": 1.2, "winner_ratio": 0.3}]
    out = open_buy_marker(_picks(["300004"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == ""


def test_missing_panel_fail_open(tmp_path):
    out = open_buy_marker(_picks(["300001"]), TRADE_DATE, panel_path=tmp_path / "nope.parquet")
    assert "明日开盘" in out.columns
    assert (out["明日开盘"] == "").all()


def test_symbol_absent_from_panel(tmp_path):
    rows = _many(20)
    out = open_buy_marker(_picks(["999999"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == ""


def test_boundary_pct_rise(tmp_path):
    # 恰好 7.0 = 命中 (≥ 闭区间)
    rows = _many(20) + [{"symbol": "300005", "pctChg": 7.0, "amplitude_5d": 1.0, "winner_ratio": 0.2}]
    out = open_buy_marker(_picks(["300005"]), TRADE_DATE, panel_path=_panel(tmp_path, rows))
    assert out.loc[0, "明日开盘"] == "勿买·防高开低走"


def _flagged_df(with_gate_col=False):
    rows = [
        {"symbol": "300001", "flag": True, "gate": "过闸"},
        {"symbol": "600002", "flag": True, "gate": "没过闸"},
        {"symbol": "600003", "flag": False, "gate": "没过闸"},
    ]
    df = pd.DataFrame(rows)
    df["明日开盘"] = df["flag"].map({True: "勿买·防高开低走", False: ""})
    return df if with_gate_col else df.drop(columns=["gate", "flag"])


def test_kill_line_disabled_noop(monkeypatch):
    # parallel 线开关 False → 原样返回 (不删不调涨闸)
    monkeypatch.setattr("scripts._open_buy_marker._gate_exempt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调涨闸")))
    df = _flagged_df()
    out = apply_open_buy_kill(df, TRADE_DATE, line="parallel")
    assert len(out) == 3


def test_kill_removes_flagged_not_passed_gate():
    # genious 线 (自带涨闸列): flagged+没过闸 → 删; flagged+过闸 → 豁免; 未标 → 留
    df = _flagged_df(with_gate_col=True).drop(columns=["flag"]).rename(columns={"gate": "涨闸"})
    out = apply_open_buy_kill(df, TRADE_DATE, line="genious")
    assert list(out["symbol"]) == ["300001", "600003"]


def test_kill_gate_recompute_exempts(monkeypatch):
    # legacy 线 (无涨闸列) → _gate_exempt 复算: 600002 过闸豁免, 300001 删
    df = _flagged_df()
    monkeypatch.setattr("scripts._open_buy_marker._gate_exempt", lambda syms, td, pp=None: {"600002"})
    out = apply_open_buy_kill(df, TRADE_DATE, line="legacy")
    assert list(out["symbol"]) == ["600002", "600003"]


def test_kill_failopen_on_gate_error(monkeypatch):
    # 涨闸复算抛错 → fail-open 一只不删 (标注仍在)
    def boom(syms, td, pp=None):
        raise RuntimeError("panel 读失败")

    monkeypatch.setattr("scripts._open_buy_marker._gate_exempt", boom)
    df = _flagged_df()
    out = apply_open_buy_kill(df, TRADE_DATE, line="legacy")
    assert len(out) == 3


def test_kill_no_flagged_noop(monkeypatch):
    monkeypatch.setattr("scripts._open_buy_marker._gate_exempt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("无 flagged 不该调涨闸")))
    df = pd.DataFrame({"symbol": ["600003"], "明日开盘": [""]})
    out = apply_open_buy_kill(df, TRADE_DATE, line="legacy")
    assert len(out) == 1
