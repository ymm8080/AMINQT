"""冲高回落闸单测 (2026-09-09): 删线 + 事件提示 + fail-open + 因果因子计算 + WORM 留痕."""

import pandas as pd

from scripts import _fade_gate as gate

CFG = {"enable": True, "kill_score": 0.75, "flag_enable": True}


def _mk(vals):
    return pd.DataFrame(
        {"symbol": [f"{i:06d}" for i in range(len(vals))], "board": "main", "v": vals}
    )


def _prof(scores, today=()):
    idx = [f"{i:06d}" for i in range(len(scores))]
    return {
        "score": pd.Series(scores, index=idx, name="fade_score"),
        "fade_today": pd.Series([i in set(today) for i in idx], index=idx),
    }


def test_kill_by_threshold_and_write_doc(tmp_path):
    out = gate.apply_fade_gate(
        _mk([0.1] * 8 + [0.8, 0.95]),
        "2026-09-09",
        "m_test",
        profile=_prof([0.1] * 8 + [0.8, 0.95]),
        cfg=CFG,
        list_dir=tmp_path,
        line="legacy",
    )
    assert list(out["symbol"]) == [f"{i:06d}" for i in range(8)]
    doc = tmp_path / "faderemoved_20260909__m_test__legacy.csv"
    assert doc.exists()
    cut = pd.read_csv(doc, dtype={"symbol": str})
    assert sorted(cut["symbol"]) == ["000008", "000009"]
    # 删的票带 fade_score, 留的也带 (CSV 透明)
    assert out["fade_score"].iloc[0] == 0.1


def test_boundary_equal_is_killed(tmp_path):
    # ≥0.75 即删 (回放口径)
    out = gate.apply_fade_gate(
        _mk([0.75, 0.749]),
        "2026-09-09",
        "m",
        profile=_prof([0.75, 0.749]),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert list(out["symbol"]) == ["000001"]


def test_nan_score_never_deleted(tmp_path):
    out = gate.apply_fade_gate(
        _mk([0.9, 0.3, None]),
        "2026-09-09",
        "m",
        profile=_prof([0.9, 0.3, float("nan")]),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert "000002" in set(out["symbol"])  # 无因子 fail-open 不删
    assert "000000" not in set(out["symbol"])


def test_fade_today_flag_not_deleted(tmp_path):
    out = gate.apply_fade_gate(
        _mk([0.5, 0.6]),
        "2026-09-09",
        "m",
        profile=_prof([0.5, 0.6], today=("000001",)),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert len(out) == 2  # 事件型只标不删
    assert out["fade_flag"].iloc[1] != ""  # 000001 被标
    assert out["fade_flag"].iloc[0] == ""
    assert not (tmp_path / "faderemoved_20260909__m.csv").exists()


def test_failopen_when_profile_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "compute_fade_profile", lambda *a, **k: None)
    df = _mk([0.9, 0.1])
    out = gate.apply_fade_gate(df, "2026-09-09", "m", cfg=CFG, list_dir=tmp_path)
    assert len(out) == 2
    assert "fade_flag" not in out.columns


def test_disabled_noop(tmp_path):
    out = gate.apply_fade_gate(
        _mk([0.9, 0.1]),
        "2026-09-09",
        "m",
        profile=_prof([0.9, 0.1]),
        cfg={"enable": False},
        list_dir=tmp_path,
    )
    assert len(out) == 2
    assert not (tmp_path / "faderemoved_20260909__m.csv").exists()


def test_kill_disabled_still_scores(tmp_path):
    # 09-09 撤删线: kill_enable=False → 不删但 fade_score 列照给 (透明)
    out = gate.apply_fade_gate(
        _mk([0.9, 0.1]),
        "2026-09-09",
        "m",
        profile=_prof([0.9, 0.1]),
        cfg={"enable": True, "kill_enable": False, "flag_enable": True},
        list_dir=tmp_path,
    )
    assert len(out) == 2
    assert "fade_score" in out.columns
    assert out["fade_score"].iloc[0] == 0.9
    assert not (tmp_path / "faderemoved_20260909__m.csv").exists()


def test_empty_df_passthrough(tmp_path):
    out = gate.apply_fade_gate(
        _mk([]), "2026-09-09", "m", profile=_prof([]), cfg=CFG, list_dir=tmp_path
    )
    assert len(out) == 0


def _synth_panel(tmp_path):
    """3 只 × 70 日合成面板: 000001 全维最极端 (高波/高换/已涨/贴高点) → 分最高;
    末日 000002 冲高回落 (g≥3%, 吐回≥70%), 000001/000003 无."""
    dates = pd.date_range("2026-06-01", periods=70)
    rows = []
    prof = {
        "000001": dict(base=10.0, drift=0.06, vol=3.0, turn=9.0),   # 暴涨高波高换
        "000002": dict(base=20.0, drift=0.0, vol=0.4, turn=1.0),    # 平稳
        "000003": dict(base=50.0, drift=-0.01, vol=0.8, turn=2.0),  # 阴跌
    }
    for s, p in prof.items():
        for i, dt in enumerate(dates):
            close = p["base"] * (1 + p["drift"]) ** i + (i % 2) * p["vol"] * 0.05
            high = close * (1 + p["vol"] * 0.01)
            rows.append({
                "symbol": s, "date": dt, "high": high, "close": close,
                "pre_close": close, "turnover_rate": p["turn"],
            })
    df = pd.DataFrame(rows)
    # 末日: 000002 冲高 g=5% 后收在 +1.4% → 吐回 (5-1.4)/5 = 72%
    m = (df["symbol"] == "000002") & (df["date"] == dates[-1])
    pc = float(df.loc[m, "close"].iloc[0])
    df.loc[m, "pre_close"] = pc
    df.loc[m, "high"] = pc * 1.05
    df.loc[m, "close"] = pc * 1.014
    # 000001 末日收在最高 (否则高=收×1.03+收=前收 → 恒 100% 吐回被误标)
    m1 = (df["symbol"] == "000001") & (df["date"] == dates[-1])
    df.loc[m1, "close"] = df.loc[m1, "high"]
    fp = tmp_path / "panel.parquet"
    df.to_parquet(fp, index=False)
    return fp, dates[-1]


def test_compute_profile_causal_synthetic(tmp_path, monkeypatch):
    fp, last = _synth_panel(tmp_path)
    monkeypatch.setattr(gate, "PANEL_V3_PATH", fp)
    prof = gate.compute_fade_profile(last, panel_path=fp)
    assert prof is not None
    assert prof["score"].idxmax() == "000001"  # 高波高换已涨贴高点 → 最易冲高回落
    assert prof["score"].idxmin() == "000003"  # 阴跌股远离高点+低换手 → 最不易
    assert prof["fade_today"]["000002"]
    assert not prof["fade_today"]["000001"]


def test_compute_profile_stale_panel_no_today_flag(tmp_path, monkeypatch):
    # 面板末行 < day_ts (当日未入库) → fade_today 全 False, score 仍可算
    fp, last = _synth_panel(tmp_path)
    monkeypatch.setattr(gate, "PANEL_V3_PATH", fp)
    prof = gate.compute_fade_profile(last + pd.Timedelta(days=1), panel_path=fp)
    assert prof is not None
    assert not prof["fade_today"].any()
