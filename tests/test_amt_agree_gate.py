"""量价删查线单测 (2026-09-08): 删档规则 + fail-open + 因果因子计算 + WORM 留痕."""

import pandas as pd

from scripts import _amt_agree_gate as gate

CFG = {"enable": True, "del_top": 0.2, "window": 10}


def _mk(vals):
    return pd.DataFrame(
        {"symbol": [f"{i:06d}" for i in range(len(vals))], "board": "main", "v": vals}
    )


def _ag(vals):
    return pd.Series({f"{i:06d}": v for i, v in enumerate(vals)}, name="amt_agree10")


def test_kill_top20_and_write_doc(tmp_path):
    # 10 只可算 → k=ceil(10×0.2)=2, 删因子最高两只
    out = gate.apply_amt_agree_kill(
        _mk([0.1] * 8 + [0.9, 0.95]),
        "2026-09-08",
        "m_test",
        ag=_ag([0.1] * 8 + [0.9, 0.95]),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert list(out["symbol"]) == [f"{i:06d}" for i in range(8)]
    doc = tmp_path / "amtagree_removed_20260908__m_test.csv"
    assert doc.exists()
    cut = pd.read_csv(doc, dtype={"symbol": str})
    assert sorted(cut["symbol"]) == ["000008", "000009"]


def test_small_list_min_one(tmp_path):
    out = gate.apply_amt_agree_kill(
        _mk([0.2, 0.5, 0.1]),
        "2026-09-08",
        "m",
        ag=_ag([0.2, 0.5, 0.1]),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert len(out) == 2
    assert "000001" not in set(out["symbol"])


def test_nan_factor_pick_never_deleted(tmp_path):
    # 000003 无因子 → 不删; k 只数可算 3 只 → k=1 删 000000
    out = gate.apply_amt_agree_kill(
        _mk([0.9, 0.3, 0.1, 0.5]),
        "2026-09-08",
        "m",
        ag=_ag([0.9, 0.3, 0.1]),
        cfg=CFG,
        list_dir=tmp_path,
    )
    assert "000003" in set(out["symbol"])
    assert "000000" not in set(out["symbol"])


def test_failopen_when_factor_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "compute_amt_agree10", lambda *a, **k: None)
    df = _mk([0.9, 0.1])
    out = gate.apply_amt_agree_kill(df, "2026-09-08", "m", cfg=CFG, list_dir=tmp_path)
    assert len(out) == 2
    assert not (tmp_path / "amtagree_removed_20260908__m.csv").exists()


def test_line_tag_in_doc_name(tmp_path):
    # 两线共享模块 tag → 留痕文件名靠 line 区分, 防同名互覆 (09-08 首跑实发)
    gate.apply_amt_agree_kill(
        _mk([0.9, 0.1]),
        "2026-09-08",
        "m",
        ag=_ag([0.9, 0.1]),
        cfg=CFG,
        list_dir=tmp_path,
        line="legacy",
    )
    assert (tmp_path / "amtagree_removed_20260908__m__legacy.csv").exists()
    assert not (tmp_path / "amtagree_removed_20260908__m.csv").exists()


def test_disabled_noop(tmp_path):
    out = gate.apply_amt_agree_kill(
        _mk([0.9, 0.1]),
        "2026-09-08",
        "m",
        ag=_ag([0.9, 0.1]),
        cfg={"enable": False},
        list_dir=tmp_path,
    )
    assert len(out) == 2
    assert not (tmp_path / "amtagree_removed_20260908__m.csv").exists()


def test_empty_df_passthrough(tmp_path):
    out = gate.apply_amt_agree_kill(
        _mk([]), "2026-09-08", "m", ag=_ag([]), cfg=CFG, list_dir=tmp_path
    )
    assert len(out) == 0


def test_compute_factor_causal_synthetic(monkeypatch, tmp_path):
    # 000001: 11 行, close 恒涨 (R>0), amount 前 8 个差分正、第 9 差分负、第 10 正
    # → agree 9/10 = 0.9; 000002 仅 5 行 → 不够窗, 剔除
    dates = pd.date_range("2026-08-20", periods=11)
    p1 = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 11,
            "date": dates,
            "close_hfq": [float(x) for x in range(10, 21)],
            "amount": [100, 101, 102, 103, 104, 105, 106, 107, 108, 100, 101],
        }
    )
    p2 = pd.DataFrame(
        {
            "symbol": ["000002.SZ"] * 5,
            "date": dates[:5],
            "close_hfq": [10.0, 11, 12, 13, 14],
            "amount": [100, 101, 102, 103, 104],
        }
    )
    fp = tmp_path / "panel.parquet"
    pd.concat([p1, p2], ignore_index=True).to_parquet(fp, index=False)
    monkeypatch.setattr(gate, "PANEL_V3_PATH", fp)
    ser = gate.compute_amt_agree10(["000001", "000002"], dates[-1], window=10)
    assert ser is not None
    assert ser["000001"] == 0.9
    assert "000002" not in ser.index
