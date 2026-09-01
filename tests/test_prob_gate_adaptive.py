"""LEGACY_PROB_GATE 自适应 margin 测试 (2026-08-31).

背景: 静态 margin=0.22 (08-22 定案) 在模型/市场分布漂移后不可达 → main 板
2026-08-22 起每日全灭 (08-30 剔 70/70, 08-31 剔 109/109). 自适应规则:
margin_t = clip(Q_q(近 N 日参与闸的逐股 spread = pred_prob - base_rate), min, max)
- 无历史 → 当日截面 bootstrap (等价 top-q% 语义, 无未来数据)
- 连续 3 决策日 100% 剔除 → 熔断放开至 margin_min + 大声告警
- 无历史且无当日截面 → 回退静态 margin (fail-open 保守)
- no-lookahead: 池化历史严格不含当日 (文件名日期 < today)
状态 WORM: {gate_margin_dir}/spreads_{board}_{date}.csv + margin_{board}_{date}.json
审计: emit 返回 result["gate_audit"] (E7 软区 pain / prob_gate 剔除逐股记录),
生产由 daily_pipeline 落盘 data/lists/gate_audit_{date}.parquet.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.pipeline1 import prob_head
from app.pipeline1.list_generator import ListGenerator
from config.settings import LEGACY_PROB_GATE

GATE_OFF = {"entry_prob": 0.0, "entry_ret_mult": 0.0}
TODAY = "20260901"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch, tmp_path):
    """隔离: gate 状态目录指 tmp; 默认开 rolling_q (本模块专测自适应)."""
    gdir = tmp_path / "gate_margin"
    gdir.mkdir()
    monkeypatch.setitem(LEGACY_PROB_GATE, "gate_margin_dir", str(gdir))
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin_mode", "rolling_q")
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin", 0.22)
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin_q", 0.90)
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin_min", 0.05)
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin_max", 0.25)
    monkeypatch.setitem(LEGACY_PROB_GATE, "spread_lookback_days", 20)
    return gdir


def _write_spreads(gdir, board, date, spreads):
    pd.DataFrame(
        {"symbol": [f"s{i}" for i in range(len(spreads))], "spread": spreads}
    ).to_csv(gdir / f"spreads_{board}_{date}.csv", index=False)


def _write_decision(gdir, board, date, n_total, n_kept):
    with open(gdir / f"margin_{board}_{date}.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"date": date, "board": board, "n_total": n_total, "n_kept": n_kept}, fh
        )


# ---------------- compute_adaptive_margin ----------------


def test_fixed_mode_returns_config_margin(monkeypatch):
    monkeypatch.setitem(LEGACY_PROB_GATE, "margin_mode", "fixed")
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "fixed" and m == 0.22


def test_no_history_no_today_falls_back_to_fixed(_hermetic):
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "fixed_fallback" and m == 0.22


def test_no_history_bootstrap_from_today(_hermetic):
    spreads = pd.Series([0.40, -0.10, 0.70, 0.20, 0.30])
    m, mode = prob_head.compute_adaptive_margin("main", spreads, TODAY)
    expect = float(np.clip(np.quantile(spreads.to_numpy(), 0.90), 0.05, 0.25))
    assert mode == "bootstrap" and m == pytest.approx(expect)


def test_history_pooled_excludes_today(_hermetic):
    gdir = _hermetic
    _write_spreads(gdir, "main", "20260830", [0.30, 0.31, 0.32])
    _write_spreads(gdir, "main", "20260831", [0.33, 0.34])
    # 当日文件已存在 (重跑场景): 不得混入池化历史 (no-lookahead)
    _write_spreads(gdir, "main", TODAY, [0.99])
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    pooled = np.array([0.30, 0.31, 0.32, 0.33, 0.34])
    expect = float(np.clip(np.quantile(pooled, 0.90), 0.05, 0.25))
    assert mode == "rolling_q" and m == pytest.approx(expect)


def test_lookback_window_limits_days(_hermetic):
    gdir = _hermetic
    for d in range(1, 26):  # 25 天历史, 窗口 20 → 最早 5 天不计
        _write_spreads(gdir, "main", f"202608{d + 5:02d}", [0.99])  # 高 spread 旧日
    for d in range(1, 21):
        _write_spreads(gdir, "main", f"202609{d:02d}", [0.10])
    m, _ = prob_head.compute_adaptive_margin("main", None, "20260921")
    assert m == pytest.approx(0.10)  # 若含旧 0.99 日则远高于此


def test_margin_floor_and_cap(_hermetic):
    gdir = _hermetic
    _write_spreads(gdir, "main", "20260830", [0.0, 0.01, -0.01])
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "rolling_q" and m == 0.05  # 地板
    _write_spreads(gdir, "main", "20260831", [0.60, 0.70, 0.80])
    m, _ = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert m == 0.25  # 顶


def test_breaker_after_three_zero_keep_days(_hermetic):
    gdir = _hermetic
    _write_spreads(gdir, "main", "20260829", [0.20])
    _write_spreads(gdir, "main", "20260830", [0.20])
    _write_spreads(gdir, "main", "20260831", [0.20])
    for d in ("20260829", "20260830", "20260831"):
        _write_decision(gdir, "main", d, n_total=100, n_kept=0)
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "breaker" and m == 0.05


def test_breaker_reset_by_nonzero_keep_day(_hermetic):
    gdir = _hermetic
    _write_spreads(gdir, "main", "20260829", [0.20])
    _write_spreads(gdir, "main", "20260830", [0.20])
    _write_spreads(gdir, "main", "20260831", [0.20])
    _write_decision(gdir, "main", "20260829", n_total=100, n_kept=0)
    _write_decision(gdir, "main", "20260830", n_total=100, n_kept=3)  # 中间日有保留
    _write_decision(gdir, "main", "20260831", n_total=100, n_kept=0)
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "rolling_q"


def test_corrupt_history_file_ignored(_hermetic):
    gdir = _hermetic
    (gdir / "spreads_main_20260830.csv").write_text(
        "garbage,not a csv\n\x00\x01", encoding="utf-8"
    )
    _write_spreads(gdir, "main", "20260831", [0.30, 0.31])
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "rolling_q" and m == pytest.approx(
        float(np.clip(np.quantile([0.30, 0.31], 0.90), 0.05, 0.25))
    )


def test_history_board_isolated(_hermetic):
    gdir = _hermetic
    _write_spreads(gdir, "dual", "20260830", [0.99])  # 别的板
    _write_spreads(gdir, "main", "20260830", [0.15])
    m, mode = prob_head.compute_adaptive_margin("main", None, TODAY)
    assert mode == "rolling_q" and m == pytest.approx(0.15)


# ---------------- apply_prob_gate 集成: 落盘 + 逐股 fail-open ----------------


def _mk_cands(n_main=4, date="2026-09-01"):
    return pd.DataFrame(
        {
            "symbol": [f"60000{i}" for i in range(n_main)],
            "board": ["main"] * n_main,
            "date": [pd.Timestamp(date)] * n_main,
            "pred_ret_3d": [0.01] * n_main,
            "pred_ret_5d": [0.01] * n_main,
            "pred_ret_10d": [0.03, 0.02, 0.01, 0.005][:n_main],
            "prob_up": [0.6] * n_main,
            "prob_up_3d": [0.6] * n_main,
            "prob_up_5d": [0.6] * n_main,
            "prob_up_10d": [0.6] * n_main,
        }
    )


def test_apply_gate_writes_spreads_and_decision(_hermetic, monkeypatch):
    gdir = _hermetic

    def fake_gate(board, feat_day, tail, panel_dates):
        return (
            pd.Series({"600000": 0.50, "600001": 0.60, "600002": 0.70, "600003": 0.90}),
            0.20,
        )

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    res = prob_head.apply_prob_gate(
        _mk_cands(),
        {"main": pd.DataFrame({"symbol": [f"60000{i}" for i in range(4)]})},
        pd.DataFrame(),
        np.array([]),
    )
    # spread = prob - 0.20 = {0.30,0.40,0.50,0.70} → bootstrap Q90=0.65 → clip 0.25
    # → thr=0.45 → 4 只全过
    assert sorted(res["symbol"]) == ["600000", "600001", "600002", "600003"]
    sf = gdir / f"spreads_main_{TODAY}.csv"
    dj = gdir / f"margin_main_{TODAY}.json"
    assert sf.exists() and dj.exists()
    s = pd.read_csv(sf, dtype={"symbol": str})
    assert set(s.columns) == {"symbol", "pred_prob", "spread"}
    assert len(s) == 4
    d = json.loads(dj.read_text(encoding="utf-8"))
    assert d["n_total"] == 4 and d["n_kept"] == 4 and d["mode"] == "bootstrap"
    assert d["base_rate"] == pytest.approx(0.20)


def test_apply_gate_drops_below_adaptive_thr(_hermetic, monkeypatch):
    def fake_gate(board, feat_day, tail, panel_dates):
        return (
            pd.Series({"600000": 0.21, "600001": 0.60, "600002": 0.70, "600003": 0.90}),
            0.20,
        )

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    res = prob_head.apply_prob_gate(
        _mk_cands(),
        {"main": pd.DataFrame({"symbol": [f"60000{i}" for i in range(4)]})},
        pd.DataFrame(),
        np.array([]),
    )
    # bootstrap Q90({0.01,0.40,0.50,0.70})=0.625 → clip 0.25 → thr 0.45 → 0.21 剔
    assert res["symbol"].tolist() == ["600001", "600002", "600003"]


def test_apply_gate_nan_predprob_failopen(_hermetic, monkeypatch):
    def fake_gate(board, feat_day, tail, panel_dates):
        # 600000 缺 → NaN
        return (pd.Series({"600001": 0.60, "600002": 0.70, "600003": 0.90}), 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    res = prob_head.apply_prob_gate(
        _mk_cands(),
        {"main": pd.DataFrame({"symbol": [f"60000{i}" for i in range(4)]})},
        pd.DataFrame(),
        np.array([]),
    )
    assert "600000" in res["symbol"].tolist()  # NaN → fail-open 保留


def test_apply_gate_breaker_relaxes(_hermetic, monkeypatch):
    gdir = _hermetic
    for d in ("20260829", "20260830", "20260831"):
        _write_decision(gdir, "main", d, n_total=100, n_kept=0)
        _write_spreads(gdir, "main", d, [0.40, 0.50, 0.70])

    def fake_gate(board, feat_day, tail, panel_dates):
        return (
            pd.Series({"600000": 0.50, "600001": 0.60, "600002": 0.70, "600003": 0.90}),
            0.20,
        )

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    res = prob_head.apply_prob_gate(
        _mk_cands(),
        {"main": pd.DataFrame({"symbol": [f"60000{i}" for i in range(4)]})},
        pd.DataFrame(),
        np.array([]),
    )
    # 熔断 margin=0.05 → thr=0.25 → 全留
    assert len(res) == 4
    d = json.loads((gdir / f"margin_main_{TODAY}.json").read_text(encoding="utf-8"))
    assert d["mode"] == "breaker"


# ---------------- E7 审计列 + emit result["gate_audit"] ----------------


def test_entry_filter_attaches_audit_columns():
    lg = ListGenerator(**GATE_OFF)
    df = pd.DataFrame(
        {
            "symbol": ["300001", "300002", "300003"],
            "board": ["GEM"] * 3,
            "prob_up": [0.9, 0.9, 0.9],
            "prob_up_10d": [0.9, 0.9, 0.9],
            "pred_ret_10d": [0.05, 0.05, 0.05],
            "pred_q50_3d": [0.01] * 3,
            "pred_q50_5d": [0.01] * 3,
            "pain_prob": [0.2, 0.45, 0.8],  # 过 / 软区 / 硬区
        }
    )
    scored = lg.compute_scores(df)
    passed = lg.entry_filter(scored)
    assert "_e7_pain_ok" in scored.columns
    assert scored["_e7_pain_ok"].tolist() == [True, False, False]
    assert "300001" in passed["symbol"].tolist()
    assert "300002" not in passed["symbol"].tolist()  # 软区仍硬剔 (只标注)


def test_emit_returns_gate_audit_with_pain_soft_and_gate_drops(_hermetic, monkeypatch):
    from app.pipeline1 import risk_overlays

    monkeypatch.setattr(risk_overlays, "block_trade_recent_scan", lambda *a, **k: [])
    monkeypatch.setattr(risk_overlays, "share_float_upcoming_scan", lambda *a, **k: [])

    cands = pd.DataFrame(
        {
            "symbol": ["600001", "600002", "300001", "300002"],
            "board": ["main", "main", "GEM", "GEM"],
            "date": [pd.Timestamp("2026-09-01")] * 4,
            "pred_ret_3d": [0.01] * 4,
            "pred_ret_5d": [0.01] * 4,
            "pred_ret_10d": [0.03, 0.02, 0.05, 0.04],
            "prob_up": [0.6, 0.6, 0.9, 0.9],
            "prob_up_3d": [0.6, 0.6, 0.9, 0.9],
            "prob_up_5d": [0.6, 0.6, 0.9, 0.9],
            "prob_up_10d": [0.6, 0.6, 0.9, 0.9],
            "pred_q50_3d": [0.01, 0.01, 0.01, 0.01],
            "pred_q50_5d": [0.01, 0.01, 0.01, 0.01],
            "pain_prob": [0.1, 0.1, 0.35, 0.45],  # 300002 = 软区 (0.4-0.5)
        }
    )

    def fake_gate(board, feat_day, tail, panel_dates):
        return (pd.Series({"600001": 0.90, "600002": 0.30}), 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    gate_inputs = {
        "feats": {"main": pd.DataFrame({"symbol": ["600001", "600002"]})},
        "tail": pd.DataFrame(),
        "panel_dates": np.array([]),
    }
    out = ListGenerator(**GATE_OFF).emit(cands, prob_gate=gate_inputs)
    assert not out["empty"]
    # 排名纯 pred_ret_10d 降序: 300001 (0.05) > 600001 (0.03);
    # 600002 被 prob_gate 剔 (spread 0.10 < thr), 300002 被 pain 闸剔 (软区只标注)
    assert out["list"]["symbol"].tolist() == ["300001", "600001"]
    audit = out["gate_audit"]
    gate_row = audit[(audit["symbol"] == "600002") & (audit["stage"] == "prob_gate")]
    assert len(gate_row) == 1 and gate_row["reason"].iloc[0] == "prob_gate"
    assert gate_row["spread"].iloc[0] == pytest.approx(0.10)
    soft_row = audit[(audit["symbol"] == "300002") & (audit["stage"] == "E7")]
    assert len(soft_row) == 1 and soft_row["reason"].iloc[0] == "pain_soft"
