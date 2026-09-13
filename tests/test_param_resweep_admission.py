# -*- coding: utf-8 -*-
"""周末超参再扫纯判定函数 (scripts._param_resweep_weekly, 0913 #8) — 单测.

覆盖: 半窗净差计算 / 准入闸 (样本量+双半窗稳) / 着陆四路 (save/refresh/retire/none)
/ payload 构造 (半衰期独立顶层字段, 勿混 overrides — 混入会漏进 LGBM **params).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts._param_resweep_weekly import (
    admission_verdict,
    build_payload,
    half_window_delta,
    landing_action,
)


def _series(vals: dict) -> pd.Series:
    return pd.Series(vals)


D0 = "2026-09-01"
D1 = "2026-09-02"
D2 = "2026-09-03"
D3 = "2026-09-04"


def test_half_window_delta_math_and_alignment():
    chal = _series({D0: 1.0, D1: 2.0, D2: 3.0, D3: 4.0, "2026-09-05": 9.0})
    a0 = _series({D0: 0.0, D1: 0.0, D2: 0.0, D3: 0.0, "2026-08-31": 9.0})
    d = half_window_delta(chal, a0)
    assert d["n_days"] == 4  # 交集对齐, 各自独有日期剔除
    assert d["delta_full"] == 2.5
    assert d["delta_h1"] == 1.5
    assert d["delta_h2"] == 3.5


def test_half_window_delta_too_short_is_nan():
    d = half_window_delta(_series({D0: 1.0}), _series({D0: 0.0}))
    assert d["n_days"] == 1
    assert np.isnan(d["delta_full"])


def test_admission_all_gates_pass():
    delta = {"delta_full": 0.01, "delta_h1": 0.005, "delta_h2": 0.02, "n_days": 60}
    v = admission_verdict(60, delta)
    assert v["pass"] is True
    assert all(v["gates"].values())


def test_admission_fails_on_days_and_half_window():
    ok_delta = {"delta_full": 0.01, "delta_h1": 0.005, "delta_h2": 0.02, "n_days": 60}
    assert admission_verdict(30, ok_delta)["pass"] is False  # A: < 40 日
    bad_h1 = {"delta_full": 0.01, "delta_h1": -0.001, "delta_h2": 0.02, "n_days": 60}
    v = admission_verdict(60, bad_h1)
    assert v["pass"] is False and not v["gates"]["B_h1"]
    nan_delta = {"delta_full": float("nan"), "delta_h1": 0.0, "delta_h2": 0.0, "n_days": 60}
    assert admission_verdict(60, nan_delta)["pass"] is False


def test_landing_paths():
    net = {"A0_prod": 0.01, "C_hl60": 0.03, "J_json": 0.02}
    passed = {"C_hl60": {"pass": True}, "J_json": {"pass": True}}
    # 挑战者赢+过闸 → save
    assert landing_action(net, passed, json_active=True) == {"action": "save", "arm": "C_hl60"}
    # 挑战者赢+闸败 → none (现状不动)
    failed = {"C_hl60": {"pass": False}, "J_json": {"pass": True}}
    assert landing_action(net, failed, json_active=True)["action"] == "none"
    # J 赢+过闸 → refresh; J 赢+闸败 → retire
    net_j = {"A0_prod": 0.01, "C_hl60": 0.005, "J_json": 0.02}
    assert landing_action(net_j, passed, json_active=True)["action"] == "refresh"
    failed_j = {"C_hl60": {"pass": True}, "J_json": {"pass": False}}
    assert landing_action(net_j, failed_j, json_active=True)["action"] == "retire"
    # A0 赢: json 现役 → retire; 无 json → none
    net_a0 = {"A0_prod": 0.05, "C_hl60": 0.01, "J_json": 0.02}
    assert landing_action(net_a0, passed, json_active=True)["action"] == "retire"
    assert landing_action(net_a0, passed, json_active=False)["action"] == "none"


def test_build_payload_shapes():
    ev = {"sweep": "test"}
    hl = build_payload("main", "C_hl60", {"hl60": True}, "2026-09-12", ev)
    assert hl["cls_half_life_days"] == 60 and hl["overrides"] == {}
    auc = build_payload("main", "C_auc", {"es_auc": True}, "2026-09-12", ev)
    assert auc["overrides"] == {"3d_cls": {"metric": "auc"}, "5d_cls": {"metric": "auc"}, "10d_cls": {"metric": "auc"}}
    assert "cls_half_life_days" not in auc  # 非 LGBM 键绝不混 overrides/顶层误挂
    nl = build_payload("main", "C_nl31", {"nl": 31}, "2026-09-12", ev)
    assert nl["overrides"]["5d_cls"] == {"num_leaves": 31} and "cls_half_life_days" not in nl
