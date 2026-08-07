"""Tests for scripts/_shortlist_t5_t10.select_confident — T+2/T+3 联合入选门 (2026-08-07).

背景: 用户 "考虑 T+2,T+3 一起" (301326 08-05: raw score dual 第 1, T+3 边际转负被整只剔除,
实际 2 天 +12%). 联合门 (config SHORTLIST_SCORE.select_gate):
  保留 ⇔ (T+3 > t3_min=0) 或 (T+2 > t2_min=0.01 且 T+3 > t3_floor=-0.01).
"""

import importlib.util
import os
from pathlib import Path

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "shortlist_mod", Path(ROOT) / "scripts" / "_shortlist_t5_t10.py"
)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _cand(rows):
    """rows: list of dict(symbol, mag2, mag3, prob3). prob2 默认填上以防后续用到."""
    data = [
        {
            "symbol": str(r[0]),
            "board": "dual",
            "pred_mag_2d": r[1],
            "pred_mag_3d": r[2],
            "pred_prob_2d": 0.5,
            "pred_prob_3d": r[3] if len(r) > 3 else 0.5,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def test_select_gate_present_in_config():
    sg = S.SHORTLIST_SCORE["select_gate"]
    assert sg["t3_min"] == 0.0
    assert sg["t2_min"] > 0.0
    assert sg["t3_floor"] < 0.0


def test_t3_positive_kept():
    df = _cand([("A", 0.00, 0.02, 0.60)])
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"A"}


def test_t3_marginal_negative_t2_strong_rescued():
    # T+3 = -0.5% (边际转负, 在 t3_floor 之上), T+2 = +3% (强看涨) → 联合门保留
    df = _cand([("A", 0.03, -0.005, 0.55)])
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"A"}


def test_t3_deep_negative_not_rescued_by_t2():
    # T+3 = -2% (≤ t3_floor=-1%), 即使 T+2 强看涨也剔除
    df = _cand([("A", 0.03, -0.02, 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_t3_negative_t2_weak_dropped():
    # T+3 = -0.5% 边际转负, 但 T+2 = +0.5% (≤ t2_min=1%) → 无 T+2 理由, 剔除
    df = _cand([("A", 0.005, -0.005, 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_nan_t3_dropped():
    df = _cand([("A", 0.02, float("nan"), 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_prob_min_still_filters():
    df = _cand([("A", 0.02, 0.02, 0.30), ("B", 0.02, 0.02, 0.80)])
    out = S.select_confident(df, prob_min=0.5)
    assert set(out["symbol"]) == {"B"}


def test_joint_gate_mixed():
    df = _cand(
        [
            ("keep1", 0.00, 0.02, 0.60),  # T+3>0 → 保留
            ("keep2", 0.04, -0.005, 0.55),  # T+3 边际负 + T+2 强 → 保留
            ("drop1", 0.04, -0.02, 0.55),  # T+3 深转负 → 剔除
            ("drop2", 0.005, -0.005, 0.55),  # T+3 负 + T+2 弱 → 剔除
        ]
    )
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"keep1", "keep2"}
