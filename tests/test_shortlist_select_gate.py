"""Tests for scripts/_shortlist_t5_t10.select_confident — 纯 T+3 入选门 (2026-08-09).

背景: 2026-08-09 删 2d 视界后, 原 T+2/T+3 联合门 (2026-08-07) 退化为纯 T+3 门:
  保留 ⇔ pred_ret_3d > t3_min (config SHORTLIST_SCORE.select_gate.t3_min).
  2026-08-10: 门基准从 pred_mag_3d (MFE 最大浮盈, 虚高) 改为 pred_ret_3d
  (close-to-close 可兑现净预期, 成本已扣) — 与 legacy 收益闸口径一致.
  pred_ret_3d 为 0 或负 → 剔除 (严格大于).
  2026-08-14: t3_min 分板块 dict (main=0 / dual=0.5%, _diag_t3min_sweep 定案),
  也兼容全局 float.
  2026-08-21: V3 扩建后重扫 (_diag_t3min_sweep_250d_20260821) — dual 0.5% 已输基线,
  0.25% 未过 ≥3/4 子窗纪律 → dual 回 0 (双板均 0 门槛).
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
    """rows: list of dict(symbol, ret3, prob3). prob3 默认 0.5. ret3 = c2c 净预期."""
    data = [
        {
            "symbol": str(r[0]),
            "board": "dual",
            "pred_ret_3d": r[1],
            "pred_prob_3d": r[2] if len(r) > 2 else 0.5,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


def test_select_gate_present_in_config():
    sg = S.SHORTLIST_SCORE["select_gate"]
    # 2026-08-14: dual=0.5%; 08-21 扩建后重扫 dual 回 0
    assert sg["t3_min"] == {"main": 0.0, "dual": 0.0}
    # 2d 联合门已删, t2_min/t3_floor 不应再存在
    assert "t2_min" not in sg
    assert "t3_floor" not in sg


def test_per_board_threshold():
    # 08-21 重扫: dual t3_min 改回 0 (扩建后 0.5% 输基线) → 双板均 0 门槛:
    # 正预期保留, 0/负剔除
    df = pd.DataFrame(
        [
            {
                "symbol": "D001",
                "board": "dual",
                "pred_ret_3d": 0.003,
                "pred_prob_3d": 0.5,
            },
            {
                "symbol": "M001",
                "board": "main",
                "pred_ret_3d": 0.003,
                "pred_prob_3d": 0.5,
            },
            {
                "symbol": "D002",
                "board": "dual",
                "pred_ret_3d": -0.003,
                "pred_prob_3d": 0.5,
            },
        ]
    )
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"D001", "M001"}


def test_global_float_threshold_backward_compat(monkeypatch):
    # t3_min 为全局 float 时仍工作 (向后兼容路径)
    df = pd.DataFrame(
        [
            {"symbol": "A", "board": "dual", "pred_ret_3d": 0.02, "pred_prob_3d": 0.5},
            {"symbol": "B", "board": "dual", "pred_ret_3d": 0.005, "pred_prob_3d": 0.5},
        ]
    )
    monkeypatch.setattr(
        S, "SHORTLIST_SCORE", {**S.SHORTLIST_SCORE, "select_gate": {"t3_min": 0.01}}
    )
    out = S.select_confident(df)
    # 全局门槛 1%: A(2%) 保留, B(0.5%) 剔除
    assert set(out["symbol"]) == {"A"}


def test_t3_positive_kept():
    df = _cand([("A", 0.02, 0.60)])
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"A"}


def test_t3_zero_dropped():
    # 纯 T+3 门严格大于: 恰好 0 不入
    df = _cand([("A", 0.0, 0.60)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_t3_marginal_negative_dropped():
    # 联合门退化为纯 T+3 后, 边际转负 (T+3=-0.5%) 无 T+2 可救, 直接剔除
    df = _cand([("A", -0.005, 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_t3_deep_negative_dropped():
    df = _cand([("A", -0.02, 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_nan_t3_dropped():
    df = _cand([("A", float("nan"), 0.55)])
    out = S.select_confident(df)
    assert len(out) == 0


def test_prob_min_still_filters():
    df = _cand([("A", 0.02, 0.30), ("B", 0.02, 0.80)])
    out = S.select_confident(df, prob_min=0.5)
    assert set(out["symbol"]) == {"B"}


def test_pure_gate_mixed():
    df = _cand(
        [
            ("keep1", 0.02, 0.60),  # T+3>0 → 保留
            ("drop1", -0.005, 0.55),  # T+3 边际负 → 剔除
            ("drop2", -0.02, 0.55),  # T+3 深转负 → 剔除
            ("drop3", 0.0, 0.60),  # T+3 恰好 0 → 剔除
        ]
    )
    out = S.select_confident(df)
    assert set(out["symbol"]) == {"keep1"}
