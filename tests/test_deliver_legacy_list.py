"""Tests for scripts/_deliver_legacy_list 交付 (2026-08-13).

原则 (用户定案): 板块被整体退回时仍要出该板清单, 醒目标注「未接受 + 原因」,
不许静默跳过. _board_reject_reasons 重放 E7 可计算闸, 报告被整体退回板块的原因.
"""

import pandas as pd

from scripts._deliver_legacy_list import _board_reject_reasons


def _cand():
    # main 400 只全部 q50_3d<=0 (闸3 全灭); GEM/STAR 各 160 只, 部分通过.
    rows = []
    for i in range(400):
        rows.append(
            {
                "symbol": f"60{i:04d}",
                "board": "main",
                "pred_ret_10d": 0.02 + 0.001 * i,
                "pred_q50_3d": -0.003 + 0.00001 * i,
                "pred_q50_5d": -0.001,
            }
        )
    for b, sym in (("GEM", "30"), ("STAR", "68")):
        for i in range(160):
            q3 = 0.005 if i < 40 else -0.005  # 前 40 只闸3 通过
            rows.append(
                {
                    "symbol": f"{sym}{i:04d}",
                    "board": b,
                    "pred_ret_10d": 0.02,
                    "pred_q50_3d": q3,
                    "pred_q50_5d": q3,
                }
            )
    return pd.DataFrame(rows)


def test_main_gate3_reject_reason():
    cand = _cand()
    final = cand[cand["board"].isin(["GEM", "STAR"])]
    final = final[final["pred_q50_3d"] > 0].copy()
    final["symbol"] = final["symbol"].astype(str)
    reasons = _board_reject_reasons(cand, final)
    # main 有候选但清单 0 只, 闸3 全灭 → 明确原因
    assert "main" in reasons
    assert "E7 闸3" in reasons["main"] and "中位数" in reasons["main"]
    # GEM/STAR 有候选且在最终清单 → 不标注
    assert "GEM" not in reasons and "STAR" not in reasons


def test_generic_reason_when_partial_gate_pass():
    cand = _cand()
    final = cand[cand["board"] == "main"].head(10).copy()
    final = final[final["pred_q50_3d"] > 0].copy()  # main q50 全负 → 空表
    reasons = _board_reject_reasons(cand, final)
    # main 闸3 全灭 → 明确原因
    assert "E7 闸3" in reasons["main"]
    # GEM/STAR 有候选但最终清单 0 只, 闸3 有部分通过 → 落到排名/过滤原因
    assert "GEM" in reasons and "STAR" in reasons
    assert "E7 闸3" not in reasons["GEM"]


def test_reject_reasons_no_candidates():
    assert _board_reject_reasons(None, pd.DataFrame()) == {}


def test_gate_info_sections_split_and_sorted():
    from scripts._deliver_legacy_list import _gate_info_sections

    audit = pd.DataFrame(
        {
            "symbol": ["300001", "300002", "600001", "600002", "600003"],
            "board": ["GEM", "GEM", "main", "main", "main"],
            "stage": ["E7", "E7", "prob_gate", "prob_gate", "prob_gate"],
            "reason": ["pain_soft", "pain_hard", "prob_gate", "prob_gate", "prob_gate"],
            "pred_ret_10d": [0.09, 0.01, 0.05, 0.07, 0.02],
            "pain_prob": [0.45, 0.8, None, None, None],
            "pred_prob": [None, None, 0.60, 0.55, 0.50],
            "spread": [None, None, 0.12, 0.08, 0.04],
        }
    )
    sections = _gate_info_sections(audit)
    assert len(sections) == 2
    t1, s1 = sections[0]
    assert "pain" in t1 and s1["symbol"].tolist() == ["300001"]  # 软区 only, 硬区不留
    t2, s2 = sections[1]
    assert "prob_gate" in t2
    assert s2["symbol"].tolist() == ["600001", "600002", "600003"]  # spread 降序
    # cap 10
    big = pd.DataFrame(
        {
            "symbol": [f"60{i:04d}" for i in range(15)],
            "board": ["main"] * 15,
            "stage": ["prob_gate"] * 15,
            "reason": ["prob_gate"] * 15,
            "spread": [0.01 * i for i in range(15)],
        }
    )
    _, s3 = _gate_info_sections(big)[0]
    assert len(s3) == 10 and s3["spread"].max() == 0.14
