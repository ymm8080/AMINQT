# -*- coding: utf-8 -*-
"""头选择台账 (scripts/_head_choice_ledger) — WORM + 幂等单测.

[0912 夜用户令] 双管线四格 (legacy/parallel × main/dual) 每次重训的 cls/reg
双头 OOS + 自选头入统一 CSV; (pipeline, board, tag) 为幂等键。
"""

from __future__ import annotations

import csv

from scripts._head_choice_ledger import LEDGER_COLUMNS, append_head_choice_row


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_first_write_creates_header_and_row(tmp_path):
    p = tmp_path / "ledger.csv"
    wrote = append_head_choice_row(
        "legacy",
        "main",
        "20260913",
        weighted_ic_reg=0.0214,
        weighted_ic_cls=0.0659,
        chosen="cls",
        gate_pass=True,
        switched=True,
        n_features=358,
        ledger_path=p,
    )
    assert wrote is True
    rows = _read_rows(p)
    assert len(rows) == 1
    r = rows[0]
    assert tuple(r.keys()) == LEDGER_COLUMNS
    assert r["pipeline"] == "legacy"
    assert r["board"] == "main"
    assert r["tag"] == "20260913"
    assert r["chosen"] == "cls"
    assert r["gate_pass"] == "True"
    assert r["switched"] == "True"
    assert r["n_features"] == "358"


def test_same_key_skipped_worm(tmp_path):
    p = tmp_path / "ledger.csv"
    kwargs = dict(
        weighted_ic_reg=-0.14,
        weighted_ic_cls=0.04,
        chosen="cls",
        gate_pass=False,
        switched=False,
    )
    assert append_head_choice_row("legacy", "dual", "T", ledger_path=p, **kwargs) is True
    # 同键重跑 (数值漂移也不覆盖 — WORM)
    assert (
        append_head_choice_row(
            "legacy", "dual", "T", ledger_path=p, weighted_ic_reg=9.0, **{
                k: v for k, v in kwargs.items() if k != "weighted_ic_reg"
            }
        )
        is False
    )
    rows = _read_rows(p)
    assert len(rows) == 1
    assert rows[0]["weighted_ic_reg"] == "-0.14"


def test_different_key_appends_and_pipelines_coexist(tmp_path):
    p = tmp_path / "ledger.csv"
    append_head_choice_row(
        "legacy", "dual", "T", weighted_ic_reg=1.0, weighted_ic_cls=2.0,
        chosen="cls", gate_pass=True, switched=False, ledger_path=p,
    )
    # 同板同 tag 不同管线 = 不同行; 同管线不同板 = 不同行
    assert append_head_choice_row(
        "parallel", "dual", "T", weighted_ic_reg=None, weighted_ic_cls=0.5,
        chosen="prob", gate_pass=True, switched=False, ledger_path=p,
    )
    assert append_head_choice_row(
        "legacy", "main", "T", weighted_ic_reg=0.02, weighted_ic_cls=0.07,
        chosen="cls", gate_pass=True, switched=True, ledger_path=p,
    )
    rows = _read_rows(p)
    assert [r["pipeline"] for r in rows] == ["legacy", "parallel", "legacy"]
    assert rows[0]["weighted_ic_reg"] == "1.0"  # None 之外原样字符串化
