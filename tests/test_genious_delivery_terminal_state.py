# -*- coding: utf-8 -*-
"""GENIOUS 交付链终态守卫 (2026-10-28, 10-27 事故回归测试).

10-27 事故: MemoryError 在交付链未防护中段连环炸, 进程静默死 — state 卡 running,
无文件, 无声响, 次日晨才被发现。守的契约: 交付链任何异常都必须落 **显式终态**
(state=failed + rc=3), 不许 running 假死; 推送段留在终态之外, 文件已交付则 state=ok。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts._genious_excel as ge  # noqa: E402


def _prime(monkeypatch, states):
    """把 main() 隔离成纯交付链演练: 不碰真面板/真包/真日志/真 state。"""
    monkeypatch.setattr(ge, "GENIOUS", {"enable": True, "push_to_ths": False})
    monkeypatch.setattr(ge, "_setup_logging", lambda tag: None)
    monkeypatch.setattr(
        ge, "_write_state", lambda tag, status, **k: states.append((tag, status, k))
    )
    monkeypatch.setattr(
        ge,
        "_load_fresh_panel",
        lambda target, no_fetch, wait_min: pd.DataFrame({"symbol": ["000001"]}),
    )
    monkeypatch.setattr(sys, "argv", ["genious", "20260916", "--dry-run"])


def test_delivery_failure_writes_terminal_state_and_rc3(monkeypatch):
    """交付链中段炸 (MemoryError 同型) → state=failed + rc=3, 不许 running 假死。"""
    states = []
    _prime(monkeypatch, states)

    def boom(df):  # noqa: ARG001 — 签名对齐 kt.compute_features
        raise MemoryError("Unable to allocate 9.84 MiB (10-27 事故同型)")

    monkeypatch.setattr(ge.kt, "compute_features", boom)
    rc = ge.main()
    assert rc == 3
    assert states[0][1] == "running", "入口先落 running"
    assert states[-1][1] == "failed", "交付链异常必须落终态 failed"
    assert states[-1][2]["reason"] == "delivery"
