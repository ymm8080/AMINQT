"""Tests for scripts/_babysit_daily_automation — fail-fast 监督 (2026-08-21).

以 logs/daily_automation_<tag>.state.json 为唯一终态判据: 状态文件出现
(ok/failed/interrupted) 即退出. 替代 _watch_daily_automation.py 的无限 tail
循环 — 那种循环在自动化崩溃后永不退出, 挂着的监督进程一直耗 token.
"""

import json

import scripts._babysit_daily_automation as ba


def test_exit_code_for_maps_terminal_statuses():
    assert ba._exit_code_for("ok") == 0
    assert ba._exit_code_for("failed") == 1
    assert ba._exit_code_for("interrupted") == 130
    # 守卫跳过 = 有意不跑 (并发冲突/今日已完成), 非崩溃 (2026-09-01)
    assert ba._exit_code_for("skipped") == 0
    # 手动取消当日 = 有意不跑, 非崩溃
    assert ba._exit_code_for("cancelled") == 0
    # 未知/未终态 → crash 码, 绝不当成功
    assert ba._exit_code_for("running") == 3
    assert ba._exit_code_for("nonsense") == 3


def test_read_state_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "LOG_DIR", str(tmp_path))
    assert ba._read_state("20260821") is None


def test_read_state_parses_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "LOG_DIR", str(tmp_path))
    (tmp_path / "daily_automation_20260821.state.json").write_text(
        json.dumps({"status": "interrupted", "tag": "20260821"}),
        encoding="utf-8",
    )
    state = ba._read_state("20260821")
    assert state is not None
    assert state["status"] == "interrupted"


def test_read_state_corrupt_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ba, "LOG_DIR", str(tmp_path))
    (tmp_path / "daily_automation_20260821.state.json").write_text(
        "{ not json", encoding="utf-8"
    )
    assert ba._read_state("20260821") is None
