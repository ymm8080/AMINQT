"""Tests for scripts/_run_guard — 重训/预测并发守卫 (2026-09-01).

背景: 手动重训/预测与 20:15 自动化链并发 → 页交换卡死 (08-17) / OOM 整链被杀
(08-24). 进程扫描用 procs 参数注入伪进程表, 不依赖真实进程表.
"""

import os

import scripts._run_guard as rg
from scripts._run_guard import (
    CHAIN_SENTINELS,
    HEAVY_SENTINELS,
    find_conflicts,
    skip_reason,
)


def _proc(pid, name, *cmdline):
    return {"pid": pid, "name": name, "cmdline": list(cmdline)}


# ── find_conflicts: 进程扫描 ────────────────────────────────────────────────


def test_find_conflicts_matches_sentinel_cmdline():
    procs = [
        _proc(
            100,
            "python.exe",
            "python",
            "-u",
            "scripts/_retrain_legacy_full.py",
            "20260901",
        )
    ]
    hits = find_conflicts(procs=procs)
    assert len(hits) == 1
    assert hits[0]["pid"] == 100
    assert hits[0]["sentinel"] == "_retrain_legacy_full.py"


def test_find_conflicts_ignores_non_python_cmdline_matches():
    """编辑器/终端 cmdline 恰含脚本路径 (打开的文件标签) 不得误判为冲突."""
    procs = [
        _proc(200, "Code.exe", "Code", "--file", "scripts/_retrain_legacy_full.py")
    ]
    assert find_conflicts(procs=procs) == []


def test_find_conflicts_matches_module_runner_invocation():
    procs = [
        _proc(101, "python.exe", "python", "-u", "-m", "app.pipeline_parallel.runner")
    ]
    hits = find_conflicts(procs=procs)
    assert len(hits) == 1
    assert hits[0]["sentinel"] == "app.pipeline_parallel.runner"


def test_find_conflicts_excludes_own_pid():
    """哨兵子串总出现在自己的 argv 里 — 自身 PID 必须排除, 否则永远自拦."""
    procs = [
        _proc(os.getpid(), "python.exe", "python", "-c", "x  # _gen_legacy_list.py")
    ]
    assert find_conflicts(procs=procs) == []


def test_find_conflicts_excludes_given_pids():
    procs = [_proc(300, "python.exe", "python", "-m", "app.pipeline_parallel.runner")]
    assert find_conflicts(exclude_pids={300}, procs=procs) == []


def test_find_conflicts_tolerates_missing_fields():
    """psutil 对无权限进程返回 None 字段 — 不得抛异常."""
    assert find_conflicts(procs=[{"pid": None, "name": None, "cmdline": None}]) == []


def test_chain_sentinels_include_orchestrator_script_level_not():
    """链级要识别另一条链实例; 脚本级不含 orchestrator (防被自己的父链误杀)."""
    assert rg.ORCHESTRATOR_SENTINEL in CHAIN_SENTINELS
    assert rg.ORCHESTRATOR_SENTINEL not in HEAVY_SENTINELS
    assert CHAIN_SENTINELS[: len(HEAVY_SENTINELS)] == HEAVY_SENTINELS


# ── skip_reason: 三闸判定 ───────────────────────────────────────────────────


def test_skip_reason_live_process_has_top_priority():
    r = skip_reason(
        [{"pid": 1, "sentinel": "x"}],
        today_state={"status": "ok"},
        deliverable_exists=True,
    )
    assert r is not None and r[0] == "live_process"


def test_skip_reason_live_process_lists_count():
    conflicts = [{"pid": 1, "sentinel": "a"}, {"pid": 2, "sentinel": "b"}]
    code, detail = skip_reason(conflicts, today_state=None, deliverable_exists=False)
    assert code == "live_process"
    assert "PID 1" in detail and "2 个" in detail


def test_skip_reason_state_ok_today():
    r = skip_reason([], today_state={"status": "ok"}, deliverable_exists=False)
    assert r is not None and r[0] == "state_ok_today"


def test_skip_reason_dry_run_ok_does_not_block():
    """dry-run 会写 ok 终态 — 若算"已完成", 中午试跑会把当晚真实链静默跳过."""
    r = skip_reason(
        [], today_state={"status": "ok", "reason": "dry_run"}, deliverable_exists=False
    )
    assert r is None


def test_skip_reason_cancelled_today():
    """state=cancelled = 用户手动取消当日 — 守卫直接跳过."""
    r = skip_reason(
        [],
        today_state={"status": "cancelled", "reason": "manual_cancel"},
        deliverable_exists=False,
    )
    assert r is not None and r[0] == "cancelled_today"
    assert "manual_cancel" in r[1]


def test_skip_reason_already_delivered():
    r = skip_reason([], today_state=None, deliverable_exists=True)
    assert r is not None and r[0] == "already_delivered"


def test_skip_reason_passes_when_clear():
    assert skip_reason([], today_state=None, deliverable_exists=False) is None


def test_skip_reason_non_ok_states_do_not_block():
    """failed/interrupted 允许重试; running 且无活进程 = 上条链死了, 应继续跑."""
    for st in ("failed", "interrupted", "running"):
        assert (
            skip_reason([], today_state={"status": st}, deliverable_exists=False)
            is None
        )
