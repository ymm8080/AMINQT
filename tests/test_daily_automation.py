"""Tests for scripts/run_daily_automation 步骤编排 (2026-08-13).

plan_steps 是纯函数, 决定当日四模块自动化的执行序列:
  [cyq legacy_prob_head legacy deliver] [refresh?] [retrain?]
  [parallel? prob_head deliver_parallel] drift drift_parallel shadow_xmodule
关键不变式:
  1. 交付保底优先 (2026-08-27): legacy 预测链 (cyq→lph→legacy→deliver) 恒在一切
     重活 (refresh/retrain/parallel) 之前 — 重活卡死/超时被杀不影响当日清单落盘
     (08-27 事故: refresh 卡 9h 全链无清单). legacy_prob_head 恒在 legacy 预测前
     (概率闸依赖 bundle).
  2. deliver_parallel 依赖当日 fresh parallel 重生成 (run_dir), 故 parallel 被跳过
     (--skip-parallel) 时 deliver_parallel 必须同步丢弃, 否则会交付旧 run_dir 脏数据;
     prob_head 读 parallel 检查点面板, 同样只随 parallel 出现 (2026-08-15).
看门狗 (2026-08-27): 超时强杀走外部线程 taskkill 整树, 不再信 subprocess.run 内部
计时器 (08-27 它没触发).
"""

import datetime as _dt
import json
import subprocess
import sys

import scripts.run_daily_automation as ma
from scripts.run_daily_automation import (
    _STEP_TIMEOUT_S,
    _STEPS,
    RETRAIN_WEEKDAY,
    _exit_status,
    _is_interrupt_rc,
    _run_step_with_watchdog,
    _write_state,
    plan_steps,
)

# 2026-08-13 = Thursday (weekday 3, 非重训日), 2026-08-14 = Friday (weekday 4, 重训日).
THU, FRI = _dt.date(2026, 8, 13), _dt.date(2026, 8, 14)

_LEGACY_CHAIN = ["cyq", "legacy_prob_head", "legacy", "deliver"]
_TAIL = ["drift", "drift_parallel", "shadow_xmodule"]
_PARALLEL_CHAIN = ["parallel", "prob_head", "deliver_parallel"]


def test_fixture_dates_hit_expected_weekdays():
    assert RETRAIN_WEEKDAY == 4
    assert THU.weekday() != RETRAIN_WEEKDAY
    assert FRI.weekday() == RETRAIN_WEEKDAY


def test_every_step_has_timeout():
    """每步都有超时兜底 (08-17 事故: legacy 卡死 7h 无超时), 且不低于 15min."""
    for step in _STEPS:
        assert step in _STEP_TIMEOUT_S, f"{step} 缺超时配置"
        assert _STEP_TIMEOUT_S[step] >= 15 * 60
    # 重头步骤超时宽松 (正常耗时 4-6 倍), 只兜底卡死不误杀慢跑
    assert _STEP_TIMEOUT_S["retrain"] >= 8 * 3600
    assert _STEP_TIMEOUT_S["legacy"] >= 2 * 3600


def test_plan_steps_weekday_full_chain():
    assert plan_steps(THU) == _LEGACY_CHAIN + ["refresh"] + _PARALLEL_CHAIN + _TAIL


def test_plan_steps_retrain_day_inserts_retrain():
    assert plan_steps(FRI) == (
        _LEGACY_CHAIN + ["refresh", "retrain"] + _PARALLEL_CHAIN + _TAIL
    )


def test_plan_steps_force_retrain_inserts_retrain_on_non_retrain_day():
    """--force-retrain 在非周五也强制插入 retrain 步骤."""
    assert plan_steps(THU, force_retrain=True) == (
        _LEGACY_CHAIN + ["refresh", "retrain"] + _PARALLEL_CHAIN + _TAIL
    )


def test_plan_steps_delivery_chain_precedes_all_heavy_steps():
    """交付保底 (2026-08-27): legacy 交付链恒在 refresh/retrain/parallel 之前."""
    for steps in (
        plan_steps(THU),
        plan_steps(FRI),
        plan_steps(THU, force_retrain=True),
    ):
        assert steps[:4] == _LEGACY_CHAIN
        for heavy in ("refresh", "retrain", "parallel"):
            if heavy in steps:
                assert steps.index(heavy) > 3, f"{heavy} 不得排在 legacy 交付链前"


def test_plan_steps_skip_parallel_drops_deliver_parallel():
    """--skip-parallel 只跳并行链 (refresh 仍刷新检查点); 并行交付同步丢弃.

    shadow_xmodule 仍跑: 它读磁盘上已交付清单 (任意一侧缺也容忍), 非交付步骤.
    """
    assert plan_steps(THU, skip_parallel=True) == _LEGACY_CHAIN + ["refresh"] + _TAIL
    assert plan_steps(FRI, skip_parallel=True) == (
        _LEGACY_CHAIN + ["refresh", "retrain"] + _TAIL
    )


def test_plan_steps_skip_checkpoints_and_retrain():
    steps = plan_steps(FRI, skip_checkpoints=True, skip_retrain=True)
    assert steps == _LEGACY_CHAIN + _PARALLEL_CHAIN + _TAIL
    assert "refresh" not in steps and "retrain" not in steps


def test_plan_steps_all_skip_keeps_legacy_chain():
    assert (
        plan_steps(THU, skip_checkpoints=True, skip_retrain=True, skip_parallel=True)
        == _LEGACY_CHAIN + _TAIL
    )


# ── 中断中止 + 终态 state 文件 (08-21 事故: cyq 被 Ctrl+C 杀后仍启动 retrain,
#    且无终态标记 → 监督方无限等待一直耗 token) ──────────────────────────────


def test_interrupt_rc_is_status_control_c_exit():
    # 0xC000013A STATUS_CONTROL_C_EXIT — 08-21 事故 cyq 的真实返回码
    assert ma._INTERRUPT_RC == 3221225786


def test_is_interrupt_rc_identifies_control_c_exit():
    assert _is_interrupt_rc(3221225786) is True
    for rc in (0, 1, 2, 124):
        assert _is_interrupt_rc(rc) is False


def test_exit_status_maps_failures():
    assert _exit_status([]) == "ok"
    assert _exit_status(["legacy"]) == "failed"
    assert _exit_status(["cyq", "legacy"]) == "failed"


def test_write_state_writes_terminal_json(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "LOG_DIR", str(tmp_path))
    _write_state("20260821", "interrupted", step="cyq", rc=3221225786)
    state = json.loads(
        (tmp_path / "daily_automation_20260821.state.json").read_text("utf-8")
    )
    assert state["status"] == "interrupted"
    assert state["tag"] == "20260821"
    assert state["step"] == "cyq"
    assert state["rc"] == 3221225786


def test_write_state_missing_dir_creates_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "LOG_DIR", str(tmp_path / "sub" / "logs"))
    _write_state("20260821", "ok")
    state = json.loads(
        (tmp_path / "sub" / "logs" / "daily_automation_20260821.state.json").read_text(
            "utf-8"
        )
    )
    assert state["status"] == "ok"


# ── 外部看门狗超时强杀 (2026-08-27: subprocess.run 内部 timeout 未触发,
#    refresh 爬行 8h+; 改 Popen + 看门狗线程 taskkill 整树) ────────────────────


# ── 启动守卫 (2026-09-01): 手动重训/预测与链并发 → 页交换卡死/OOM, 不启动 ──────


def test_read_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "LOG_DIR", str(tmp_path))
    assert ma._read_state("20260901") is None
    ma._write_state("20260901", "ok", reason="dry_run")
    state = ma._read_state("20260901")
    assert state["status"] == "ok"
    assert state["reason"] == "dry_run"


def test_today_list_delivered_matches_tag_glob(tmp_path, monkeypatch):
    monkeypatch.setattr(ma, "STOCK_LIST_DIR", tmp_path)
    assert ma._today_list_delivered("20260901") is False
    (tmp_path / "legacy_stocklist_20260901__main.csv").write_text(
        "x\n", encoding="utf-8"
    )
    assert ma._today_list_delivered("20260901") is True
    # 其它日期不受影响
    assert ma._today_list_delivered("20260902") is False


def test_startup_guard_verdicts(monkeypatch):
    monkeypatch.setattr(ma, "_guard_log", lambda tag, msg: None)
    monkeypatch.setattr(ma, "_read_state", lambda tag: None)
    monkeypatch.setattr(ma, "_today_list_delivered", lambda tag: False)

    monkeypatch.setattr(
        ma,
        "_startup_guard_conflicts",
        lambda: [{"pid": 1, "sentinel": "s", "cmdline": "x"}],
    )
    # 活进程冲突 → wait (守候循环, 不直接放弃)
    assert ma._run_startup_guard("20260901") == "wait"

    monkeypatch.setattr(ma, "_startup_guard_conflicts", lambda: [])
    # 无冲突无交付 → 放行
    assert ma._run_startup_guard("20260901") == "go"
    # 今日链已 ok → skip
    monkeypatch.setattr(ma, "_read_state", lambda tag: {"status": "ok"})
    assert ma._run_startup_guard("20260901") == "skip"
    # 今日 ok 但来自 dry-run → 放行 (否则中午试跑会拦掉当晚真实链)
    monkeypatch.setattr(
        ma, "_read_state", lambda tag: {"status": "ok", "reason": "dry_run"}
    )
    assert ma._run_startup_guard("20260901") == "go"
    # 今日手动取消 → skip
    monkeypatch.setattr(
        ma,
        "_read_state",
        lambda tag: {"status": "cancelled", "reason": "manual_cancel"},
    )
    assert ma._run_startup_guard("20260901") == "skip"
    # 今日清单已交付 → skip
    monkeypatch.setattr(ma, "_read_state", lambda tag: None)
    monkeypatch.setattr(ma, "_today_list_delivered", lambda tag: True)
    assert ma._run_startup_guard("20260901") == "skip"


def test_wait_for_clearance_starts_when_conflicts_clear(monkeypatch):
    monkeypatch.setattr(ma, "_guard_log", lambda tag, msg: None)
    monkeypatch.setattr(ma, "_today_list_delivered", lambda tag: False)
    monkeypatch.setattr(ma, "_startup_guard_conflicts", lambda: [])
    monkeypatch.setattr(ma.time, "sleep", lambda s: None)
    assert ma._wait_for_clearance("20260901", tick_s=0, max_ticks=3) is True


def test_wait_for_clearance_aborts_when_deliverable_lands(monkeypatch):
    """守候期间今日清单已出 (手动链/手动预测交付) → 不必再启动."""
    monkeypatch.setattr(ma, "_guard_log", lambda tag, msg: None)
    monkeypatch.setattr(ma, "_today_list_delivered", lambda tag: True)
    monkeypatch.setattr(ma.time, "sleep", lambda s: None)
    assert ma._wait_for_clearance("20260901", tick_s=0, max_ticks=3) is False


def test_wait_for_clearance_gives_up_after_max_ticks(monkeypatch):
    monkeypatch.setattr(ma, "_guard_log", lambda tag, msg: None)
    monkeypatch.setattr(ma, "_today_list_delivered", lambda tag: False)
    monkeypatch.setattr(
        ma,
        "_startup_guard_conflicts",
        lambda: [{"pid": 9, "sentinel": "_retrain_legacy_full.py", "cmdline": "x"}],
    )
    monkeypatch.setattr(ma.time, "sleep", lambda s: None)
    assert ma._wait_for_clearance("20260901", tick_s=0, max_ticks=3) is False


def test_guard_skipped_state_is_terminal_for_babysitter():
    """main() 在 skip/wait 路径写 state="skipped" — babysitter 必须视为终态."""
    # _exit_status 只映射步骤失败; skipped 由守卫路径直接写, 这里锁住约定
    assert ma._exit_status([]) == "ok"
    import scripts._babysit_daily_automation as ba

    assert ba._exit_code_for("skipped") == 0


def test_watchdog_fast_child_not_killed():
    """快跑完的子进程不受看门狗影响, 返回真实 rc."""
    rc, timed_out = _run_step_with_watchdog(
        [sys.executable, "-c", "print('ok')"],
        subprocess.DEVNULL,
        env=None,
        timeout_s=60,
    )
    assert rc == 0
    assert timed_out is False


def test_watchdog_hung_child_tree_killed():
    """卡死子进程到点被整树强杀, timed_out=True (调用方映射 rc=124)."""
    rc, timed_out = _run_step_with_watchdog(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        subprocess.DEVNULL,
        env=None,
        timeout_s=2,
    )
    assert timed_out is True
    assert rc != 0
