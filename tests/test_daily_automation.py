"""Tests for scripts/run_daily_automation 步骤编排 (2026-08-13).

plan_steps 是纯函数, 决定当日四模块自动化的执行序列:
  [refresh?] [retrain?] legacy_prob_head legacy deliver
  [parallel? prob_head deliver_parallel] drift
关键不变式: deliver_parallel 依赖当日 fresh parallel 重生成 (run_dir), 故
  parallel 被跳过 (--skip-parallel) 时 deliver_parallel 必须同步丢弃,
  否则会交付旧 run_dir 脏数据; prob_head 读 parallel 检查点面板, 同样
  只随 parallel 出现 (2026-08-15 概率头接线). legacy_prob_head 读面板+
  特征现场构建, 无前置依赖, 恒在 legacy 预测前 (概率闸依赖 bundle) (2026-08-17).
  (2026-08-20) legacy 链优先: legacy 与 parallel 链互不依赖, legacy 短名单
  先落盘, 不再被 parallel+prob_head 挡住 (~4 分/重训日).
"""

import datetime as _dt
import json

import scripts.run_daily_automation as ma
from scripts.run_daily_automation import (
    _STEP_TIMEOUT_S,
    _STEPS,
    RETRAIN_WEEKDAY,
    _exit_status,
    _is_interrupt_rc,
    _write_state,
    plan_steps,
)

# 2026-08-13 = Thursday (weekday 3, 非重训日), 2026-08-14 = Friday (weekday 4, 重训日).
THU, FRI = _dt.date(2026, 8, 13), _dt.date(2026, 8, 14)


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
    assert plan_steps(THU) == [
        "refresh",
        "cyq",
        "legacy_prob_head",
        "legacy",
        "deliver",
        "parallel",
        "prob_head",
        "deliver_parallel",
        "drift",
        "drift_parallel",
    ]


def test_plan_steps_retrain_day_inserts_retrain():
    assert plan_steps(FRI) == [
        "refresh",
        "cyq",
        "retrain",
        "legacy_prob_head",
        "legacy",
        "deliver",
        "parallel",
        "prob_head",
        "deliver_parallel",
        "drift",
        "drift_parallel",
    ]


def test_plan_steps_force_retrain_inserts_retrain_on_non_retrain_day():
    """--force-retrain 在非周五也强制插入 retrain 步骤."""
    assert plan_steps(THU, force_retrain=True) == [
        "refresh",
        "cyq",
        "retrain",
        "legacy_prob_head",
        "legacy",
        "deliver",
        "parallel",
        "prob_head",
        "deliver_parallel",
        "drift",
        "drift_parallel",
    ]


def test_plan_steps_skip_parallel_drops_deliver_parallel():
    """--skip-parallel 只跑 legacy 链 (refresh 仍刷新检查点); 并行交付同步丢弃."""
    assert plan_steps(THU, skip_parallel=True) == [
        "refresh",
        "cyq",
        "legacy_prob_head",
        "legacy",
        "deliver",
        "drift",
        "drift_parallel",
    ]
    assert plan_steps(FRI, skip_parallel=True) == [
        "refresh",
        "cyq",
        "retrain",
        "legacy_prob_head",
        "legacy",
        "deliver",
        "drift",
        "drift_parallel",
    ]


def test_plan_steps_skip_checkpoints_and_retrain():
    steps = plan_steps(FRI, skip_checkpoints=True, skip_retrain=True)
    assert steps == [
        "legacy_prob_head",
        "legacy",
        "deliver",
        "parallel",
        "prob_head",
        "deliver_parallel",
        "drift",
        "drift_parallel",
    ]
    assert "refresh" not in steps and "retrain" not in steps


def test_plan_steps_all_skip_keeps_legacy_chain():
    assert plan_steps(
        THU, skip_checkpoints=True, skip_retrain=True, skip_parallel=True
    ) == ["legacy_prob_head", "legacy", "deliver", "drift", "drift_parallel"]


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
