# -*- coding: utf-8 -*-
"""LEGACY 夜链 (run_legacy_automation.py, 2026-10-17) 计划与接线测试。

覆盖: 周五 retrain 插入位置、步骤可执行性 (有 timeout/在 _STEPS)、依赖顺序、
链级哨兵注册 (与 PARALLEL 链互斥)、PARALLEL 链尾跨模块步骤 (密度先于 combined)。
"""

from __future__ import annotations

import datetime as _dt

from scripts._run_guard import (
    CHAIN_SENTINELS,
    LEGACY_ORCHESTRATOR_SENTINEL,
    PARALLEL_ORCHESTRATOR_SENTINEL,
)
from scripts.run_daily_automation import _STEP_TIMEOUT_S, _STEPS
from scripts.run_legacy_automation import (
    _DEPENDS_LOCAL,
    _NON_CRITICAL_STEPS,
    STEP_ORDER,
    plan_steps,
)
from scripts.run_parallel_automation import (
    _DEPENDS_LOCAL as _PARALLEL_DEPENDS,
)
from scripts.run_parallel_automation import (
    _NON_CRITICAL_STEPS as _PARALLEL_NON_CRITICAL,
)
from scripts.run_parallel_automation import (
    _WAIT_MAX_TICKS as _PARALLEL_WAIT_MAX_TICKS,
)
from scripts.run_parallel_automation import (
    STEP_ORDER as _PARALLEL_STEP_ORDER,
)

THU = _dt.date(2026, 10, 15)  # 周四
FRI = _dt.date(2026, 10, 16)  # 周五
SAT = _dt.date(2026, 10, 17)  # 周六


def test_plan_steps_weekday_no_retrain():
    for d in (THU, SAT):
        assert plan_steps(d) == STEP_ORDER


def test_plan_steps_friday_inserts_retrain_after_canary():
    steps = plan_steps(FRI)
    assert "retrain" in steps
    assert steps.index("retrain") == steps.index("canary") + 1
    assert steps.index("retrain") < steps.index("legacy_prob_head")


def test_plan_steps_force_retrain_on_non_retrain_day():
    steps = plan_steps(THU, force_retrain=True)
    assert "retrain" in steps


def test_every_step_has_definition_and_timeout():
    for steps in (STEP_ORDER, plan_steps(FRI)):
        for step in steps:
            assert step in _STEPS, f"{step} 不在 _STEPS"
            assert step in _STEP_TIMEOUT_S, f"{step} 无超时"


def test_deps_precede_dependents():
    for steps in (STEP_ORDER, plan_steps(FRI)):
        for step, dep in _DEPENDS_LOCAL.items():
            if step in steps and dep in steps:
                assert steps.index(dep) < steps.index(step), f"{dep} 必须先于 {step}"


def test_critical_delivery_steps_not_non_critical():
    for critical in ("legacy_prob_head", "legacy", "deliver"):
        assert critical not in _NON_CRITICAL_STEPS


def test_legacy_chain_registered_as_sentinel():
    assert LEGACY_ORCHESTRATOR_SENTINEL in CHAIN_SENTINELS
    assert PARALLEL_ORCHESTRATOR_SENTINEL in CHAIN_SENTINELS


def test_parallel_chain_cross_module_tail_order():
    """密度/slowbull 必须在 combined 之前; combined 在 THS 推送之前。"""
    order = _PARALLEL_STEP_ORDER
    assert order.index("prob10dens_push") < order.index("stocklist_combined")
    assert order.index("slowbull_shadow") < order.index("stocklist_combined")
    assert order.index("stocklist_combined") < order.index("ths_push")
    assert order.index("final_stocklist") == len(order) - 1
    for step in order:
        assert step in _STEPS, f"{step} 不在 _STEPS"
        assert step in _STEP_TIMEOUT_S, f"{step} 无超时"


def test_parallel_chain_density_and_combined_critical():
    """密度与 combined 是四模块交付物, 失败要判链失败 (勿降级为旁路)。"""
    assert "prob10dens_push" not in _PARALLEL_NON_CRITICAL
    assert "stocklist_combined" not in _PARALLEL_NON_CRITICAL


def test_parallel_chain_wait_window_covers_friday_legacy_tail():
    """周五 LEGACY 链 (21:30 + retrain) ~02:00 才清场: 23:30 起等窗须 ≥ 2.5h。"""
    assert _PARALLEL_WAIT_MAX_TICKS * 5 * 60 >= 2.5 * 3600
    for step, dep in _PARALLEL_DEPENDS.items():
        if step in _PARALLEL_STEP_ORDER and dep in _PARALLEL_STEP_ORDER:
            assert _PARALLEL_STEP_ORDER.index(dep) < _PARALLEL_STEP_ORDER.index(step)
