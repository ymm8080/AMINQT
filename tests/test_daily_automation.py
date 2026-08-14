"""Tests for scripts/run_daily_automation 步骤编排 (2026-08-13).

plan_steps 是纯函数, 决定当日四模块自动化的执行序列:
  [refresh?] [retrain?] [parallel?] legacy deliver [deliver_parallel?]
关键不变式: deliver_parallel 依赖当日 fresh parallel 重生成 (run_dir), 故
  parallel 被跳过 (--skip-parallel) 时 deliver_parallel 必须同步丢弃,
  否则会交付旧 run_dir 脏数据.
"""

import datetime as _dt

from scripts.run_daily_automation import RETRAIN_WEEKDAY, plan_steps

# 2026-08-13 = Thursday (weekday 3, 非重训日), 2026-08-14 = Friday (weekday 4, 重训日).
THU, FRI = _dt.date(2026, 8, 13), _dt.date(2026, 8, 14)


def test_fixture_dates_hit_expected_weekdays():
    assert RETRAIN_WEEKDAY == 4
    assert THU.weekday() != RETRAIN_WEEKDAY
    assert FRI.weekday() == RETRAIN_WEEKDAY


def test_plan_steps_weekday_full_chain():
    assert plan_steps(THU) == [
        "refresh",
        "parallel",
        "legacy",
        "deliver",
        "deliver_parallel",
    ]


def test_plan_steps_retrain_day_inserts_retrain():
    assert plan_steps(FRI) == [
        "refresh",
        "retrain",
        "parallel",
        "legacy",
        "deliver",
        "deliver_parallel",
    ]


def test_plan_steps_skip_parallel_drops_deliver_parallel():
    """--skip-parallel 只跑 legacy 链 (refresh 仍刷新检查点); 并行交付同步丢弃."""
    assert plan_steps(THU, skip_parallel=True) == ["refresh", "legacy", "deliver"]
    assert plan_steps(FRI, skip_parallel=True) == [
        "refresh",
        "retrain",
        "legacy",
        "deliver",
    ]


def test_plan_steps_skip_checkpoints_and_retrain():
    steps = plan_steps(FRI, skip_checkpoints=True, skip_retrain=True)
    assert steps == ["parallel", "legacy", "deliver", "deliver_parallel"]
    assert "refresh" not in steps and "retrain" not in steps


def test_plan_steps_all_skip_keeps_legacy_chain():
    assert plan_steps(
        THU, skip_checkpoints=True, skip_retrain=True, skip_parallel=True
    ) == ["legacy", "deliver"]
