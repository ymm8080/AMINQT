"""Tests for 手工 pipeline 数据前置 (scripts/_manual_preflight.py) +
_backfill_cyq_panel.cache_is_current 新鲜度秒退判据 (2026-09-08)."""

import pandas as pd

import scripts._manual_preflight as mp
from scripts._backfill_cyq_panel import cache_is_current

_EXPECTED_ARGV = [
    "scripts/_backfill_cyq_panel.py",
    "scripts/fetch_sw_daily_history.py",
    "scripts/_freshness_check.py",
]


def test_cache_is_current_true_when_covered():
    assert cache_is_current(
        pd.Timestamp("2026-09-08"),
        {"000001", "300001"},
        pd.Timestamp("2026-09-08"),
        {"000001", "300001"},
    )


def test_cache_is_current_false_when_date_lags():
    assert not cache_is_current(
        pd.Timestamp("2026-09-07"), {"000001"}, pd.Timestamp("2026-09-08"), {"000001"}
    )


def test_cache_is_current_false_when_symbol_missing():
    assert not cache_is_current(
        pd.Timestamp("2026-09-08"),
        {"000001"},
        pd.Timestamp("2026-09-08"),
        {"000001", "300001"},
    )


def test_run_preflight_runs_three_steps_and_survives_failure(monkeypatch, tmp_path):
    recorded: list[tuple[str, int]] = []

    def fake_watchdog(argv, fh, env, timeout_s):
        script = argv[2]
        # sw_history 失败 + freshness 超时 — 均不拦后续/不抛错
        rc, timed_out = (0, False)
        if script.endswith("fetch_sw_daily_history.py"):
            rc = 1
        if script.endswith("_freshness_check.py"):
            rc, timed_out = 0, True
        recorded.append((script, rc, timed_out))
        return rc, timed_out

    monkeypatch.setattr(mp, "_run_step_with_watchdog", fake_watchdog)
    monkeypatch.setattr(mp, "_under_chain", lambda: False)
    monkeypatch.setattr(mp, "LOG_DIR", str(tmp_path))

    results = mp.run_preflight("test")

    assert [s for s, _ in results] == ["cyq", "sw_history", "freshness"]
    # 实际子进程 argv 按链 _STEPS 顺序拼出
    assert [a[0] for a in recorded] == _EXPECTED_ARGV
    assert dict(results)["sw_history"] == 1
    # 超时步记 rc=124 (与链语义一致)
    assert dict(results)["freshness"] == 124
    assert len(recorded) == 3


def test_run_preflight_skips_under_chain(monkeypatch):
    monkeypatch.setattr(mp, "_under_chain", lambda: True)
    called = []
    monkeypatch.setattr(mp, "_run_step_with_watchdog", lambda *a, **k: called.append(a))
    assert mp.run_preflight("test") is None
    assert called == []
