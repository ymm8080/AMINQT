"""重训进度静默看门狗单测 (2026-09-11).

规则: 非 ram_guard 的 INFO+ 日志静默超阈值 → rc=86 硬退 (换页泥潭判死;
08-14 / 09-10 a1 两起 6h+ 零进度均靠人工击杀). ram_guard 的 WARNING 唠叨
不算进度 — 泥潭里它是唯一还活着的日志源, 若算进度看门狗永不触发.

注: 原 wall-clock 时序测试在 CI 低配 runner 上偶发失败 (线程调度延迟 > 2.0s
即误触发), 现改为直接测 ProgressTracker.silent_for() 状态 +
exit_fn 调用, 去 wall-clock 依赖, 语义等价.

★ 2026-09-14 补: 硬退前必须先清子进程树 (kill_fn), 顺序 kill → exit。
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest

from app.pipeline1.stall_watchdog import (
    STALL_EXIT_CODE,
    ProgressTracker,
    kill_process_children,
    start_stall_watchdog,
)

_TIMEOUT = 2.0
_POLL = 0.05


@pytest.fixture(autouse=True)
def _clean_root_logger():
    # 生产入口 basicConfig(level=INFO); 单测里 root 默认 WARNING 会滤掉 INFO.
    # 保存/恢复 + 清掉本文件测试挂上的 ProgressTracker, 不污染其他测试.
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.INFO)
    target = logging.getLogger("app.pipeline1.train_runner")
    prev_target_level = target.level
    target.setLevel(logging.INFO)
    yield
    root.setLevel(prev_level)
    target.setLevel(prev_target_level)
    for h in list(root.handlers):
        if isinstance(h, ProgressTracker):
            # 显式停止 watchdog 线程, 防止 daemon 线程泄漏到后续测试
            if hasattr(h, "_stop_evt"):
                h._stop_evt.set()
            root.removeHandler(h)


def test_silence_beyond_timeout_fires_once():
    """watchdog 线程在静默超阈值后调 exit_fn(86) 一次."""
    calls: list[int] = []
    tracker = start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    # 直接 mock time.time 让 silent_for() 返回 > timeout 的值, 避免 wall-clock 依赖
    with patch.object(tracker, "silent_for", return_value=_TIMEOUT + 1.0):
        # 等 watchdog 线程下一轮 poll (最多 poll_s + 余量)
        time.sleep(_POLL + 0.2)
    assert calls == [STALL_EXIT_CODE]


def test_regular_info_logs_keep_alive():
    """非 ram_guard 的 INFO+ 日志持续刷新 silent_for → watchdog 不触发."""
    calls: list[int] = []
    tracker = start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    # 模拟 6s 内持续发 INFO (silent_for 始终 < timeout)
    for _ in range(120):
        logging.getLogger("app.pipeline1.train_runner").info("tick")
        # 直接断言: 每次发日志后 silent_for 立即归零 (核心机制)
        assert tracker.silent_for() < _TIMEOUT
        time.sleep(0.001)  # 1ms 让出, 不依赖 wall-clock
    assert calls == []


def test_ram_guard_chatter_is_not_progress():
    """ram_guard WARNING 不刷新 progress → watchdog 触发."""
    calls: list[int] = []
    tracker = start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    # 模拟 ram_guard 在叫, 但 silent_for 仍超阈值 (emit 里过滤了 ram_guard)
    with patch.object(tracker, "silent_for", return_value=_TIMEOUT + 1.0):
        logging.getLogger("app.pipeline1.ram_guard").warning("内存挤兑警报")
        time.sleep(_POLL + 0.2)
    assert calls == [STALL_EXIT_CODE]


def test_debug_records_do_not_count():
    """DEBUG 日志被 handler filter 滤掉, 不刷新 progress → watchdog 触发."""
    calls: list[int] = []
    tracker = start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    # 模拟 DEBUG 刷屏, 但 silent_for 仍超阈值 (handler level=INFO 过滤 DEBUG)
    with patch.object(tracker, "silent_for", return_value=_TIMEOUT + 1.0):
        logger = logging.getLogger("app.pipeline1.feature_engine_v35")
        logger.debug("debug tick")
        time.sleep(_POLL + 0.2)
    assert calls == [STALL_EXIT_CODE]


def test_progress_tracker_emit_resets_silent():
    """emit 后 last_ts 更新, silent_for 归零."""
    tracker = ProgressTracker()
    # 初始 last_ts 为构造时, silent_for 应为一个很小值
    assert tracker.silent_for() < 0.1
    time.sleep(0.2)
    assert tracker.silent_for() >= 0.2
    # 发一条 INFO → emit → last_ts 刷新
    record = logging.LogRecord(
        name="app.pipeline1.train_runner",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="tick",
        args=(),
        exc_info=None,
    )
    tracker.emit(record)
    assert tracker.silent_for() < 0.01


def test_progress_tracker_ignores_ram_guard():
    """ram_guard 的日志被 emit 过滤, 不刷新 last_ts."""
    tracker = ProgressTracker()
    time.sleep(0.2)
    assert tracker.silent_for() >= 0.2
    # ram_guard 的 WARNING → emit 过滤, last_ts 不动
    record = logging.LogRecord(
        name="app.pipeline1.ram_guard",
        level=logging.WARNING,
        pathname="test.py",
        lineno=1,
        msg="ram alert",
        args=(),
        exc_info=None,
    )
    tracker.emit(record)
    # silent_for 仍保持原值 (未被重置)
    assert tracker.silent_for() >= 0.2


def test_stall_watchdog_kills_children_before_exit():
    """硬退前先清子进程树: 09-14 孤儿 `_dual_pkg_finaltop_compare.py` 占运行守卫,
    把紧随其后的 legacy 预测打成 rc=3 → 当天 LEGACY 交付整段缺失。顺序必须是
    先 kill 再 exit。"""
    order: list[str] = []
    tracker = start_stall_watchdog(
        _TIMEOUT,
        poll_s=_POLL,
        exit_fn=lambda code: order.append(f"exit:{code}"),
        kill_fn=lambda: order.append("kill") or 1,
    )
    with patch.object(tracker, "silent_for", return_value=_TIMEOUT + 1.0):
        time.sleep(_POLL + 0.2)
    assert order == ["kill", f"exit:{STALL_EXIT_CODE}"]


def test_stall_watchdog_exits_even_if_kill_raises():
    """清理抛错不能挡住硬退 — 泥潭里退不出去比留孤儿更糟。"""
    calls: list[int] = []

    def _boom():
        raise RuntimeError("psutil 挂了")

    tracker = start_stall_watchdog(
        _TIMEOUT, poll_s=_POLL, exit_fn=calls.append, kill_fn=_boom
    )
    with patch.object(tracker, "silent_for", return_value=_TIMEOUT + 1.0):
        time.sleep(_POLL + 0.2)
    assert calls == [STALL_EXIT_CODE]


def test_kill_process_children_smoke():
    """真 psutil 路径跑得通且不抛错 (子进程有无都返回计数 int)。"""
    n = kill_process_children(timeout_s=0.5)
    assert isinstance(n, int) and n >= 0
