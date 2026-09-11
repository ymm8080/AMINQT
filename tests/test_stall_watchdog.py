"""重训进度静默看门狗单测 (2026-09-11).

规则: 非 ram_guard 的 INFO+ 日志静默超阈值 → rc=86 硬退 (换页泥潭判死;
08-14 / 09-10 a1 两起 6h+ 零进度均靠人工击杀). ram_guard 的 WARNING 唠叨
不算进度 — 泥潭里它是唯一还活着的日志源, 若算进度看门狗永不触发.
"""

from __future__ import annotations

import logging
import time

import pytest

from app.pipeline1.stall_watchdog import (
    STALL_EXIT_CODE,
    ProgressTracker,
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
    calls: list[int] = []
    start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    time.sleep(_TIMEOUT + 6 * _POLL + 0.5)
    assert calls == [STALL_EXIT_CODE]


def test_regular_info_logs_keep_alive():
    calls: list[int] = []
    start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    for _ in range(
        60
    ):  # 3.0s 总时长 > 阈值, 每 0.05s 一条 INFO 刷新 (留足 CI 调度余量)
        logging.getLogger("app.pipeline1.train_runner").info("tick")
        time.sleep(0.05)
    assert calls == []


def test_ram_guard_chatter_is_not_progress():
    calls: list[int] = []
    start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    for _ in range(12):  # 泥潭形态: 只有 ram_guard 在 WARNING, 仍须判死
        time.sleep(0.1)
        logging.getLogger("app.pipeline1.ram_guard").warning("内存挤兑警报")
    time.sleep(_TIMEOUT + 4 * _POLL + 0.5)
    assert calls == [STALL_EXIT_CODE]


def test_debug_records_do_not_count():
    calls: list[int] = []
    start_stall_watchdog(_TIMEOUT, poll_s=_POLL, exit_fn=calls.append)
    logger = logging.getLogger("app.pipeline1.feature_engine_v35")
    for _ in range(12):  # DEBUG 刷屏不算进度 (handler 级 INFO 过滤)
        time.sleep(0.1)
        logger.debug("debug tick")
    time.sleep(_TIMEOUT + 4 * _POLL + 0.5)
    assert calls == [STALL_EXIT_CODE]
