"""重训进度静默看门狗 (2026-09-11).

病理: 特征构建 dim29-36 + auto-adopt + 时序变化段全无日志, 健康跑最长静默
25.1min (2026-09-10 clean run 校准, tmp_t/_retrain_20260910_1712_clean.out);
换页泥潭下同段 5h51m+ 零进度 (2026-08-14 / 2026-09-10 a1 两起, 均靠人工
击杀). ram_guard 只 WARNING 永不动手, 启动闸也拦不住 (a1 启动时 10GB 可用,
4 分钟后即挤兑; 健康跑同样会探 0.0G free — 本机 15.8G 训练恒贴崖边,
free 内存不是死活判据, 进度静默才是).

规则: 连续 timeout 秒无非 ram_guard 的 INFO+ 日志 → 判死, CRITICAL 记录 +
硬退 rc=86 (泥潭里常规退出/清理也会卡死, 必须 os._exit), 由外层队列/手工
按 rc 冷却重试. 默认阈值 45min = 1.8x 健康最大静默.
"""

from __future__ import annotations

import logging
import os
import threading
import time

_RAM_GUARD_PREFIX = "app.pipeline1.ram_guard"
STALL_EXIT_CODE = 86


class ProgressTracker(logging.Handler):
    """root handler: 非监控类 INFO+ 记录刷新进度时间戳."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._lock_ts = threading.Lock()
        self.last_ts = time.time()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == _RAM_GUARD_PREFIX or record.name.startswith(
            _RAM_GUARD_PREFIX + "."
        ):
            return
        with self._lock_ts:
            self.last_ts = time.time()

    def silent_for(self) -> float:
        with self._lock_ts:
            return time.time() - self.last_ts


def start_stall_watchdog(
    timeout_s: float, poll_s: float = 60.0, exit_fn=os._exit
) -> ProgressTracker:
    """挂 root handler + 起 daemon 线程; 静默超阈值调 exit_fn(86) 后线程退出."""
    tracker = ProgressTracker()
    logging.getLogger().addHandler(tracker)
    started_at = time.strftime("%H:%M:%S")
    _stop_evt = threading.Event()

    def _loop() -> None:
        while not _stop_evt.is_set():
            _stop_evt.wait(timeout=poll_s)
            if _stop_evt.is_set():
                return
            silent = tracker.silent_for()
            if silent < timeout_s:
                continue
            logging.getLogger(__name__).critical(
                "[stall-watchdog] %.0f 分钟无非监控日志 (阈值 %.0f 分钟), "
                "判换页泥潭, 硬退 rc=%d (watchdog %s 起跑)",
                silent / 60.0,
                timeout_s / 60.0,
                STALL_EXIT_CODE,
                started_at,
            )
            exit_fn(STALL_EXIT_CODE)
            return

    threading.Thread(target=_loop, name="stall-watchdog", daemon=True).start()
    tracker._stop_evt = _stop_evt  # noqa: SLF001 — 供单测 teardown 显式停止
    return tracker
