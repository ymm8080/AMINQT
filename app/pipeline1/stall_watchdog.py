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

★ 2026-09-14 补: 硬退前先清子进程树。`os._exit` 不带走子进程 —— 09-14 链上
重训判死时, 它派生的 `_dual_pkg_finaltop_compare.py` 成孤儿, 该脚本在
HEAVY_SENTINELS 里, 占着运行守卫不放; 紧随其后的 `legacy` 预测撞守卫 rc=3
秒退、`deliver` 无清单 rc=1, 当天 LEGACY 交付整段缺失 (09-13 同型复发)。
故 exit 前 `kill_process_children()` 递归终结子孙, 清理失败也照退。
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


def kill_process_children(timeout_s: float = 5.0) -> int:
    """递归终结本进程的全部子孙, 返被清理的进程数 (失败返 0).

    硬退前的清场, 防孤儿占运行守卫 (见模块 docstring 2026-09-14 条)。
    """
    try:
        import psutil
    except ImportError:  # 无 psutil 时降级: 照退, 只是可能留孤儿
        logging.getLogger(__name__).warning(
            "[stall-watchdog] psutil 不可用, 跳过子进程清理"
        )
        return 0
    try:
        kids = psutil.Process().children(recursive=True)
    except Exception:  # noqa: BLE001 — 枚举失败不该挡住硬退
        logging.getLogger(__name__).warning(
            "[stall-watchdog] 子进程枚举失败, 跳过清理", exc_info=True
        )
        return 0
    if not kids:
        return 0
    for k in kids:
        try:
            k.terminate()
        except Exception:  # noqa: BLE001 — 已死/无权限都无所谓
            pass
    _gone, alive = psutil.wait_procs(kids, timeout=timeout_s)
    for k in alive:
        try:
            k.kill()
        except Exception:  # noqa: BLE001
            pass
    psutil.wait_procs(alive, timeout=2.0)
    return len(kids)


def start_stall_watchdog(
    timeout_s: float,
    poll_s: float = 60.0,
    exit_fn=os._exit,
    kill_fn=kill_process_children,
) -> ProgressTracker:
    """挂 root handler + 起 daemon 线程; 静默超阈值先清子进程树再调 exit_fn(86)."""
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
            try:
                n = kill_fn()
            except Exception:  # noqa: BLE001 — 清场失败也必须硬退
                logging.getLogger(__name__).warning(
                    "[stall-watchdog] 子进程清理抛错 (忽略), 仍硬退", exc_info=True
                )
            else:
                if n:
                    logging.getLogger(__name__).critical(
                        "[stall-watchdog] 已清理 %d 个子进程, 防孤儿占运行守卫", n
                    )
            exit_fn(STALL_EXIT_CODE)
            return

    threading.Thread(target=_loop, name="stall-watchdog", daemon=True).start()
    tracker._stop_evt = _stop_evt  # noqa: SLF001 — 供单测 teardown 显式停止
    return tracker
