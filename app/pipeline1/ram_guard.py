"""重训内存独占闸 (2026-08-15).

规则 (用户定案): 重训期间机器须独占 — 其他重活 (250d 复验/参数扫描等) 抢内存
→ 08-14 训练 8.4h 零模型完成. 代码强制两层:

1. 启动闸: 可用物理内存低于下限 → 拒绝启动 (SystemExit 2), 大白话提示;
2. 运行期警报: daemon 线程每 poll_s 采样, 低于下限 → 每段挤兑打一条 WARNING
   (不杀进程 — 训练有 per-model 检查点, 误杀反而破坏可恢复性).
"""

from __future__ import annotations

import logging
import threading

import psutil

logger = logging.getLogger(__name__)


def free_physical_bytes() -> int:
    """当前可用物理内存 (字节)."""
    return int(psutil.virtual_memory().available)


def should_block(free_bytes: int, min_free_bytes: int) -> bool:
    """纯判定: 可用内存低于下限 → True (拒绝启动)."""
    return free_bytes < min_free_bytes


def _fmt_gb(b: int) -> str:
    return f"{b / 1024**3:.1f}G"


def check_startup_gate(min_free_bytes: int) -> None:
    """启动闸: 内存不足 → 日志大白话提示 + SystemExit(2)."""
    free = free_physical_bytes()
    if should_block(free, min_free_bytes):
        logger.error(
            "重训启动被拒: 机器可用内存只剩 %s, 低于下限 %s. 重训须独占机器 "
            "(08-14 教训: 抢内存 → 训练 8.4h 零模型完成). 请先关掉其他重活再跑, "
            "确认无重活后可调低 RETRAIN_RAM_GUARD_MIN_FREE_GB.",
            _fmt_gb(free),
            _fmt_gb(min_free_bytes),
        )
        raise SystemExit(2)


def start_monitor(min_free_bytes: int, poll_s: float) -> threading.Thread:
    """运行期警报线程 (daemon): 低于下限持续 → 每段挤兑一条 WARNING. 永不杀进程."""
    stop = threading.Event()

    def _loop():
        warned = False
        while not stop.wait(poll_s):
            low = should_block(free_physical_bytes(), min_free_bytes)
            if low and not warned:
                logger.warning(
                    "内存挤兑警报: 可用内存 %s 低于下限 %s — 检测到其他重活抢内存, "
                    "训练可能像 08-14 那样数小时零完成. 建议立刻停掉并行的回测/扫描.",
                    _fmt_gb(free_physical_bytes()),
                    _fmt_gb(min_free_bytes),
                )
                warned = True
            elif not low:
                warned = False

    t = threading.Thread(target=_loop, name="ram-guard", daemon=True)
    t.start()
    return t
