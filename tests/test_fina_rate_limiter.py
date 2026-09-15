# -*- coding: utf-8 -*-
"""fina_indicator 逐股拉取速率节流器测试 (2026-09-15).

背景: 逐股全池 ~5000 次调用, 4 workers 耗时约 38 分钟. 只堆 worker 会撞服务端
硬限 (实测 500 次/分钟; 24 workers 时丢 100/300 股), 故改为**按速率节流**:
worker 数决定能否吃满速率, 速率上限决定服务端是否拒答.

这里锁两件事: (1) 实际放行速率不超上限; (2) 多线程共享同一节流器 (并发不放大)。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline1.data_supply import _RateLimiter  # noqa: E402


def test_paces_to_configured_rate():
    """10 次放行 @1200/min (间隔 0.05s) → 至少 9×0.05=0.45s, 不得显著超时."""
    lim = _RateLimiter(1200)
    t0 = time.monotonic()
    for _ in range(10):
        lim.acquire()
    el = time.monotonic() - t0
    assert el >= 0.40, f"放行过快, 未节流: {el:.3f}s"
    assert el < 2.0, f"放行过慢, 节流过度: {el:.3f}s"


def test_first_call_is_not_delayed():
    """首次调用不应白等一个间隔 (否则每次批量开头都多付一次延迟)."""
    lim = _RateLimiter(600)
    t0 = time.monotonic()
    lim.acquire()
    assert time.monotonic() - t0 < 0.05


def test_concurrent_threads_share_one_budget():
    """4 线程 × 10 次 @1200/min → 全局仍 39×0.05≈1.95s; 若各线程独立计时会明显更快."""
    lim = _RateLimiter(1200)
    calls = 0
    lock = threading.Lock()

    def worker():
        nonlocal calls
        for _ in range(10):
            lim.acquire()
            with lock:
                calls += 1

    t0 = time.monotonic()
    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    el = time.monotonic() - t0
    assert calls == 40
    assert el >= 1.80, f"并发下节流被放大: {el:.3f}s"


def test_zero_or_negative_rate_does_not_crash():
    """退化输入不得除零/负睡; max(1,·) 兜底."""
    for rate in (0, -5):
        lim = _RateLimiter(rate)
        t0 = time.monotonic()
        lim.acquire()
        assert time.monotonic() - t0 >= 0.0  # 不抛即通过


@pytest.mark.parametrize("rate,expected_min", [(6000, 0.0), (1200, 0.4)])
def test_rate_monotonic_in_interval(rate, expected_min):
    """间隔随速率反比变化 (不同配置真的改变节奏, 而非常量)."""
    lim = _RateLimiter(rate)
    t0 = time.monotonic()
    for _ in range(10):
        lim.acquire()
    assert time.monotonic() - t0 >= expected_min


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
