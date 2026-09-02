"""_rankkey_rerun_chain.py — 等第一轮 125d 全量退出后, 自动发射撤q50闸重跑 (2026-09-02).

背景: 第一轮 (pid 由 argv 传入) 池口径 = q50 闸仍在 (旧生产口径), 其判词留作
对照基准. 生产已撤 q50 符号闸 (q50_sign_gate=False, 三臂回放判死), 重跑按新
口径出正式判词. wf 头检查点与池无关 → 重跑免训练 (~2h), 只剩推理 ~2-3h.

流程: 轮询等 pid 退出 → 校验第一轮日志含 === DONE === (否则 fail-fast 中止,
第一轮失败重跑大概率同死) → 等 CHAIN_SENTINELS 无冲突 (早 08:30 链让路) →
发射 scripts/_rankkey_multiseed_sweep.py → logs/_rankkey_multiseed_rerun.log.
rc=2 守卫撞车退避 10 min ×3; 12h 硬超时.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import psutil

from scripts._run_guard import CHAIN_SENTINELS, find_conflicts

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
SWEEP = REPO / "scripts" / "_rankkey_multiseed_sweep.py"
RUN1_LOG = LOGS / "_rankkey_multiseed_full.log"
RERUN_LOG = LOGS / "_rankkey_multiseed_rerun.log"
POLL_S = 30
CHAIN_POLL_S = 300
LAUNCH_RETRY = 3
RETRY_BACKOFF_S = 600
CAP_S = 12 * 3600


def log(msg: str) -> None:
    logger.info("[%s] %s", f"{datetime.now():%m-%d %H:%M:%S}", msg)


def main() -> int:
    wait_pid = int(sys.argv[1])
    t0 = time.time()
    log(f"chain-launcher 启动 pid={os.getpid()} 等待第一轮 pid={wait_pid}")
    while psutil.pid_exists(wait_pid):
        if time.time() - t0 > CAP_S:
            log("HARD TIMEOUT 等第一轮退出超 12h, 放弃")
            return 1
        time.sleep(POLL_S)
    log("第一轮已退出, 校验 DONE 标记")
    txt = (
        RUN1_LOG.read_text(encoding="utf-8", errors="replace")
        if RUN1_LOG.exists()
        else ""
    )
    if "=== DONE ===" not in txt:
        log("第一轮无 DONE 标记 (崩溃/未完成) — fail-fast 不发射重跑, 人工排查")
        return 2

    for attempt in range(1, LAUNCH_RETRY + 1):
        hits = find_conflicts(sentinels=CHAIN_SENTINELS)
        if hits:
            log(f"有链/重活冲突 (尝试 {attempt}/{LAUNCH_RETRY}), 5 min 后再查: {hits}")
            time.sleep(CHAIN_POLL_S)
            continue
        try:
            fh = open(RERUN_LOG, "a", encoding="utf-8")
        except OSError as exc:
            log(f"打开日志失败: {RERUN_LOG.name}: {exc}")
            return 3
        with fh:
            log(f"发射重跑 (q50_sign_gate=False 口径) → {RERUN_LOG.name}")
            try:
                rc = subprocess.call(
                    [sys.executable, "-u", str(SWEEP)],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO),
                )
            except Exception as exc:
                log(f"重跑子进程异常 (尝试 {attempt}/{LAUNCH_RETRY}): {exc}")
                return 3
        if rc == 0:
            log("重跑 rc=0 完成")
            return 0
        if rc == 2:
            log(f"重跑 rc=2 守卫拦截 (尝试 {attempt}/{LAUNCH_RETRY}), 退避重试")
            time.sleep(RETRY_BACKOFF_S)
            continue
        log(f"重跑 FAILED rc={rc} — fail-fast; wf 检查点在, 可续跑")
        return 3
    log("LAUNCH RETRIES EXHAUSTED")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
