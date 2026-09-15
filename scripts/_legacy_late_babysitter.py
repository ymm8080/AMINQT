"""_legacy_late_babysitter.py — 链跑完后补跑今天漏掉的 LEGACY 派生交付 (2026-09-14).

背景 (09-14 事故): 链的 retrain 被 stall_watchdog 硬退 rc=86 时**没带走子进程** —
横插出来的 `_dual_pkg_finaltop_compare.py` 成孤儿, 在哨兵表里, 占着运行守卫。紧随其后的
`legacy` (预测) 撞守卫 rc=3 秒退 → 无 `data/lists/list_{tag}.parquet` → `deliver` rc=1。
链把这两步判 FAIL 后继续往下走, **不会重试**。

一击带走的不止 LEGACY 本身 —— 判 FAIL 后链仍继续, 一路降级跳页:
  · `prob10dens_push` → `candidates_{tag}.parquet` 是 LEGACY 管道产的 → 跳 (fail-safe)
  · `final_stocklist` / `stocklist_combined` → LEGACY 页 + 密度页双双缺失
即 LEGACY 一步死, 当天合并表少两页。

本脚本等链进程退干净 + 守卫连续空档后, 顺序补跑这五步:
  legacy(预测) → deliver → prob10dens → final_stocklist → stocklist_combined
最前面两步失败即退出 (下游全部无源); 后三步互相不中止 (缺页会降级跳页), 末尾汇总报失败。
以独立进程运行 (Start-Process / cmd /c), 会话退出不影响。发射后 rc=3 退避重试。

启动:
  cmd /c "python -u scripts\\_legacy_late_babysitter.py --tag 20260914 --chain-pid 16816
          > logs\\legacy_late_20260914.log 2>&1"
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts._run_guard import HEAVY_SENTINELS, find_conflicts

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
POLL_S = 60
FREE_STREAK = 3  # 连续 3 次无冲突才发射 (避开链两步之间的空档)
CONFLICT_RETRY = 6
CONFLICT_BACKOFF_S = 300


def log(msg: str) -> None:
    logger.info("[%s] %s", f"{datetime.now():%m-%d %H:%M:%S}", msg)


def _alive(pid: int) -> bool:
    import psutil

    try:
        return psutil.Process(pid).is_running()
    except Exception:
        return False


def wait_chain_gone(chain_pid: int | None, max_hours: float) -> bool:
    """等链进程退出。无 pid 则只用守卫判空。超时返 False。"""
    deadline = time.monotonic() + max_hours * 3600
    if chain_pid and _alive(chain_pid):
        log(f"等链 (PID {chain_pid}) 退出 …")
        while _alive(chain_pid):
            if time.monotonic() > deadline:
                log("硬超时: 链仍未退出, 放弃")
                return False
            time.sleep(POLL_S)
        log("链已退出, 进守卫空档等待")
    return True


def wait_free_slot(max_hours: float) -> bool:
    """连续 FREE_STREAK 次守卫无重活 → 空档。"""
    deadline = time.monotonic() + max_hours * 3600
    streak = 0
    while True:
        hits = find_conflicts(sentinels=HEAVY_SENTINELS)
        if not hits:
            streak += 1
            if streak >= FREE_STREAK:
                log(f"守卫连续 {streak} 次空档, 发射")
                return True
        else:
            if streak:
                log(f"又见重活 {hits[0]['sentinel']} (PID {hits[0]['pid']}), 重新计数")
            streak = 0
        if time.monotonic() > deadline:
            log("硬超时: 守卫一直不空, 放弃")
            return False
        time.sleep(POLL_S)


def run_step(name: str, argv: list[str]) -> int:
    """跑一步; rc=3 (撞守卫) 退避重试, 其他非零直接返回。"""
    for attempt in range(1, CONFLICT_RETRY + 1):
        log(f"{name} 发射 (第 {attempt} 次): {' '.join(argv)}")
        rc = subprocess.call(argv, cwd=str(REPO))
        if rc == 0:
            log(f"{name} ok rc=0")
            return 0
        if rc != 3:
            log(f"{name} FAIL rc={rc} — 非守卫冲突, 不重试")
            return rc
        log(f"{name} 撞守卫 rc=3, {CONFLICT_BACKOFF_S}s 后重试")
        time.sleep(CONFLICT_BACKOFF_S)
    log(f"{name} 重试 {CONFLICT_RETRY} 次仍撞守卫, 放弃")
    return 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--chain-pid", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--skip-deliver", action="store_true")
    a = ap.parse_args()

    log(f"启动 tag={a.tag} chain_pid={a.chain_pid} max_hours={a.max_hours}")
    if not wait_chain_gone(a.chain_pid, a.max_hours):
        return 1
    if not wait_free_slot(a.max_hours):
        return 1

    rc = run_step("legacy", [PY, "-u", "scripts/_gen_legacy_list.py", a.tag])
    if rc != 0:
        log("legacy 预测失败 — 下游全部无源, 退出")
        return rc
    if a.skip_deliver:
        log("--skip-deliver: 只跑预测, 交付与下游留给链外手动")
        return 0
    rc = run_step("deliver", [PY, "-u", "scripts/_deliver_legacy_list.py", a.tag])
    if rc != 0:
        log("deliver 失败 — 下游无源, 退出")
        return rc

    # 下游三步: 缺一页只是降级跳页, 不互相中止, 末尾汇总。
    # 密度步带 `--gen-only`: 只出清单/清单页, **不推同花顺自选股** —— 补跑不该替你
    # 触发不可逆的自选股写入 (链今晚的 parallel 推已被死区闸拦, 密度该不该推由用户定)。
    downstream = (
        (
            "prob10dens",
            [PY, "-u", "scripts/_prob10_density_shadow.py", a.tag, "--gen-only"],
        ),
        ("final_stocklist", [PY, "-u", "scripts/_final_stocklist.py", a.tag]),
        ("stocklist_combined", [PY, "-u", "scripts/_stocklist_combined.py", a.tag]),
    )
    failed = [n for n, argv in downstream if run_step(n, argv) != 0]
    if failed:
        log(f"补跑结束, 失败步 {failed} — 需看日志")
        return 1
    log("补跑五步全 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
