# -*- coding: utf-8 -*-
"""[0913 夜] 排队接力 (夜链第14棒): 等 relay-8 链 (main feat 复审) 整体退场 →
跑 dual 特征集 cls 对齐复审 (_dual_cls_feat_review_0913)。

与 relay-8 等待器不同点: 上游判据用**进程存活**而非 state 文件 — 等待器只在
终态写 state, 等待期数小时无文件 → 45min 宽限必然过期抢跑 (坑⑥)。relay-8
等待器用 subprocess.run 阻塞跑目标 → 它的进程退出 == 目标已跑完 (或自身挂),
此时再等 find_conflicts 清场即可, 无需宽限。
本文件名不含任何哨兵子串 (坑⑤)。
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from scripts._run_guard import find_conflicts

STATE = os.path.join(HERE, "_wait_r14_dualfeat_0913.state.json")
LOG = os.path.join(HERE, "_wait_r14_dualfeat_0913.log")
TARGET = os.path.join(HERE, "_dual_cls_feat_review_0913.py")
UPSTREAM_WAITER = "_wait_r8_featrev_0913.py"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def state(phase: str, **kw) -> None:
    payload = {"phase": phase, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def upstream_gone() -> bool:
    """relay-8 等待器进程 (含其阻塞中的目标) 是否已退场."""
    import psutil

    me = os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            cl = p.info["cmdline"] or []
            if any(UPSTREAM_WAITER in str(a) for a in cl):
                return False
        except Exception:
            continue
    return True


def main() -> int:
    waited = 0
    while True:
        conflicts = find_conflicts()
        gone = upstream_gone()
        if not conflicts and gone:
            break
        if waited % 10 == 0:
            log(f"等待: 冲突={[c['sentinel'] for c in conflicts]} 上游退场={gone}")
        time.sleep(60)
        waited += 1
        if waited > 720:  # 12h 上限 (夜链 relay7→6→8→本棒 链条长)
            state("FAIL_WAIT_TIMEOUT")
            log("FAIL_WAIT_TIMEOUT (12h)")
            return 4
    log("清场+上游退场, 启动 dual 特征集 cls 对齐复审")
    rc = None
    for attempt in range(1, 3):
        with open(LOG, "a", encoding="utf-8") as out:
            rc = subprocess.run(
                [sys.executable, "-u", TARGET],
                cwd=ROOT,
                stdout=out,
                stderr=subprocess.STDOUT,
            ).returncode
        if rc == 0:
            state("DONE", attempts=attempt)
            log(f"DONE (尝试 {attempt})")
            return 0
        log(f"尝试 {attempt} rc={rc}, 冷却 300s 重试")
        time.sleep(300)
    state("FAIL_RC", last_rc=rc)
    log(f"FAIL_RC rc={rc}")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
