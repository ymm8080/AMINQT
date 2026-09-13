# -*- coding: utf-8 -*-
"""[0913 夜] 排队接力 (夜链第8棒, 改名重发版): 等 cls 参数杠杆扫描等待器终态 →
跑 main 特征集 cls 对齐复审 (_main_cls_feat_review_0913)。

改名原因: 原等待器文件名 _wait_main_cls_feat_review_0913.py 恰包含哨兵
_main_cls_feat_review_0913.py 为子串 → 自己的 cmdline 被其他重活的
find_conflicts 误判为活冲突 (01:20 missfeat 清场被卡, 对抗性审查坐实)。
本文件名不含任何哨兵子串。逻辑与原版逐字一致, 仅 STATE/LOG 前缀改名。

排序确定性: 看 _wait_cls_param_sweep_0913.state.json phase ∈ TERMINAL 且
find_conflicts 清场才启动。state 缺失 → 宽限 45min 后仅凭清场放行。
重试x2 + 300s 冷却 + 终态 state 文件 (fail-fast)。
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

STATE = os.path.join(HERE, "_wait_r8_featrev_0913.state.json")
LOG = os.path.join(HERE, "_wait_r8_featrev_0913.log")
TARGET = os.path.join(HERE, "_main_cls_feat_review_0913.py")
UPSTREAM_STATE = os.path.join(HERE, "_wait_cls_param_sweep_0913.state.json")
TERMINAL = {"DONE", "FAIL_RC", "FAIL_WAIT_TIMEOUT"}
GRACE_ABSENT_S = 45 * 60


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def state(phase: str, **kw) -> None:
    payload = {"phase": phase, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def upstream_terminal(started: float) -> bool:
    try:
        with open(UPSTREAM_STATE, encoding="utf-8") as fh:
            return json.load(fh).get("phase") in TERMINAL
    except Exception:
        return time.time() - started > GRACE_ABSENT_S


def main() -> int:
    started = time.time()
    waited = 0
    while True:
        conflicts = find_conflicts()
        if not conflicts and upstream_terminal(started):
            break
        if waited % 10 == 0:
            log(f"等待: 冲突={[c['sentinel'] for c in conflicts]} 上游终态={upstream_terminal(started)}")
        time.sleep(60)
        waited += 1
        if waited > 480:  # 8h 上限
            state("FAIL_WAIT_TIMEOUT")
            log("FAIL_WAIT_TIMEOUT (8h)")
            return 4
    log("清场+上游终态, 启动 main 特征集 cls 对齐复审")
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
