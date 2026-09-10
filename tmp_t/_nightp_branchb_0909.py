# -*- coding: utf-8 -*-
"""parallel 支独立 worker (09-09 夜并行编排 B 支): refresh→parallel→prob_head→deliver_parallel.
由 _nightp_orch_0909.py 拉起; 过载时被整支杀掉、A(retrain) 结束后串行重拉一次.
自身不查守卫 (这四步脚本本就不嵌脚本级守卫; retrain 的 rc=3 型守卫只保护先启动的 A).
日志 logs/nightp_branchb_0909.log, 步输出 logs/nightp_b_<step>.log.
"""
import datetime as _dt
import os
import sys
import time

sys.path.insert(0, ".")
from scripts.run_daily_automation import (  # noqa: E402
    _STEP_TIMEOUT_S,
    _STEPS,
    _run_step_with_watchdog,
    sys as _rsys,
)

TAG = "20260909"
STEPS = ["refresh", "parallel", "prob_head", "deliver_parallel"]
PY = _rsys.executable


def log(m):
    line = f"[{_dt.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open("logs/nightp_branchb_0909.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for step in STEPS:
        argv = [PY, "-u", *[a.format(tag=TAG) for a in _STEPS[step]]]
        log(f"B start {step}")
        fh = open(f"logs/nightp_b_{step}.log", "w", encoding="utf-8")
        t0 = time.time()
        rc, timed_out = _run_step_with_watchdog(argv, fh, env, _STEP_TIMEOUT_S[step])
        fh.close()
        rc = 124 if timed_out else rc
        log(f"[{_dt.datetime.now():%H:%M:%S} {'ok' if rc == 0 else 'FAIL'}] B {step} rc={rc} ({time.time() - t0:.0f}s)")
        if rc != 0:
            log(f"BRANCH_B_DONE rc={rc} failed_at={step}")
            return rc
    log("BRANCH_B_DONE rc=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
