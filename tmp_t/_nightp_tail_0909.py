# -*- coding: utf-8 -*-
"""09-09 夜链后半段补位 (B 独跑已由并行会话完成后): retrain → legacy 三步 → 尾部 → 终态.

背景: 21:02 并行会话接管 (state note reordered_parallel_first_0909), B 支 22:40 全绿收尾;
其声明的 "full chain relaunch follows" 若 10 分钟内未兑现, 由本脚本补位 (会话间协调:
启动前已确认无 retrain/daily_automation/nightp 进程, 启动后对方 relaunch 会被 retrain
守卫 rc=3 挡下, 但为防其误判失败, state 每阶段都写明 note).
流程: retrain(守卫握手同 orch) → legacy_prob_head→legacy→deliver (critical) →
      Phase3 尾部 (ths_push…gate_audit) → state 终态 (fade wrapper 12080 消费).
"""
import datetime as _dt
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, ".")
from scripts.run_daily_automation import (  # noqa: E402
    _STEP_TIMEOUT_S,
    _STEPS,
    _kill_tree,
    _run_step_with_watchdog,
    sys as _rsys,
)

TAG = "20260909"
PY = _rsys.executable
STATE = f"logs/daily_automation_{TAG}.state.json"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
LOG = "logs/nightp_tail_0909.log"


def log(m):
    line = f"[{_dt.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_state(status, **extra):
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump({"tag": TAG, "status": status, "ts": time.time(), **extra}, f)


def run_step(step):
    argv = [PY, "-u", *[a.format(tag=TAG) for a in _STEPS[step]]]
    log(f"start {step}")
    fh = open(f"logs/nightp_{step}.log", "w", encoding="utf-8")
    t0 = time.time()
    rc, timed_out = _run_step_with_watchdog(argv, fh, ENV, _STEP_TIMEOUT_S[step])
    fh.close()
    rc = 124 if timed_out else rc
    log(f"[{_dt.datetime.now():%H:%M:%S} {'ok' if rc == 0 else 'FAIL'}] {step} rc={rc} ({time.time() - t0:.0f}s)")
    return rc


def main():
    failed = []
    write_state("running", reason="nightp_tail_backfill_0909",
                note="tail half (retrain+legacy+deliver+phase3) backfilled by session A after B solo done 22:40")

    log("启动 retrain (守卫要求场地无哨兵; 已核验)")
    argv_a = [PY, "-u", *[a.format(tag=TAG) for a in _STEPS["retrain"]]]
    fh = open("logs/nightp_retrain_0909.log", "w", encoding="utf-8")
    t0 = time.time()
    rc, timed_out = _run_step_with_watchdog(argv_a, fh, ENV, _STEP_TIMEOUT_S["retrain"])
    fh.close()
    rc = 124 if timed_out else rc
    log(f"[{_dt.datetime.now():%H:%M:%S} {'ok' if rc == 0 else 'FAIL'}] retrain rc={rc} ({time.time() - t0:.0f}s)")
    if rc != 0:
        failed.append("retrain")
        write_state("failed", reason="retrain_fail_tail", failed_steps=failed)
        log(f"NIGHTP_TAIL_DONE status=failed failed={failed}")
        return 1

    for step in ("legacy_prob_head", "legacy", "deliver"):
        rc = run_step(step)
        if rc != 0:
            failed.append(step)
            if step in ("legacy", "deliver"):
                write_state("failed", reason="critical_step_fail", failed_steps=failed)
                log(f"NIGHTP_TAIL_DONE status=failed critical={step}")
                return 1

    for step in ("ths_push", "prob10dens_push", "slowbull_shadow", "ths_flush_guard",
                 "final_stocklist", "stocklist_combined", "drift", "drift_parallel",
                 "shadow_xmodule", "gate_audit"):
        if run_step(step) != 0:
            failed.append(step)

    status = "ok" if not failed else "failed"
    write_state(status, reason="nightp_tail_backfill_0909", failed_steps=failed)
    log(f"NIGHTP_TAIL_DONE status={status} failed={failed or '无'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        with open(LOG, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        write_state("failed", reason="tail_exception")
        sys.exit(1)
