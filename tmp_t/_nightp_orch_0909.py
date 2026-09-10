# -*- coding: utf-8 -*-
"""09-09 夜并行编排 (用户指令: 四模块重训+预测并行跑, 过载则串行, 最快出四模块清单).

拓扑 (取代被杀的串行链 3792 与 20:15 等待器 1120):
  Phase0 preflight 串行: cyq→sw_history→freshness→canary (与链全同)
  Phase1 并行: A=retrain (先启动, 其 rc=3 型守卫要求启动时场地干净)
               +120s 握手 (A 活且日志无 [guard]) 后拉 B=parallel 支 worker
               RAM 监督: psutil available<1GB×3 连击 → 杀 B (A 永不被杀), B 转 A 后串行重跑
  Phase2 汇合 (A、B 均终): legacy_prob_head→legacy→deliver (legacy/deliver 失败即止=链 critical 语义)
  Phase3 尾部: ths_push→prob10dens_push→slowbull_shadow→ths_flush_guard→
               final_stocklist→stocklist_combined→drift→drift_parallel→shadow_xmodule→gate_audit
  state 写 logs/daily_automation_20260909.state.json (fade A/B wrapper 12080 轮询此文件终态).
守卫零绕过: 本编排器与 B worker 均不在 HEAVY_SENTINELS; legacy 带守卫故置双支汇合后;
  A 启动时唯一在场 python 重活 = 无 (phase0 步骤非哨兵).
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
RAM_MIN_GB = 1.0
RAM_STRIKES = 3
POLL_S = 30
B_WORKER = "tmp_t/_nightp_branchb_0909.py"


def log(m):
    line = f"[{_dt.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    with open("logs/nightp_orch_0909.log", "a", encoding="utf-8") as f:
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


def popen_logged(argv, logfile):
    fh = open(logfile, "w", encoding="utf-8")
    proc = subprocess.Popen(
        argv, cwd=".", stdout=fh, stderr=subprocess.STDOUT, env=ENV,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    return proc, fh


def main():
    import psutil

    write_state("running", reason="night_parallel_manual_0909")
    failed = []

    # Phase 0: preflight (与链全同; 全部非 fatal — 链语义)
    # NIGHTP_SKIP_PHASE0=1: 跳过 (今晚已全部通过的续跑场景, canary 30min 是大头)
    if os.environ.get("NIGHTP_SKIP_PHASE0", "0") != "1":
        for step in ("cyq", "sw_history", "freshness", "canary"):
            if run_step(step) != 0:
                failed.append(step)
    else:
        log("Phase0 跳过 (NIGHTP_SKIP_PHASE0=1)")

    # Phase 1A: retrain 先启动 (守卫要求启动瞬间场地无哨兵)
    log("Phase1: A=retrain 启动 (先于 B, 过闸时场地干净)")
    argv_a = [PY, "-u", *[a.format(tag=TAG) for a in _STEPS["retrain"]]]
    a_log = "logs/nightp_retrain_0909.log"
    a, a_fh = popen_logged(argv_a, a_log)

    def _a_watchdog():
        time.sleep(_STEP_TIMEOUT_S["retrain"])
        if a.poll() is None:
            log(f"retrain 超时 {_STEP_TIMEOUT_S['retrain'] // 3600}h → 杀树")
            _kill_tree(a.pid)

    threading.Thread(target=_a_watchdog, daemon=True).start()

    # Phase 1B: +120s 握手后拉 B (A 的守卫在 import 后 main 开头, 120s 足够过闸)
    time.sleep(120)
    if a.poll() is not None:
        log(f"A 在握手期内退出 rc={a.returncode} (疑守卫/ram_guard) — 查 {a_log}")
        with open(a_log, encoding="utf-8", errors="replace") as f:
            for ln in f:
                if "[guard]" in ln or "ram_guard" in ln.lower() or "Error" in ln:
                    log(f"A-log: {ln.strip()}")
        write_state("failed", reason="retrain_guard_abort", failed_steps=["retrain"])
        return 1
    with open(a_log, encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    if "[guard]" in head:
        log("A 日志出现 [guard] — 场地非净, 中止编排")
        write_state("failed", reason="retrain_guard_abort", failed_steps=["retrain"])
        _kill_tree(a.pid)
        return 1

    log("Phase1: B=parallel 支启动 (refresh→parallel→prob_head→deliver_parallel)")
    argv_b = [PY, "-X", "utf8", B_WORKER]
    b, b_fh = popen_logged(argv_b, "logs/nightp_branchb_0909.log")

    # RAM 监督: 双活且 available<1GB×3 → 杀 B 保 A (用户条款: 过载转串行)
    strikes = 0
    polls = 0
    b_killed = False
    while True:
        a_alive = a.poll() is None
        b_alive = b.poll() is None
        if not a_alive and not b_alive:
            break
        avail = psutil.virtual_memory().available / 2**30
        polls += 1
        if polls % 8 == 0:
            log(f"RAM avail={avail:.2f}GB A={'活' if a_alive else '终'} B={'活' if b_alive else '终'}")
        if a_alive and b_alive and avail < RAM_MIN_GB:
            strikes += 1
            log(f"RAM 低 {avail:.2f}GB strike {strikes}/{RAM_STRIKES}")
            if strikes >= RAM_STRIKES:
                log(f"OVERBURDEN: avail={avail:.2f}GB → 杀 B 支保 A, B 转 A 后串行重跑")
                _kill_tree(b.pid)
                b_killed = True
                strikes = 0
        else:
            strikes = 0
        time.sleep(POLL_S)

    a_rc = a.returncode
    log(f"A=retrain 终 rc={a_rc}")
    if a_rc != 0:
        failed.append("retrain")
    a_fh.close()

    # B 终态/重跑 (被杀或并发期失败 → A 后串行重跑一次)
    b_rc = b.returncode
    if b_killed or b_rc != 0:
        why = "被杀(过载)" if b_killed else f"rc={b_rc}"
        log(f"B 支并发期未完成 ({why}) → A 后串行重跑")
        b_fh.close()
        r = subprocess.run([PY, "-X", "utf8", B_WORKER], cwd=".")
        b_rc = r.returncode
    else:
        b_fh.close()
        log("B 支并发完成")
    if b_rc != 0:
        failed.append("parallel_branch")

    # Phase 2: legacy 出清单 (critical: legacy/deliver 失败即止)
    for step in ("legacy_prob_head", "legacy", "deliver"):
        rc = run_step(step)
        if rc != 0:
            failed.append(step)
            if step in ("legacy", "deliver"):
                write_state("failed", reason="critical_step_fail", failed_steps=failed)
                log(f"NIGHTP_DONE status=failed critical={step}")
                return 1

    # Phase 3: 尾部交付/审计 (全非 fatal)
    for step in ("ths_push", "prob10dens_push", "slowbull_shadow", "ths_flush_guard",
                 "final_stocklist", "stocklist_combined", "drift", "drift_parallel",
                 "shadow_xmodule", "gate_audit"):
        if run_step(step) != 0:
            failed.append(step)

    status = "ok" if not failed else "failed"
    write_state(status, reason="night_parallel_manual_0909", failed_steps=failed)
    log(f"NIGHTP_DONE status={status} failed={failed or '无'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        with open("logs/nightp_orch_0909.log", "a", encoding="utf-8") as f:
            f.write(traceback.format_exc() + "\n")
        write_state("failed", reason="orchestrator_exception")
        sys.exit(1)
