# -*- coding: utf-8 -*-
"""v5 接力链 babysitter: 等 v4 dual (PID 参数) 退出 → RAM guard → 依次跑 v5 main / v5 dual.

状态文件: tmp_t/_regcal_v5_chain_state.json (每次阶段切换即落盘).
fail-fast: 任一 v5 段 rc!=0 即停链, 状态记 failed.
"""

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "tmp_t", "_regcal_minibacktest_0910.py")
STATE = os.path.join(ROOT, "tmp_t", "_regcal_v5_chain_state.json")
LOGDIR = os.path.join(ROOT, "tmp_t")


def pid_alive(pid):
    import psutil

    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def set_state(stage, **kw):
    st = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, **kw}
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, ensure_ascii=False, indent=2)
    print(f"[chain] {st}", flush=True)


def main():
    wait_pid = int(sys.argv[1])
    t0 = time.time()
    set_state("waiting_v4_dual", wait_pid=wait_pid)
    while pid_alive(wait_pid):
        time.sleep(30)
    set_state("v4_dual_exited", waited_s=int(time.time() - t0))

    import psutil

    free_gb = psutil.virtual_memory().available / 1024**3
    if free_gb < 5.0:
        set_state("aborted_ram", free_gb=round(free_gb, 2))
        return 3

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for board in ("main", "dual"):
        set_state(f"v5_{board}_running")
        out = open(os.path.join(LOGDIR, f"_regcal_v5_{board}_run1.out"), "w", encoding="utf-8")
        err = open(os.path.join(LOGDIR, f"_regcal_v5_{board}_run1.err"), "w", encoding="utf-8")
        rc = subprocess.run(
            [sys.executable, SCRIPT, "--board", board],
            cwd=ROOT, stdout=out, stderr=err, env=env,
        ).returncode
        out.close()
        err.close()
        if rc != 0:
            set_state(f"v5_{board}_failed", rc=rc)
            return rc
        set_state(f"v5_{board}_done")
    set_state("all_done", total_s=int(time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
