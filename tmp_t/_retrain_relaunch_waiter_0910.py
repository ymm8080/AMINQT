# -*- coding: utf-8 -*-
"""_retrain_relaunch_waiter_0910.py — 16:26 重训被 OOM 击杀后的自动重启器.

死因 (2026-09-10): 启动时机器仅剩 0.3G (并行重活挤兑), 重训自身涨到 ~9G 后
commit 耗尽, 16:57 后被 Windows 无声击杀 (无 traceback 无 verdict)。
本包装器等待: _daily_fetch.py 进程退出 且 空闲 RAM >= 8GB → 重启重训
(WORM 新日志 tmp_t/_retrain_20260910_rerun.out/.err, 不覆盖 16:26 原日志)。
子进程退出码落盘 tmp_t/_retrain_relaunch_state_0910.json。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_LOG = os.path.join(HERE, "_retrain_20260910_rerun.out")
ERR_LOG = os.path.join(HERE, "_retrain_20260910_rerun.err")
STATE = os.path.join(HERE, "_retrain_relaunch_state_0910.json")
POLL_SEC = 120
FREE_MIN_GB = 8.0
FETCH_TAG = "_daily_fetch.py"


def _fetch_alive() -> bool:
    import psutil

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if "python" not in (p.info["name"] or "").lower():
            continue
        if FETCH_TAG in " ".join(p.info["cmdline"] or []):
            return True
    return False


def main() -> int:
    while True:
        alive = _fetch_alive()
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        print(
            f"[relaunch-wait] {datetime.now():%H:%M:%S} daily_fetch_alive={alive} "
            f"free={free_gb:.1f}GB (need {FREE_MIN_GB:.0f}GB)",
            flush=True,
        )
        if not alive and free_gb >= FREE_MIN_GB:
            break
        time.sleep(POLL_SEC)

    print(f"[relaunch] 启动重训: scripts/_retrain_legacy_full.py 20260910", flush=True)
    out_fh = open(OUT_LOG, "w", encoding="utf-8")
    err_fh = open(ERR_LOG, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "..", "scripts", "_retrain_legacy_full.py"), "20260910"],
        cwd=os.path.join(HERE, ".."),
        stdout=out_fh,
        stderr=err_fh,
    )
    print(f"[relaunch] retrain PID={proc.pid} out={OUT_LOG}", flush=True)
    rc = proc.wait()
    out_fh.close()
    err_fh.close()
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "retrain_pid": proc.pid,
                "rc": rc,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[relaunch] retrain rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
