"""等 slow_bull 并行任务结束后再启动 Tiered TopN A/B 验证.

本机 15.8GB 物理内存, slow_bull 并行任务占用 ~9GB, 与验证任务并发会 OOM.
轮询探测 slow_bull (cmdline 含 slow_bull 的 python) 退出后, 等 30s 让内存回落,
再启动 scripts/_diag_tiered_topn_ab.py (后台, 输出到 data/_diag_tiered_topn_ab_run.log).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def slowbull_alive() -> bool:
    import psutil

    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            cl = " ".join(p.info["cmdline"] or [])
            if "python" in name and "slow_bull" in cl:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False


def main() -> None:
    t0 = time.time()
    print(f"[launcher] 等待 slow_bull 结束 (每 60s 探测) ...", flush=True)
    while slowbull_alive():
        time.sleep(60)
    print(
        f"[launcher] slow_bull 已结束 (等了 {time.time() - t0:.0f}s), 30s 后启动 A/B ...",
        flush=True,
    )
    time.sleep(30)

    log = os.path.join(ROOT, "data", "_diag_tiered_topn_ab_run.log")
    cmd = [
        sys.executable,
        os.path.join("scripts", "_diag_tiered_topn_ab.py"),
        "--window-days",
        "380",
        "--oos-days",
        "120",
    ]
    with open(log, "w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    print(f"[launcher] A/B 结束 exit={proc.returncode}, 日志={log}", flush=True)


if __name__ == "__main__":
    main()
