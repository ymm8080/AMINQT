# -*- coding: utf-8 -*-
"""看板常驻守护进程 — 崩溃自动重启, 日志落盘.

用法:
  pythonw scripts\\dashboard_watchdog.py   # 无窗口后台运行(推荐)
  python scripts\\dashboard_watchdog.py    # 前台调试
停止: 运行仓库根目录的 停止看板.bat
"""
from __future__ import annotations

import datetime
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "dashboard.log")
PORT = 8501
LOCK_PORT = 8765  # 单实例锁: 已在跑的守护进程会占用它

# pythonw 无控制台时 stdout/stderr 为 None, 直接 print 会崩溃, 先兜底
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{_now()}] {msg}\n")
    print(f"[{_now()}] {msg}", flush=True)


def dashboard_alive() -> bool:
    """看板是否已在响应 (用于避免重复起一份)."""
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=2):
            return True
    except OSError:
        return False


def main() -> None:
    # 单实例锁: 若锁端口已被占, 说明已有守护进程在跑, 本实例直接退出
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # 注意: 不能设 SO_REUSEADDR — Windows 上它会让第二次 bind 同端口成功, 破坏单实例锁
        lock.bind(("127.0.0.1", LOCK_PORT))
        lock.listen(1)
    except OSError:
        # 已有守护进程在跑 (含定时心跳重复触发), 静默退出即可
        return

    os.chdir(ROOT)
    log(f"守护进程启动, 工作目录={ROOT}")
    while True:
        if dashboard_alive():
            # 看板已由别处(如手动 bat)拉起, 无需重复启动, 定期复查
            time.sleep(30)
            continue
        log("启动 streamlit ...")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
            rc = proc.wait()
        log(f"streamlit 退出 rc={rc}, 5 秒后自动重启")
        time.sleep(5)


if __name__ == "__main__":
    main()
