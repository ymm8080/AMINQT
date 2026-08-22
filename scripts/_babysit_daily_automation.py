"""fail-fast 监督 daily_automation — 终态 (ok/failed/interrupted) 一到立刻退出.

替代 _watch_daily_automation.py 的无限 tail 循环. 那种循环在自动化崩溃后永不
退出, 挂着的监督进程会一直耗 token (08-21 事故). 本脚本以
logs/daily_automation_<tag>.state.json 为唯一终态判据 (run_daily_automation.py
在每个退出路径都写终态: 成功/失败/Ctrl+C/0xC013A), 状态文件出现即退出;
可选 --timeout 兜底. 不重启自动化 (崩溃就是结束, RestartCount=0).

用法:
  python scripts/_babysit_daily_automation.py --tag 20260821
  python scripts/_babysit_daily_automation.py --tag 20260821 --timeout 3600
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

_EXIT_OK = 0
_EXIT_FAILED = 1
_EXIT_INTERRUPTED = 130
_EXIT_TIMEOUT = 124
_EXIT_CRASHED = 3


def _state_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"daily_automation_{tag}.state.json")


def _read_state(tag: str) -> dict | None:
    try:
        with open(_state_path(tag), encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _exit_code_for(status: str) -> int:
    """终态 status → 退出码 (未知/未终态 → crash 码, 绝不当成功)."""
    return {
        "ok": _EXIT_OK,
        "failed": _EXIT_FAILED,
        "interrupted": _EXIT_INTERRUPTED,
    }.get(status, _EXIT_CRASHED)


def main() -> int:
    ap = argparse.ArgumentParser(description="fail-fast 监督 daily_automation")
    ap.add_argument("--tag", default=None, help="清单交易日 YYYYMMDD (默认今天)")
    ap.add_argument("--poll", type=float, default=30.0, help="轮询间隔秒 (默认 30)")
    ap.add_argument("--timeout", type=float, default=0.0, help="最大等待秒, 0=不限")
    args = ap.parse_args()

    tag = args.tag or _dt.date.today().strftime("%Y%m%d")
    t0 = time.time()
    print(f"[babysit] tag={tag} 等待终态 (state 文件出现即退出)", flush=True)
    while True:
        state = _read_state(tag)
        if state is not None and state.get("status") != "running":
            status = state.get("status")
            print(
                f"[babysit] 终态: {status} (tag={tag}, 用时 {time.time() - t0:.0f}s) → 退出",
                flush=True,
            )
            return _exit_code_for(status)
        if args.timeout > 0 and time.time() - t0 >= args.timeout:
            print(
                f"[babysit] 超时 {args.timeout:.0f}s 未见终态 → 退出 {_EXIT_TIMEOUT}",
                flush=True,
            )
            return _EXIT_TIMEOUT
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
