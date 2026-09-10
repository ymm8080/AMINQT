# -*- coding: utf-8 -*-
"""_pin_brute_prob_ab_wait_0910.py — 概率头 A/B 排队包装器 (轻量轮询, 非重活).

放行条件 (全部满足才启动 harness):
  1. find_conflicts() 空 (重训/链/其他重活全部结束);
  2. 今日 legacy 清单已交付 (legacy_stocklist_20260910*__*.csv 存在) 或已过 23:00。
防饿死: 23:00 后只要无冲突就放行 (链若死了不让 A/B 永远等)。
状态落盘: tmp_t/_pin_brute_prob_ab_state_0910.json (WORM 一次性写终态)。
"""

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts._run_guard import find_conflicts  # noqa: E402

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pin_brute_prob_ab_0910.py")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pin_brute_prob_ab_state_0910.json")
TODAY = "20260910"
POLL_SEC = 300


def _delivered() -> bool:
    from config.settings import STOCK_LIST_DIR

    hits = glob.glob(os.path.join(str(STOCK_LIST_DIR), f"legacy_stocklist_{TODAY}*__*.csv"))
    return bool(hits)


def main() -> int:
    while True:
        conflicts = find_conflicts()
        delivered = _delivered()
        hour = datetime.now().hour
        status = (
            f"[wait] {datetime.now():%H:%M:%S} conflicts={len(conflicts)} "
            f"delivered={delivered} hour={hour}"
        )
        if conflicts:
            status += f" first={conflicts[0]['sentinel']}"
        print(status, flush=True)
        if not conflicts and (delivered or hour >= 23):
            break
        time.sleep(POLL_SEC)

    print(f"[run] 启动概率头 A/B: {HARNESS}", flush=True)
    rc = subprocess.run([sys.executable, HARNESS]).returncode
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "harness_rc": rc,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[done] harness rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
