# -*- coding: utf-8 -*-
"""_night_queue_0910.py (v2) — 今夜监督队列: 重训/预测/概率头A/B 带冲突重触发.

用户指令 (2026-09-10 夜): "FIX THE ISSUE, MAKE SURE IF THERE IS CONFLICT,
THERE WILL BE RETRIGGER OF RETRAIN AND PREDICTION"。

设计:
  每步 = wait_slot (CHAIN_SENTINELS 无冲突 且 RAM>=8GB) → 跑 → rc!=0 则
  冷却 (10min 起, 每次翻倍, 封顶 60min) 后回到 wait_slot 重试。
  ① 重训   scripts/_retrain_legacy_full.py 20260910   最多 8 次尝试
  ② 预测链 run_daily_automation.py --skip-retrain --skip-parallel --force
     (与今晚 20:15 链同一命令, 重训晋升后用新模型重出清单) 最多 8 次
  ③ 概率头 A/B (独立于①②结果, 最多 3 次)
  全部 rc + 尝试数落盘 tmp_t/_night_queue_0910_state.json (终态一次性写)。
  日志 WORM: 每次尝试独立文件 _retrain_20260910_rerun2_a{N}.out / _pred_a{N}.out。
取代 v1 (无重试, 一步失败即死)。
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts._run_guard import CHAIN_SENTINELS, find_conflicts  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
HARNESS = os.path.join(HERE, "_pin_brute_prob_ab_0910.py")
STATE = os.path.join(HERE, "_night_queue_0910_state.json")
POLL_SEC = 180
FREE_MIN_GB = 8.0
MAX_ATTEMPTS_MAIN = 8
MAX_ATTEMPTS_AB = 3
COOLDOWN_BASE = 600
COOLDOWN_CAP = 3600

RETRAIN_CMD = [sys.executable, os.path.join(ROOT, "scripts", "_retrain_legacy_full.py"), "20260910"]
PREDICT_CMD = [
    sys.executable,
    os.path.join(ROOT, "scripts", "run_daily_automation.py"),
    "--skip-retrain",
    "--skip-parallel",
    "--force",
]


def wait_slot(tag: str) -> None:
    while True:
        conflicts = find_conflicts(CHAIN_SENTINELS)
        import psutil

        free_gb = psutil.virtual_memory().available / 1024**3
        first = conflicts[0]["sentinel"] if conflicts else "-"
        print(
            f"[queue:{tag}] {datetime.now():%H:%M:%S} conflicts={len(conflicts)} "
            f"first={first} free={free_gb:.1f}GB",
            flush=True,
        )
        if not conflicts and free_gb >= FREE_MIN_GB:
            return
        time.sleep(POLL_SEC)


def run_with_retry(tag: str, cmd: list[str], max_attempts: int, log_prefix: str) -> tuple[int, int]:
    """返回 (最终 rc, 用的尝试数); rc==0 = 成功."""
    cooldown = COOLDOWN_BASE
    for attempt in range(1, max_attempts + 1):
        wait_slot(f"{tag}:a{attempt}")
        out_log = os.path.join(HERE, f"{log_prefix}_a{attempt}.out")
        err_log = os.path.join(HERE, f"{log_prefix}_a{attempt}.err")
        print(f"[queue:{tag}] 尝试 {attempt}/{max_attempts} -> {os.path.basename(out_log)}", flush=True)
        out_fh = open(out_log, "w", encoding="utf-8")
        err_fh = open(err_log, "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out_fh, stderr=err_fh)
        print(f"[queue:{tag}] PID={proc.pid}", flush=True)
        rc = proc.wait()
        out_fh.close()
        err_fh.close()
        print(f"[queue:{tag}] 尝试 {attempt} rc={rc}", flush=True)
        if rc == 0:
            return 0, attempt
        if attempt < max_attempts:
            print(f"[queue:{tag}] 冷却 {cooldown}s 后重触发", flush=True)
            time.sleep(cooldown)
            cooldown = min(cooldown * 2, COOLDOWN_CAP)
    return 1, max_attempts


def main() -> int:
    rc_retrain, n_ret = run_with_retry("retrain", RETRAIN_CMD, MAX_ATTEMPTS_MAIN, "_retrain_20260910_rerun2")
    rc_pred, n_pred = (None, 0)
    if rc_retrain == 0:
        rc_pred, n_pred = run_with_retry("predict", PREDICT_CMD, MAX_ATTEMPTS_MAIN, "_pred_20260910")
    else:
        print("[queue:predict] 跳过 (重训 8 次全败)", flush=True)

    rc_ab, n_ab = run_with_retry("probab", [sys.executable, HARNESS], MAX_ATTEMPTS_AB, "_probab_20260910")

    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "retrain_rc": rc_retrain,
                "retrain_attempts": n_ret,
                "predict_rc": rc_pred,
                "predict_attempts": n_pred,
                "prob_ab_rc": rc_ab,
                "prob_ab_attempts": n_ab,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[queue] 终态 retrain={rc_retrain}({n_ret}) predict={rc_pred}({n_pred}) probab={rc_ab}({n_ab})", flush=True)
    return 0 if rc_retrain == 0 and rc_pred == 0 and rc_ab == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
