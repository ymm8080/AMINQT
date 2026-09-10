# -*- coding: utf-8 -*-
"""SEP09 夜链手动排队器 (2026-09-09): cyq 重算结束后启动全链重训+预测.

用户指令: "CYQ 结束后先跑 SEP09的重训和预测。四个模型" + "还要跑 密度模型和SLOW BULL"
→ run_daily_automation 全链 (legacy main+dual / parallel sniper+fusion / prob头 /
  prob10dens 密度 / slowbull_shadow 全在链内); --force-retrain 跨周五闸 (今天周三).
今夜 2330/2030 计划任务均已禁用 → 本实例 = 今夜唯一链.
fade 队列 (12080) 按用户指令睡到 23:35 且自带全量哨兵等待; 链重活步骤预计
~23:00 前完 → 无相撞; 就算超时 fade 队列会自己等, 无死锁面.

协议: PRED 只含 cyq+自身名, **每一次** find_conflicts 都带 sentinels=PRED;
本脚本**不注册** HEAVY_SENTINELS — 它是链的父进程, 注册会让链内子步骤
(rc=3 即退型守卫) 反查时看见父进程, 整链被自己的父锁死.
终态: CHAIN_SEP09_DONE status=ok|fail|aborted + tmp_t/_chain_sep09_result.json
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "tmp_t", "_chain_sep09_0909.log")
RESULT = os.path.join(ROOT, "tmp_t", "_chain_sep09_result.json")
CHAIN = os.path.join(ROOT, "scripts", "run_daily_automation.py")
CYQ_JSON = os.path.join(ROOT, "tmp_t", "_cyq_recompute_result.json")
PRED = ("_cyq_recompute_v3_0909.py", "_chain_sep09_0909.py")

_lf = open(LOG, "a", encoding="utf-8")


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    _lf.write(line + "\n")
    _lf.flush()


def main():
    from scripts._run_guard import find_conflicts

    os.chdir(ROOT)
    conflicts = find_conflicts(sentinels=PRED)
    attempt = 0
    while conflicts and attempt < 90:
        log(f"等前驱 ({attempt + 1}/90): {[c.get('sentinel') for c in conflicts]}")
        time.sleep(600)
        attempt += 1
        conflicts = find_conflicts(sentinels=PRED)
    if conflicts:
        log("CHAIN_SEP09_DONE status=aborted reason=pred_still_alive")
        return 2

    try:
        with open(CYQ_JSON, encoding="utf-8") as f:
            cyq_status = json.load(f).get("status", "?")
    except Exception:
        cyq_status = "no-json (verify/cache 段 aborted? 面板原子替换未受损)"
    log(f"前驱清空; cyq 终态={cyq_status}; 启动夜链 --force-retrain --force")

    t0 = datetime.now()
    _lf.write(f"\n===== chain start {t0:%F %T} (cyq={cyq_status}) =====\n")
    _lf.flush()
    p = subprocess.Popen(
        [sys.executable, "-X", "utf8", CHAIN, "--force-retrain", "--force"],
        cwd=ROOT,
        stdout=_lf,
        stderr=subprocess.STDOUT,
    )
    rc = p.wait()
    status = "ok" if rc == 0 else "fail"
    out = {
        "status": status,
        "chain_rc": int(rc),
        "cyq_status": cyq_status,
        "started": t0.isoformat(timespec="seconds"),
        "finished": datetime.now().isoformat(timespec="seconds"),
        "note": "SEP09 手动夜链 (2330 任务禁用, 本实例=今夜唯一); "
                "四模型 + 密度 prob10dens + slowbull 全链",
    }
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"CHAIN_SEP09_DONE status={status} chain_rc={rc} -> {RESULT}")
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        _lf.write(traceback.format_exc() + "\n")
        _lf.flush()
        log("CHAIN_SEP09_DONE status=aborted")
        sys.exit(1)
