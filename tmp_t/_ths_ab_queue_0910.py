# -*- coding: utf-8 -*-
"""THS 看涨池特征 A/B 排队 wrapper (09-10): 等今晚干净重训 + 23:30 四模块夜链全部
完成后跑 _ths_ab_minibacktest_v1_0910.py (用户指令: retrain and predict FIRST, A/B 随后).

判据双条件: 夜链 state 文件终态 (status != running) + run-guard 冲突清空;
23:35 起轮询 (确保 23:30 链已被调度器拉起); 链没跑的兜底: 00:35 后仅凭冲突清空放行;
07:35 仍未清场 → 放弃留日志. rc=3 (RAM guard) 时等 30min 重试, 最多到 deadline.
Start-Process 独立进程启动 (长跑惯例), 日志 tmp_t/_ths_ab_queue_0910.log.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")

DT0 = dt.date.today()
LOG = open("tmp_t/_ths_ab_queue_0910.log", "a", encoding="utf-8")


def log(msg):
    print(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}", file=LOG, flush=True)


def conflicts():
    from scripts._run_guard import find_conflicts

    return find_conflicts()


def chain_state():
    p = os.path.join("logs", f"daily_automation_{DT0:%Y%m%d}.state.json")
    if not os.path.exists(p):
        return "absent"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("status", "?")
    except Exception:
        return "?"


wake = dt.datetime.now().replace(hour=23, minute=35, second=0, microsecond=0)
if dt.datetime.now() < wake:
    log(f"睡到 {wake} 等干净重训 (17:00 起) + 23:30 夜链全部完成")
    time.sleep((wake - dt.datetime.now()).total_seconds())
deadline = wake + dt.timedelta(hours=8)
absent_ok_after = wake + dt.timedelta(hours=1)
attempts = 0
while True:
    c = conflicts()
    st = chain_state()
    chain_terminal = st not in ("absent", "?", "running")
    absent_ok = st == "absent" and dt.datetime.now() >= absent_ok_after
    if not c and (chain_terminal or absent_ok):
        log(f"夜链={st} 冲突清空, 启动 A/B")
        break
    if dt.datetime.now() >= deadline:
        log(f"{deadline:%m-%d %H:%M} 仍未清场 (链={st}), 放弃 (下次会话重跑)")
        sys.exit(3)
    attempts += 1
    log(f"等待 10min (#{attempts}): 链={st} 冲突={[x['sentinel'] for x in c]}")
    time.sleep(600)

log("清场, 启动 THS A/B (main)")
rc = None
while True:
    r = subprocess.run([sys.executable, "-X", "utf8",
                        "tmp_t/_ths_ab_minibacktest_v1_0910.py", "--board", "main"])
    rc = r.returncode
    if rc != 3:
        break
    if dt.datetime.now() >= deadline:
        log("RAM guard 反复不过, 放弃")
        sys.exit(3)
    log("rc=3 (RAM<5GB), 等 30min 重试")
    time.sleep(1800)
log(f"A/B exit={rc}")
sys.exit(rc)
