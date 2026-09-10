# -*- coding: utf-8 -*-
"""fade pin A/B 排队 wrapper (09-09): 等 ohlc/cyq/ovd 三修复清场后跑 _pin_ab_fade_0909.py.
用户指令 (20:xx 修订): A/B 放今天 20:15 重训 + 23:30 四模块预测链全部完成之后.
判据双条件: 夜链 state 文件终态 (status != running) + run-guard 冲突清空;
23:35 起轮询 (确保 23:30 链已被调度器拉起); 链没跑的兜底: 00:35 后仅凭冲突清空放行;
07:35 仍未清场 → 放弃留日志. Start-Process 独立进程启动 (长跑惯例),
日志 tmp_t/_pin_ab_fade_queue_0909.log.
"""
import datetime as dt
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")

LOG = open("tmp_t/_pin_ab_fade_queue_0909.log", "a", encoding="utf-8")


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=LOG, flush=True)


def conflicts():
    from scripts._run_guard import find_conflicts

    return find_conflicts()


def chain_state():
    p = os.path.join("logs", f"daily_automation_{dt.date.today():%Y%m%d}.state.json")
    if not os.path.exists(p):
        return "absent"
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("status", "?")
    except Exception:
        return "?"


wake = dt.datetime.now().replace(hour=23, minute=35, second=0, microsecond=0)
if dt.datetime.now() < wake:
    log(f"用户指令: 睡到 {wake} 等 20:15 重训 + 23:30 夜链全部完成")
    time.sleep((wake - dt.datetime.now()).total_seconds())
deadline2 = wake + dt.timedelta(hours=8)
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
    if dt.datetime.now() >= deadline2:
        log(f"{deadline2:%m-%d %H:%M} 仍未清场 (链={st}), 放弃 (下次会话重跑)")
        sys.exit(3)
    attempts += 1
    log(f"等待 10min (#{attempts}): 链={st} 冲突={[x['sentinel'] for x in c]}")
    time.sleep(600)

log("清场, 启动 A/B")
r = subprocess.run([sys.executable, "-X", "utf8", "tmp_t/_pin_ab_fade_0909.py"])
log(f"A/B exit={r.returncode}")

# mech_pilot 只读诊断排到重训后 (用户 09-09 指令): A/B 结束后再等清场跑,
# 单进程串行避免与重训/夜链并发 (mech_pilot 自身守卫只等 ohlc/ovd/cyq 前驱)
for _ in range(60):
    c = conflicts()
    if not c:
        break
    log(f"mech_pilot 前冲突等待 10min: {[x['sentinel'] for x in c]}")
    time.sleep(600)
c = conflicts()
if c:
    log(f"mech_pilot 放弃 (清场超时): {[x['sentinel'] for x in c]}")
else:
    log("清场, 启动 mech_pilot")
    r2 = subprocess.run([sys.executable, "-X", "utf8", "tmp_t/_mech_pilot_0909.py"])
    log(f"mech_pilot exit={r2.returncode}")
sys.exit(r.returncode)
