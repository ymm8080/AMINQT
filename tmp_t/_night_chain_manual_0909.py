# -*- coding: utf-8 -*-
"""今晚一次性手动夜链 (09-09): 两个调度任务 (Retrain-2030 / DailyAutomation-2330) 自
昨日 user_stop 后处于 Disabled, 不会自行触发. 用户 09-09 指令 "20:15 FADE重训,
A/B 放 RETRAIN AND PREDICT 完成以后" → 本 wrapper 20:15 启动
run_daily_automation.py --force-retrain (重训+预测+交付一条链, 与 Retrain-2030
一次性任务同命令). 终态 state json 落 logs/daily_automation_20260909.state.json,
已排队的 fade A/B wrapper (pid 12080) 检测到终态+冲突清空后自动接管.
不动常驻调度 (用户昨日的停用保持原状). Start-Process 独立进程启动.
"""
import datetime as dt
import subprocess
import sys

PY = r"C:\Users\91454\AppData\Local\Programs\Python\Python312\python.exe"
LOG = open("tmp_t/_night_chain_manual_0909.log", "a", encoding="utf-8")


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", file=LOG, flush=True)


wake = dt.datetime.now().replace(hour=20, minute=15, second=0, microsecond=0)
if dt.datetime.now() < wake:
    log(f"睡到 {wake} 启动 --force-retrain 夜链 (调度任务 Disabled, 手动补跑)")
    import time
    time.sleep((wake - dt.datetime.now()).total_seconds())

log("启动 run_daily_automation.py --force-retrain")
r = subprocess.run([PY, "-X", "utf8", "scripts/run_daily_automation.py", "--force-retrain"])
log(f"NIGHT_CHAIN_DONE rc={r.returncode}")
sys.exit(r.returncode)
