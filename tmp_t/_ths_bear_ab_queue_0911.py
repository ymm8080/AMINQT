# -*- coding: utf-8 -*-
"""THS 看跌池特征 A/B #2 排队 wrapper (09-11): 三重前置后跑
_ths_bear_ab_minibacktest_v2_0911.py --board main (镜像 bull A/B 单板口径,
LEGACY 清单现为 main-only, main 板判定即决策对象).

前置链:
  1. pass-2 v2 队列 (PID 9612) 退出且其日志含 "[pass2] DONE" — 否则 fail-fast
     (面板无 bear 全史数据, A/B 无意义);
  2. bull A/B 队列 (PID 19128) 退出 — 两台 A/B 各自 5GB RAM 档, 并发必互踩;
  3. run-guard 冲突清空 + RAM≥5GB (给下午 16:26 周五 cron 重训让路)。
rc=3 (RAM guard) 等 30min 重试; 20:00 死线放弃留日志。
WORM 日志 tmp_t/_ths_bear_ab_queue_0911.log。
"""
import datetime as dt
import subprocess
import sys
import time

sys.path.insert(0, ".")

PASS2_PID = 9612
BULL_AB_PID = 19128
PASS2_LOG = "tmp_t/_bear_pass2_queue2_0910.log"
HARNESS = "tmp_t/_ths_bear_ab_minibacktest_v2_0911.py"
LOG = open("tmp_t/_ths_bear_ab_queue_0911.log", "a", encoding="utf-8")
DEADLINE = dt.datetime.now().replace(hour=20, minute=0, second=0, microsecond=0)
if dt.datetime.now() >= DEADLINE:
    DEADLINE += dt.timedelta(days=1)


def log(msg):
    print(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}", file=LOG, flush=True)


def alive(pid):
    import psutil

    return psutil.pid_exists(pid)


def conflicts():
    from scripts._run_guard import find_conflicts

    return find_conflicts()


def free_gb():
    import psutil

    return psutil.virtual_memory().available / 1024**3


def past_deadline():
    return dt.datetime.now() >= DEADLINE


# ── 1. 等 pass-2 v2 完成 (DONE 才算数据落地) ──
while alive(PASS2_PID):
    if past_deadline():
        log("deadline 过, pass-2 未完, 放弃")
        sys.exit(3)
    time.sleep(300)
with open(PASS2_LOG, encoding="utf-8", errors="replace") as fh:
    txt = fh.read()
if "[pass2] DONE" not in txt:
    log("FAIL-LOUD: pass-2 队列退出但无 DONE 行 (rc 见其日志), 不跑 A/B #2")
    sys.exit(2)
log("[pre1] pass-2 DONE 确认")

# ── 2. 等 bull A/B 队列退场 ──
while alive(BULL_AB_PID):
    if past_deadline():
        log("deadline 过, bull A/B 仍未完, 放弃")
        sys.exit(3)
    time.sleep(300)
log("[pre2] bull A/B 队列已退场")

# ── 3. 冲突+RAM 清场 ──
while True:
    if past_deadline():
        log("deadline 过 (清场未成), 放弃")
        sys.exit(3)
    c = conflicts()
    if not c and free_gb() >= 5.0:
        break
    log(f"等待 (清场): 冲突={[x['sentinel'] for x in c]} free={free_gb():.1f}GB")
    time.sleep(600)

# ── 4. 跑 A/B (rc=3 重试 30min) ──
while True:
    log(f"启动 bear A/B #2 (main)")
    r = subprocess.run([sys.executable, "-X", "utf8", HARNESS, "--board", "main"])
    rc = r.returncode
    log(f"harness rc={rc}")
    if rc != 3:
        break
    if past_deadline():
        log("RAM guard 反复不过, 放弃")
        sys.exit(3)
    log("rc=3 (RAM<5GB), 等 30min 重试")
    time.sleep(1800)
sys.exit(rc if isinstance(rc, int) else 1)
