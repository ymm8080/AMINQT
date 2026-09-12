# -*- coding: utf-8 -*-
"""Bear pass-2 v3 + A/B #2 单链 (09-11): v2 enrich 路径 bug 修复版 + 后置 A/B.

v2 事故: 队列调 enrich 未传 --panel, DEFAULT_PANEL=repo内 data/ 不存在 (真实面板
在 D:/AMINQT/PARQUET/) → rc=1. sync 已于 06:14 完成并验证 733+733, 不重跑.

本链:
  A. 等 冲突清空 + RAM>=6GB (deadline 19:00) → enrich --refresh --panel <真实路径>
  B. 覆盖率验证 (bear_pool >= 600 日) → "[pass2v3] DONE"
  C. 等 冲突清空 + RAM>=5GB + now>=19:45 (16:26 周五 cron 重训之后, 勿抢周重训)
     → 跑 _ths_bear_ab_minibacktest_v2_0911.py --board main, rc=3 等30min重试
WORM 日志 tmp_t/_bear_pass2v3_ab_chain_0911.log
"""
import datetime as dt
import subprocess
import sys
import time

sys.path.insert(0, ".")

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
HARNESS = "tmp_t/_ths_bear_ab_minibacktest_v2_0911.py"
LOG = open("tmp_t/_bear_pass2v3_ab_chain_0911.log", "a", encoding="utf-8")


def log(msg):
    print(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}", file=LOG, flush=True)


def now():
    return dt.datetime.now()


def at(h, m):
    t = now().replace(hour=h, minute=m, second=0, microsecond=0)
    if t < now():
        t += dt.timedelta(days=1)
    return t


DL_ENRICH = at(19, 0)
DL_AB = at(23, 30)
AB_NO_EARLY = at(19, 45)


def past(t):
    return now() >= t


def conflicts():
    from scripts._run_guard import find_conflicts

    return find_conflicts()


def free_gb():
    import psutil

    return psutil.virtual_memory().available / 1024**3


def wait_room(need_gb, deadline, tag):
    while True:
        c = conflicts()
        if not c and free_gb() >= need_gb:
            return True
        if past(deadline):
            log(f"deadline 过 ({tag} 未跑), 放弃")
            return False
        log(f"等待 ({tag}): 冲突={[x['sentinel'] for x in c]} free={free_gb():.1f}GB")
        time.sleep(600)


# ── A. enrich (修复: 显式 --panel 真实路径) ──
if not wait_room(6.0, DL_ENRICH, "enrich"):
    sys.exit(3)
log(f"[pass2v3] enrich 启动 (free={free_gb():.1f}GB)")
r = subprocess.run([
    sys.executable, "-X", "utf8",
    "scripts/enrich_one_source.py", "--source", "ths_signal", "--refresh",
    "--panel", PANEL,
])
if r.returncode != 0:
    log(f"FAIL: enrich rc={r.returncode}")
    sys.exit(5)
log("[pass2v3] enrich ok")

# ── B. 覆盖率验证 ──
import pandas as pd

v = pd.read_parquet(PANEL, columns=["date", "ths_bear_pool", "ths_bull"])
bear_days = int(v.loc[v["ths_bear_pool"].notna(), "date"].nunique())
bull_days = int(v.loc[v["ths_bull"].notna(), "date"].nunique())
log(f"[pass2v3] 面板覆盖: bear_pool {bear_days} 日 / bull {bull_days} 日")
del v
if bear_days < 600:
    log("FAIL-LOUD: bear_pool 覆盖 <600 日, 兜底: "
        "python tmp_t/_merge_ths_into_panel_0910.py --replace 后复验")
    sys.exit(6)
log("[pass2v3] DONE — bear 全史已入板, A/B 转入 C 段等窗")

# ── C. A/B #2 (严格 post-cron: >=19:45) ──
while True:
    c = conflicts()
    ok = (not c) and free_gb() >= 5.0 and now() >= AB_NO_EARLY
    if ok:
        break
    if past(DL_AB):
        log("deadline 过 (A/B 未跑), 放弃")
        sys.exit(3)
    log(f"等待 (A/B): 冲突={[x['sentinel'] for x in c]} free={free_gb():.1f}GB "
        f"now={now():%H:%M} (须>=19:45)")
    time.sleep(600)

while True:
    log("[ab] 启动 bear A/B #2 (main)")
    r = subprocess.run([sys.executable, "-X", "utf8", HARNESS, "--board", "main"])
    rc = r.returncode
    log(f"[ab] harness rc={rc}")
    if rc != 3:
        break
    if past(DL_AB):
        log("RAM guard 反复不过, 放弃")
        sys.exit(3)
    log("rc=3 (RAM<5GB), 等 30min 重试")
    time.sleep(1800)
sys.exit(rc if isinstance(rc, int) else 1)
