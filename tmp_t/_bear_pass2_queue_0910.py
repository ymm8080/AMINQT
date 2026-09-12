# -*- coding: utf-8 -*-
"""Bear pass-2 重灌排队器 (09-10): 等 _bear_backfill3y_0910.py (PID 8408) 全量扫描
终态后, 自动执行 sync → enrich --refresh (幂等 drop 即 replace 语义) → 覆盖率验证,
把看跌池全史灌回生产面板, 为 bear A/B #2 铺路.

终态判据 (bull 教训): 必须看 "[done] audit complete N/N" 日志行, 进程死亡但无该行
= 中途崩, fail-fast 报错不静默. 步骤前查 run-guard 冲突 + RAM≥5GB (A/B #1 与
babysitter refresh 错峰). WORM 日志 tmp_t/_bear_pass2_queue_0910.log.
"""
import datetime as dt
import subprocess
import sys
import time

sys.path.insert(0, ".")

SWEEP_TAG = "_bear_backfill3y_0910.py"
SWEEP_OUT = "tmp_t/_bear_backfill3y_0910.out"
LOG = open("tmp_t/_bear_pass2_queue_0910.log", "a", encoding="utf-8")
DEADLINE = dt.datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
if dt.datetime.now() >= DEADLINE:
    DEADLINE += dt.timedelta(days=1)


def log(msg):
    print(f"[{dt.datetime.now():%m-%d %H:%M:%S}] {msg}", file=LOG, flush=True)


def sweep_alive():
    import psutil

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        if "python" not in (p.info["name"] or "").lower():
            continue
        if SWEEP_TAG in " ".join(p.info["cmdline"] or []):
            return True
    return False


def conflicts():
    from scripts._run_guard import find_conflicts

    return find_conflicts()


def free_gb():
    import psutil

    return psutil.virtual_memory().available / 1024**3


def past_deadline():
    return dt.datetime.now() >= DEADLINE


# ── 1. 等扫描进程退出 ──
while sweep_alive():
    if past_deadline():
        log("deadline 过, 扫描仍未完, 放弃 (下次会话重跑)")
        sys.exit(3)
    time.sleep(300)
log("扫描进程已退出, 等 30s 缓冲刷盘")
time.sleep(30)

# ── 2. 终态判据: audit complete 行 ──
with open(SWEEP_OUT, encoding="utf-8", errors="replace") as fh:
    txt = fh.read()
if "[done] audit complete" not in txt:
    log("FAIL-LOUD: 扫描进程死亡但无 '[done] audit complete' 行 = 中途崩, 不重灌")
    sys.exit(2)
log("[pass2] 扫描终态确认 (audit complete), 等 180s 文件稳定后接力")
time.sleep(180)

# ── 3. sync (轻, 先做) ──
while True:
    if past_deadline():
        log("deadline 过 (sync 未跑), 放弃")
        sys.exit(3)
    c = conflicts()
    if not c and free_gb() >= 5.0:
        break
    log(f"等待 (sync): 冲突={[x['sentinel'] for x in c]} free={free_gb():.1f}GB")
    time.sleep(600)
r = subprocess.run([sys.executable, "-X", "utf8", "tmp_t/_ths_sync_supply_0910.py"])
if r.returncode != 0:
    log(f"FAIL: sync rc={r.returncode}")
    sys.exit(4)
log("[pass2] sync ok")

# ── 4. enrich --refresh (重, 再查一轮冲突+RAM) ──
while True:
    if past_deadline():
        log("deadline 过 (enrich 未跑), 放弃")
        sys.exit(3)
    c = conflicts()
    if not c and free_gb() >= 5.0:
        break
    log(f"等待 (enrich): 冲突={[x['sentinel'] for x in c]} free={free_gb():.1f}GB")
    time.sleep(600)
r = subprocess.run([
    sys.executable, "-X", "utf8",
    "scripts/enrich_one_source.py", "--source", "ths_signal", "--refresh",
])
if r.returncode != 0:
    log(f"FAIL: enrich rc={r.returncode}")
    sys.exit(5)
log("[pass2] enrich --refresh ok")

# ── 5. 覆盖率验证 ──
import pandas as pd

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
v = pd.read_parquet(PANEL, columns=["date", "ths_bear_pool", "ths_bull"])
bear_days = int(v.loc[v["ths_bear_pool"].notna(), "date"].nunique())
bull_days = int(v.loc[v["ths_bull"].notna(), "date"].nunique())
log(f"[pass2] 面板覆盖: bear_pool {bear_days} 日 / bull {bull_days} 日")
if bear_days < 600:
    log(f"FAIL-LOUD: bear_pool 覆盖 {bear_days} < 600 日, 兜底路径: "
        f"python tmp_t/_merge_ths_into_panel_0910.py --replace 后复验")
    sys.exit(6)
log("[pass2] DONE — bear 全史已入板, bear A/B #2 可评")
sys.exit(0)
