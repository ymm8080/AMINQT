# -*- coding: utf-8 -*-
"""Bear pass-2 重灌排队器 v2 (09-11): v1 误判修复版.

v1 事故: 终态判据用了字面量 "[done] audit complete", 但 bear 脚本实际格式是
"[done] 全部交易日完成 audit complete 733/733" — 子串不匹配 → 733/733 全成后
仍误判崩溃 exit 2. v2 判据改为 ASCII 子串 "audit complete" (bull/bear 格式通吃).

流程同 v1: 等扫描进程退出 → 终态确认 → sync → enrich --refresh (幂等 drop 即
replace) → 覆盖率验证 (bear_pool ≥600 日). 步骤前查 run-guard 冲突 + RAM≥5GB.
WORM 日志 tmp_t/_bear_pass2_queue2_0910.log.
"""
import datetime as dt
import subprocess
import sys
import time

sys.path.insert(0, ".")

SWEEP_TAG = "_bear_backfill3y_0910.py"
SWEEP_OUT = "tmp_t/_bear_backfill3y_0910.out"
LOG = open("tmp_t/_bear_pass2_queue2_0910.log", "a", encoding="utf-8")
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


# ── 1. 等扫描进程退出 (v1 事故时它已死, 此循环立即通过) ──
while sweep_alive():
    if past_deadline():
        log("deadline 过, 扫描仍未完, 放弃 (下次会话重跑)")
        sys.exit(3)
    time.sleep(300)
log("扫描进程已退出, 等 30s 缓冲刷盘")
time.sleep(30)

# ── 2. 终态判据: "audit complete" ASCII 子串 (v2 修正) ──
with open(SWEEP_OUT, encoding="utf-8", errors="replace") as fh:
    txt = fh.read()
if "audit complete" not in txt:
    log("FAIL-LOUD: 扫描进程死亡但无 'audit complete' 行 = 中途崩, 不重灌")
    sys.exit(2)
import re

m = re.search(r"audit complete (\d+)/(\d+)", txt)
log(f"[pass2] 扫描终态确认: audit complete {m.group(1) if m else '?'}/{m.group(2) if m else '?'}")
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
