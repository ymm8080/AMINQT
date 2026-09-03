# -*- coding: utf-8 -*-
"""手动晋升 main_20260902 -> current (复刻 _retrain_legacy_full.py 切换语义, 仅 main).

依据: 终榜回放 _dual_pkg_finaltop_compare_20260902_201413.json
main B vs A: d3_full +0.55pp/日, 双半 +0.78/+0.32pp, 胜率 30/18 — 新判据三条件全过.
IC 闸已过 (0.0955, logs/daily_automation_20260902.log). dual 不动 (B FAIL, C 平).
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.model_meta import load_modules, save_modules

MODEL_DIR = "models/pipeline1"
TAG = "20260902"
src = os.path.join(MODEL_DIR, f"main_{TAG}.pkl")
cur = os.path.join(MODEL_DIR, "main_current.pkl")
bak = os.path.join(MODEL_DIR, "main_current_retrain_backup.pkl")

assert os.path.exists(src), f"缺新包: {src}"
if os.path.exists(cur) and not os.path.exists(bak):
    shutil.copy(cur, bak)
    print(f"[main] 旧 current 备份 -> {bak}", flush=True)
shutil.copy(src, cur)
mods = load_modules()
mods["main"] = {
    "tag": TAG,
    "file": os.path.basename(src),
    "updated": time.strftime("%Y-%m-%d %H:%M"),
}
save_modules(mods)
print(f"[main] switched -> current = {src}", flush=True)
print(f"[meta] main = {mods['main']}", flush=True)
print(f"[meta] dual (不动) = {mods.get('dual')}", flush=True)
