# -*- coding: utf-8 -*-
"""0910 收盘后 THS 终值刷新 (无人值守, 全输出落盘).

背景: 午间临时抓取的 bull_20260910 仅含上午盘 (非终值), 看跌池未抓.
本脚本: 挪开临时文件(WORM改名不删) → 生产抓取器抓终值(看涨+看跌) → enrich → merge --replace → 验证.
退出码: 0=成功, 非0=失败 (fail-fast, 供监控判终态).
"""
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:/AMINQT/AMINQT CODES")
LOG = ROOT / "tmp_t" / "_ths_evening_refresh_0910.log"
CACHE = ROOT / "data" / "supply_cache" / "ths_signal"
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
DSTR = "20260910"

_log_f = open(LOG, "a", encoding="utf-8", buffering=1)
sys.stdout = _log_f
sys.stderr = _log_f


def run(cmd, tag):
    print(f"[{tag}] $ {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=str(ROOT))
    for line in (r.stdout or "").splitlines():
        print(f"[{tag}|out] {line}")
    for line in (r.stderr or "").splitlines():
        print(f"[{tag}|err] {line}")
    print(f"[{tag}] exit={r.returncode} {time.time()-t0:.0f}s")
    if r.returncode != 0:
        print(f"[FATAL] {tag} 失败, 终止")
        sys.exit(r.returncode or 1)
    return r


# ── 1. 挪开午间临时文件 (WORM: 改名不删) ──
for name in (f"bull_{DSTR}.parquet", f"bear_{DSTR}.parquet"):
    f = CACHE / name
    if f.exists():
        dst = f.with_suffix(".parquet.provisional_midday")
        if dst.exists():
            dst.unlink()  # 只覆盖本脚本自己产出的临时备份
        f.rename(dst)
        print(f"[move] {name} -> {dst.name}")

# ── 2. 生产抓取器: 看涨+看跌终值 ──
run([sys.executable, str(ROOT / "scripts" / "_fetch_ths_signal.py"),
     "--date", DSTR], "fetch")

bull = CACHE / f"bull_{DSTR}.parquet"
bear = CACHE / f"bear_{DSTR}.parquet"
assert bull.exists(), "bull_20260910.parquet 未产出"
assert bear.exists(), "bear_20260910.parquet 未产出"

# ── 3. enrich (幂等重跑已修) ──
run([sys.executable, str(ROOT / "scripts" / "enrich_one_source.py"),
     "--source", "ths_signal", "--panel", PANEL], "enrich")

# ── 4. merge --replace ──
run([sys.executable, str(ROOT / "tmp_t" / "_merge_ths_into_panel_0910.py"),
     "--replace"], "merge")

# ── 5. 验证 ──
import pandas as pd

v = pd.read_parquet(PANEL, columns=["symbol", "date", "ths_bull", "ths_bear_pool"],
                    filters=[("date", "=", pd.Timestamp("2026-09-10"))])
nb = int((v["ths_bull"] == 1.0).sum())
nr = int(v["ths_bear_pool"].notna().sum())
print(f"[verify] 0910: rows={len(v)} ths_bull={nb} bear_nonnull={nr}")
assert nb >= 20, f"看涨池行数异常: {nb}"
assert nr > 0, "看跌池未入板"
print("[refresh] 终值刷新完成")
print("SCRIPT-END")
