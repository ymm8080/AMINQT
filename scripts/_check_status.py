"""检查 V3 宇宙扩建进度."""

import glob
import json
import os

import pandas as pd

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"

# panel 日历
p = pd.read_parquet(PANEL, columns=["date"])
c = pd.DatetimeIndex(sorted(p["date"].unique()))
print(f"panel dates: {len(c)} ({c[0].date()}..{c[-1].date()})")
print(f"last date str: {c[-1].strftime('%Y%m%d')}")

# progress.json
prog_path = os.path.join(OUT_DIR, "progress.json")
if os.path.exists(prog_path):
    prog = json.load(open(prog_path, encoding="utf-8"))
    print(f"progress.json: {prog}")
else:
    print("progress.json: NOT FOUND")

# alt_data 文件统计
alt_dirs = {
    "adj_factor": "adj_*.parquet",
    "stk_limit": "*_all__.parquet",
    "cyq_tushare": "cyq_*.parquet",
    "lhb": "*.parquet",
    "block_trade": "*.parquet",
}
for d, pat in alt_dirs.items():
    files = glob.glob(os.path.join(ALT_DIR, d, pat))
    print(f"  alt/{d}: {len(files)} files")
margin_files = glob.glob(os.path.join(ALT_DIR, "margin_*.parquet"))
print(f"  alt/margin(root): {len(margin_files)} files")

# new_symbols_raw 文件统计
for d in ["daily", "daily_basic", "suspend", "top_inst", "fina"]:
    files = glob.glob(os.path.join(OUT_DIR, d, "*.parquet"))
    print(f"  new/{d}: {len(files)} files")

# 计算缺口
cal_set = set(c.strftime("%Y%m%d"))

adj_present = {
    os.path.basename(f)[4:12]
    for f in glob.glob(os.path.join(ALT_DIR, "adj_factor", "adj_*.parquet"))
}
lim_present = {
    os.path.basename(f)[:8]
    for f in glob.glob(os.path.join(ALT_DIR, "stk_limit", "*_all__.parquet"))
}

print(f"\nadj gap: {len(cal_set - adj_present)}")
print(f"limit gap: {len(cal_set - lim_present)}")

# 新符号文件数
ns = sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))
if ns:
    univ = pd.read_parquet(ns[-1])
    print(f"\nnew_universe: {len(univ)} symbols (from {os.path.basename(ns[-1])})")
else:
    print("\nnew_universe: NOT FOUND")
