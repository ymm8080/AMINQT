# -*- coding: utf-8 -*-
"""同步看涨池快照 + 看跌计数 CSV 到 supply_cache/ths_signal/ (enrich 的读取目录)."""
import io
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd

SRC = Path(r"D:/AMINQT/AMINQT CODES/tmp_t/ths_bull_daily_0910")
DST = Path(r"D:/AMINQT/AMINQT CODES/data/supply_cache/ths_signal")
DST.mkdir(parents=True, exist_ok=True)

# ── 1. bull parquets: 只拷 mtime>60s 的稳定文件, 拷完逐一读取验证 ──
now = time.time()
copied, skipped = 0, 0
for f in sorted(SRC.glob("bull_*.parquet")):
    if now - f.stat().st_mtime < 60:  # 可能还在写, 跳过等下轮
        skipped += 1
        continue
    shutil.copy2(f, DST / f.name)
    copied += 1
print(f"[bull] copied={copied} skipped_unstable={skipped}")

bad = []
for f in sorted(DST.glob("bull_*.parquet")):
    try:
        pd.read_parquet(f)
    except Exception as e:
        bad.append((f.name, str(e)[:60]))
for name, err in bad:
    (DST / name).unlink()
    print(f"[bull] BAD removed {name}: {err}")
print(f"[bull] validated {len(list(DST.glob('bull_*.parquet'))) } files in supply_cache")

# ── 1b. bear parquets: 看跌个股级 (用户指令"看跌也要到个股天的颗粒度"), 同模式 ──
BSRC = Path(r"D:/AMINQT/AMINQT CODES/tmp_t/ths_bear_daily_0910")
copied, skipped, skipped_empty = 0, 0, 0
if BSRC.exists():
    for f in sorted(BSRC.glob("bear_*.parquet")):
        if now - f.stat().st_mtime < 60:
            skipped += 1
            continue
        try:
            pd.read_parquet(f, columns=["股票代码"])
        except Exception:
            skipped_empty += 1  # 空壳(no_table/no_more日)不进supply_cache, 池计数由bear_counts.csv承担
            continue
        shutil.copy2(f, DST / f.name)
        copied += 1
print(f"[bear] copied={copied} skipped_unstable={skipped} skipped_empty={skipped_empty}")

bad = []
for f in sorted(DST.glob("bear_*.parquet")):
    try:
        pd.read_parquet(f)
    except Exception as e:
        bad.append((f.name, str(e)[:60]))
for name, err in bad:
    (DST / name).unlink()
    print(f"[bear] BAD removed {name}: {err}")
print(f"[bear] validated {len(list(DST.glob('bear_*.parquet')))} files in supply_cache")

# ── 2. bear_counts.csv: 旧(105行) + 回填版 拼接去重(回填优先), 备份旧文件 ──
old_f = DST / "bear_counts.csv"
new_f = SRC / "bear_counts.csv"
frames = []
if old_f.exists():
    for attempt in range(3):
        try:
            o = pd.read_csv(old_f, dtype={"date": str})
            frames.append(("old", o))
            break
        except Exception:
            time.sleep(2)
if new_f.exists():
    for attempt in range(5):
        try:
            n = pd.read_csv(new_f, dtype={"date": str})
            frames.append(("new", n))
            break
        except Exception:
            time.sleep(3)
if not frames:
    print("[bear_csv] no readable source, skip")
    sys.exit(0)
merged = pd.concat([f for _, f in frames], ignore_index=True)
merged = merged.drop_duplicates(subset=["date"], keep="last").sort_values("date")
if old_f.exists() and not old_f.with_suffix(".csv.bak_0910").exists():
    shutil.copy2(old_f, old_f.with_suffix(".csv.bak_0910"))
merged.to_csv(old_f, index=False)
print(f"[bear_csv] rows={len(merged)} range={merged['date'].min()}~{merged['date'].max()} "
      f"(null_count={int(merged['bear_count'].isna().sum())})")
