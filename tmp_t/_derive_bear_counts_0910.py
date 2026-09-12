# -*- coding: utf-8 -*-
"""从 bear_*.parquet 日文件派生全量 bear_counts.csv (enrich契约: date,bear_count).

用法: python tmp_t/_derive_bear_counts_0910.py [src_dir]
  src_dir 默认 data/supply_cache/ths_signal (sync后); 也可指 tmp_t/ths_bear_daily_0910 预检
输出: src_dir/bear_counts.csv (旧文件先备份为 .bak_v2_0910), 覆盖v2时代253天残缺版
"""
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:/AMINQT/AMINQT CODES/data/supply_cache/ths_signal")
OUT = SRC / "bear_counts.csv"

files = sorted(SRC.glob("bear_2*.parquet"))
if not files:
    print(f"[ERROR] no bear_2*.parquet under {SRC}")
    sys.exit(1)

rows = []
for f in files:
    dstr = f.stem.split("_")[1]
    try:
        d = pd.read_parquet(f, columns=["股票代码"])
        n = int(d["股票代码"].nunique())
    except Exception:
        n = 0  # 空壳文件(no_table/no_more日) = 0计数
    rows.append({"date": dstr, "bear_count": n})

out = pd.DataFrame(rows).drop_duplicates(subset=["date"]).sort_values("date")
missing_code = [r["date"] for r in rows if r["bear_count"] == 0]
if OUT.exists() and not OUT.name.endswith(".bak_v2_0910"):
    shutil.copy2(OUT, OUT.with_name("bear_counts.csv.bak_v2_0910"))
out.to_csv(OUT, index=False)
print(f"[done] {len(out)}天 {out['date'].iloc[0]}~{out['date'].iloc[-1]} "
      f"count 中位{int(out['bear_count'].median())} 最大{int(out['bear_count'].max())}")
if missing_code:
    print(f"[WARN] {len(missing_code)}天0计数: {missing_code[:10]}")
