# -*- coding: utf-8 -*-
"""bear入板收官验证: null计数日定位 + 0910终值 + 行数健康 + bull 0910时点核查."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

CACHE = Path(r"D:/AMINQT/AMINQT CODES/data/supply_cache/ths_signal")
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"

# 1. bear_counts.csv null 行
csv = pd.read_csv(CACHE / "bear_counts.csv", dtype={"date": str})
null_rows = csv[csv["bear_count"].isna()]
print(f"[1] bear_counts.csv rows={len(csv)} null日={null_rows['date'].tolist()}")

# 2. 面板 bear_pool 覆盖 + null 日
v = pd.read_parquet(PANEL, columns=["symbol", "date", "ths_bear", "ths_bear_pool", "ths_bull"])
in_range = v[v["date"] >= "2023-09-01"]
pool_days = in_range[in_range["ths_bear_pool"].notna()].groupby("date")["ths_bear_pool"].first()
null_pool_days = sorted(set(in_range["date"].dt.strftime("%Y%m%d").unique())
                        - set(pool_days.index.strftime("%Y%m%d")))
print(f"[2] bear_pool 覆盖={len(pool_days)}日 range={pool_days.index.min():%Y%m%d}~"
      f"{pool_days.index.max():%Y%m%d} null日={null_pool_days}")

# 3. 0910 终值 + 行数健康
for d in ("2026-09-09", "2026-09-10"):
    day = v[v["date"] == d]
    bp = day["ths_bear_pool"].dropna()
    print(f"[3] {d}: rows={len(day)} bear_pool={bp.iloc[0] if len(bp) else 'NaN'} "
          f"bear=1={int((day['ths_bear'] == 1).sum())} bull=1={int((day['ths_bull'] == 1).sum())}")

# 4. bear 个股旗标对拍: 0910 bear=1 应=4512
src = pd.read_parquet(CACHE / "bear_20260910.parquet", columns=["股票代码"])
print(f"[4] 源文件 bear_20260910 nunique={src['股票代码'].nunique()} (面板侧应一致)")

# 5. bull 0910 抓取时点 (盘中残照嫌疑)
for d in ("bull_20260909", "bull_20260910"):
    f = CACHE / f"{d}.parquet"
    if f.exists():
        import datetime
        mt = datetime.datetime.fromtimestamp(f.stat().st_mtime)
        n = pd.read_parquet(f, columns=["股票代码"])["股票代码"].nunique()
        print(f"[5] {f.name}: mtime={mt:%m-%d %H:%M} nunique={n}")
