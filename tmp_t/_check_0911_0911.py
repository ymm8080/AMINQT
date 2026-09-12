# -*- coding: utf-8 -*-
"""0911数据完整性验证 + 外部时钟锚 (机器时钟滑移嫌疑)."""
import datetime as dt
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import requests

# 1. 外部时钟锚: HTTP Date 响应头 (权威UTC)
try:
    r = requests.head("https://www.baidu.com", timeout=8)
    srv = dt.datetime.strptime(r.headers["Date"], "%a, %d %b %Y %H:%M:%S GMT")
    print(f"[1] 外部权威时间 = {srv + dt.timedelta(hours=8):%m-%d %H:%M} (北京时间) | "
          f"本机时钟 = {dt.datetime.now():%m-%d %H:%M}")
except Exception as e:
    print(f"[1] 外部时钟获取失败: {type(e).__name__} {e}")

# 2. supply_cache 0911 文件
C = Path("data/supply_cache/ths_signal")
for n in ("bull_20260911", "bear_20260911"):
    f = C / f"{n}.parquet"
    if f.exists():
        d = pd.read_parquet(f, columns=["股票代码"])
        cols = list(pd.read_parquet(f, columns=[]).columns)
        dated = [c for c in cols if "[" in c and "]" in c][:2]
        mt = dt.datetime.fromtimestamp(f.stat().st_mtime)
        print(f"[2] {n}: nunique={d['股票代码'].nunique()} mtime={mt:%m-%d %H:%M} 日期列={dated}")
    else:
        print(f"[2] {n}: 不存在")
csv = pd.read_csv(C / "bear_counts.csv", dtype={"date": str})
print(f"[2] bear_counts 0911 = {csv.loc[csv.date == '20260911', 'bear_count'].tolist()}")

# 3. 面板 0911 行一致性 + 去重
v = pd.read_parquet("D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
                    columns=["symbol", "date", "ths_bull", "ths_bear", "ths_bear_pool"])
d11 = v[v["date"] == "2026-09-11"]
dup = len(d11) - d11["symbol"].nunique()
print(f"[3] 面板0911: rows={len(d11)} nunique={d11['symbol'].nunique()} 重复={dup} "
      f"bull=1={int((d11.ths_bull == 1).sum())} bear=1={int((d11.ths_bear == 1).sum())} "
      f"pool={sorted(d11.ths_bear_pool.dropna().unique())[:2]}")

# 4. 进程时钟对照: 夜链/重训是否已跑过的痕迹
for lg in ("logs/_daily_fetch_20260911.log",):
    p = Path(lg)
    if p.exists():
        mt = dt.datetime.fromtimestamp(p.stat().st_mtime)
        print(f"[4] {lg} mtime={mt:%m-%d %H:%M}")
