# -*- coding: utf-8 -*-
"""_data_completion_0910.py — 网络真实时间 + 面板最新日期完成率核查 (09-10)."""
import sys, io, time, email.utils
import urllib.request
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 1) 网络真实时间 (勿信系统时钟 — 本地时钟曾有偏差)
for host in ("https://www.baidu.com", "https://www.taobao.com"):
    try:
        r = urllib.request.urlopen(host, timeout=8)
        net_dt = email.utils.parsedate_to_datetime(r.headers["Date"])
        print(f"[net-time] {host} -> {net_dt} (UTC {net_dt.tzinfo})")
        break
    except Exception as e:
        print(f"[net-time] {host} FAIL: {e}")

import datetime as _dt
print(f"[sys-time] local now = {_dt.datetime.now()}")

# 2) Tushare: 交易日历 + 0910 日线可用性
try:
    import tushare as ts
    pro = ts.pro_api()
    cal = pro.trade_cal(exchange="SSE", start_date="20260901", end_date="20260915")
    print("[trade_cal]"); print(cal.to_string(index=False))
    for d in ("20260909", "20260910"):
        try:
            df = pro.daily(trade_date=d, fields="ts_code")
            print(f"[tushare daily {d}] rows={len(df)}")
        except Exception as e:
            print(f"[tushare daily {d}] FAIL: {e}")
except Exception as e:
    print(f"[tushare] FAIL: {e}")

# 3) 各数据面最新日期行数完成率
def completion(path, date_col, label, n=8):
    try:
        df = pd.read_parquet(path, columns=[date_col])
        d = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
        cnt = d.value_counts().sort_index().tail(n)
        med = d.value_counts().sort_index().tail(30).median()
        print(f"\n[{label}] {path}")
        for k, v in cnt.items():
            print(f"  {k}: {v} rows ({v/med*100:.1f}% of median {med:.0f})")
    except Exception as e:
        print(f"[{label}] FAIL: {e}")

completion("D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet", "date", "v3_panel")
completion("data/cyq_panel.parquet", "date", "cyq_panel")
completion("data/supply_cache/alt_data/block_trade/block_trade_full.parquet", "date", "block_trade")
completion("data/processed/sw_daily_history.parquet", "trade_date", "sw_daily")
