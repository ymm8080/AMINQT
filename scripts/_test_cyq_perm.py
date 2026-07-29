#!/usr/bin/env python3
"""测试 Tushare cyq_perf 权限."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
import tushare as ts

token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

print("测试 cyq_perf 权限...")
try:
    df = pro.cyq_perf(ts_code="000001.SZ", start_date="20250701", end_date="20250710")
    print(f"  成功! 返回 {len(df)} 行")
    print(f"  列: {df.columns.tolist()}")
    if len(df):
        print(f"  日期范围: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
        print(f"  前3行:")
        print(df.head(3).to_string())
        # 速度测试
        t0 = time.time()
        df2 = pro.cyq_perf(ts_code="600519.SH", start_date="20250701", end_date="20250710")
        elapsed = time.time() - t0
        print(f"\n  速度: 600519 耗时 {elapsed:.2f}s, 返回 {len(df2)} 行")
except Exception as e:
    print(f"  失败: {e}")