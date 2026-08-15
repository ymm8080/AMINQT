# -*- coding: utf-8 -*-
"""Test TUSHARE_TOKEN."""
import sys
sys.path.insert(0, "D:\\AMINQT\\AMINQT CODES")
import tushare as ts
from config import settings

def main():
    if not settings.TUSHARE_TOKEN:
        print("ERROR: TUSHARE_TOKEN is empty. Set it in .env")
        return
    print(f"Token length: {len(settings.TUSHARE_TOKEN)}")
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    try:
        df = pro.daily(ts_code="000001.SZ", trade_date="20250124")
        if not df.empty:
            print("OK: TUSHARE reachable, got daily")
        else:
            print("EMPTY daily (maybe trading day not exist)")
        # test adj
        a = pro.adj_factor(ts_code="000001.SZ", trade_date="20250124")
        if not a.empty:
            print("OK: adj_factor")
        else:
            print("EMPTY adj_factor")
    except Exception as e:
        print(f"FAIL: {e}")

if __name__ == "__main__":
    main()