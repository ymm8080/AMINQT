"""Test Tushare API availability and points level for the missing data columns."""
import os, sys, time
import pandas as pd

# Load token from .env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            break

import tushare as ts

token = os.environ["TUSHARE_TOKEN"]
pro = ts.pro_api(token)

# Check user points
print("=" * 60)
print("TUSHARE API AVAILABILITY TEST")
print("=" * 60)

# 1. Check points
try:
    user = pro.query("user", token=token)
    print(f"\n[Points] {user}")
except Exception as e:
    print(f"\n[Points] Cannot query: {e}")

# 2. Test margin_detail (2000 pts)
print("\n--- margin_detail (融资融券) ---")
try:
    df = pro.margin_detail(trade_date="20260725")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 3. Test top_list (龙虎榜, 2000 pts)
print("\n--- top_list (龙虎榜) ---")
try:
    df = pro.top_list(trade_date="20260725")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 4. Test stk_holdernumber (股东户数, 2000 pts)
print("\n--- stk_holdernumber (股东户数) ---")
try:
    df = pro.stk_holdernumber(ts_code="000001.SZ")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 5. Test stk_holdertrade (股东增减持, 2000 pts)
print("\n--- stk_holdertrade (股东增减持) ---")
try:
    df = pro.stk_holdertrade(start_date="20260101", end_date="20260727")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 6. Test hsgt_top10 (北向个股持股, 5000 pts!)
print("\n--- hsgt_top10 (北向个股持股, needs 5000 pts) ---")
try:
    df = pro.hsgt_top10(trade_date="20260725")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 7. Test moneyflow_hsgt (北向市场级, 2000 pts)
print("\n--- moneyflow_hsgt (北向市场级) ---")
try:
    df = pro.moneyflow_hsgt(start_date="20260720", end_date="20260725")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

# 8. Test daily (pre_close, 2000 pts)
print("\n--- daily (pre_close source) ---")
try:
    df = pro.daily(trade_date="20260725", fields="ts_code,trade_date,pre_close")
    print(f"  OK: {len(df)} rows, pre_close non-null: {df['pre_close'].notna().sum()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 9. Test ccass_hold (港股通持股, alternative for northbound per-stock)
print("\n--- ccass_hold (港股通持股明细) ---")
try:
    df = pro.ccass_hold(trade_date="20260725")
    print(f"  OK: {len(df)} rows, cols={list(df.columns)[:8]}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("DONE")
