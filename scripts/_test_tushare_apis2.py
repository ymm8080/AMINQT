"""Test Tushare API with proper trading day."""
import os, sys
import pandas as pd

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            break

import tushare as ts
pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])

# Use 20260724 (Friday) and 20260728 (Monday) as trading days
TD = "20260724"

print("=" * 60)
print(f"TUSHARE API TEST (trade_date={TD})")
print("=" * 60)

# 1. margin_detail
print("\n--- margin_detail ---")
try:
    df = pro.margin_detail(trade_date=TD)
    print(f"  OK: {len(df)} rows, {df['ts_code'].nunique()} symbols")
    if len(df) > 0:
        print(f"  Cols: {list(df.columns)}")
        print(f"  Sample: {df.head(2).to_string()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. top_list
print("\n--- top_list ---")
try:
    df = pro.top_list(trade_date=TD)
    print(f"  OK: {len(df)} rows, {df['ts_code'].nunique() if 'ts_code' in df.columns else 0} symbols")
    if len(df) > 0:
        print(f"  Cols: {list(df.columns)}")
except Exception as e:
    print(f"  FAIL: {e}")

# 3. hsgt_top10 (per-stock northbound — the critical one!)
print("\n--- hsgt_top10 (per-stock northbound) ---")
try:
    df = pro.hsgt_top10(trade_date=TD)
    print(f"  OK: {len(df)} rows, {df['ts_code'].nunique() if 'ts_code' in df.columns else 0} symbols")
    if len(df) > 0:
        print(f"  Cols: {list(df.columns)}")
        print(f"  Sample:\n{df.head(3).to_string()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 4. daily with pre_close
print("\n--- daily (pre_close) ---")
try:
    df = pro.daily(trade_date=TD, fields="ts_code,trade_date,pre_close,close,open,high,low,vol,amount")
    print(f"  OK: {len(df)} rows")
    if len(df) > 0:
        print(f"  pre_close non-null: {df['pre_close'].notna().sum()}/{len(df)}")
        print(f"  Sample:\n{df.head(3).to_string()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 5. Test hsgt_top10 with start_date/end_date (does it support range?)
print("\n--- hsgt_top10 (range query test) ---")
try:
    df = pro.hsgt_top10(start_date="20260720", end_date="20260724")
    print(f"  Range query: {len(df)} rows, {df['trade_date'].nunique() if 'trade_date' in df.columns else 0} dates")
    if len(df) > 0:
        print(f"  Dates: {sorted(df['trade_date'].unique())}")
except Exception as e:
    print(f"  FAIL (range not supported?): {e}")

# 6. Check stk_holdertrade total for a date range
print("\n--- stk_holdertrade (range, paginated) ---")
try:
    all_rows = []
    offset = 0
    while True:
        df = pro.stk_holdertrade(start_date="20260701", end_date="20260724", limit=3000, offset=offset)
        if df is None or len(df) == 0:
            break
        all_rows.append(df)
        if len(df) < 3000:
            break
        offset += 3000
    total = pd.concat(all_rows) if all_rows else pd.DataFrame()
    print(f"  Total: {len(total)} rows, {total['ts_code'].nunique() if 'ts_code' in total.columns else 0} symbols")
    if len(total) > 0:
        print(f"  Cols: {list(total.columns)}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "=" * 60)
print("DONE")
