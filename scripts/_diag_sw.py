"""Diagnose sw_daily matching issue."""

import os

import tushare as ts
from dotenv import load_dotenv

load_dotenv()

pro = ts.pro_api(ts.get_token() or os.getenv("TUSHARE_TOKEN"))

# 1. Get sw_daily names
sw = pro.sw_daily(trade_date="20260731")
print(f"sw_daily: {len(sw)} rows")
print(f"Columns: {sw.columns.tolist()}")

# Filter to first-level SW indices (801xxx)
sw1 = sw[sw["ts_code"].str.startswith("801")].copy()
print(f"\nFirst-level SW (801xxx): {len(sw1)} indices")
sw_names = sorted(sw1["name"].unique())
print(f"SW names ({len(sw_names)}):")
for n in sw_names:
    print(f"  [{n}]")

# 2. Get stock_basic industries
sb = pro.stock_basic(list_status="L", fields="ts_code,symbol,name,industry")
industries = sorted(sb["industry"].dropna().unique())
print(f"\nstock_basic industries ({len(industries)}):")
for ind in industries:
    print(f"  [{ind}]")

# 3. Check overlap
sw_set = set(sw_names)
sb_set = set(industries)
matched = sw_set & sb_set
print(f"\nExact matches: {len(matched)}")
print(f"  Matched: {sorted(matched)}")
print(f"  In SW but not stock_basic: {sorted(sw_set - sb_set)}")
print(f"  In stock_basic but not SW: {sorted(sb_set - sw_set)}")
