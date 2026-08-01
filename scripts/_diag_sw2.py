# -*- coding: utf-8 -*-
"""Diagnose sw_daily matching issue - output to file for proper encoding."""

import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()

pro = ts.pro_api(ts.get_token() or os.getenv("TUSHARE_TOKEN"))

# 1. Get sw_daily
sw = pro.sw_daily(trade_date="20260731")
# Filter to first-level SW (801010-801890, exactly 28)
sw1 = sw[sw["ts_code"].str.match(r"^801\d{3}$")].copy()
sw1_names = sorted(sw1["name"].unique())

# 2. Get stock_basic industries
sb = pro.stock_basic(list_status="L", fields="ts_code,symbol,name,industry")
industries = sorted(sb["industry"].dropna().unique())

# 3. Write results to file
with open("scripts/_sw_diag_output.txt", "w", encoding="utf-8") as f:
    f.write(f"sw_daily total: {len(sw)} rows\n")
    f.write(f"First-level SW (801xxx): {len(sw1)} indices\n")
    f.write(f"SW names ({len(sw1_names)}):\n")
    for n in sw1_names:
        f.write(f"  [{n}]\n")
    f.write(f"\nstock_basic industries ({len(industries)}):\n")
    for ind in industries:
        f.write(f"  [{ind}]\n")

    sw_set = set(sw1_names)
    sb_set = set(industries)
    matched = sw_set & sb_set
    f.write(f"\nExact matches: {len(matched)} / {len(sw_set)} SW, {len(sb_set)} SB\n")
    f.write(f"Matched: {sorted(matched)}\n")
    f.write(f"In SW but not SB: {sorted(sw_set - sb_set)}\n")
    f.write(f"In SB but not SW: {sorted(sb_set - sw_set)}\n")

    # Count stocks per industry
    stock_counts = sb["industry"].value_counts()
    f.write("\nStock counts per industry (top 10):\n")
    for ind, count in stock_counts.head(10).items():
        in_sw = "YES" if ind in sw_set else "NO"
        f.write(f"  [{ind}]: {count} stocks -> SW match: {in_sw}\n")

print("Output written to scripts/_sw_diag_output.txt")
