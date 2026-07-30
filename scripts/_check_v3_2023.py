#!/usr/bin/env python3
"""Direct check: compare v3 before and after fill for specific stocks in 2023 H1."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd

# Load current v3 (post-fill)
v3 = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=["symbol", "date", "roe"])

# Load backup (pre-fill)
v3_old = pd.read_parquet("data/panel_full_enriched_v3_backup_20260729.parquet", columns=["symbol", "date", "roe"])

# Check 000001 in early 2023
for stock in ["000001", "000002", "000620", "600519"]:
    print(f"\n=== {stock} ===")
    # Current v3
    cur = v3[(v3["symbol"] == stock) & (v3["date"] < "2023-07-01")]
    old = v3_old[(v3_old["symbol"] == stock) & (v3_old["date"] < "2023-07-01")]
    print(f"  Pre-Jul 2023 rows: {len(cur)}")
    print(f"  Current v3 roe: NaN={cur['roe'].isna().sum()}, non-NaN={cur['roe'].notna().sum()}")
    print(f"  Backup v3 roe:  NaN={old['roe'].isna().sum()}, non-NaN={old['roe'].notna().sum()}")
    if cur['roe'].notna().sum() > 0:
        print(f"  Current v3 first non-NaN roe: {cur[cur['roe'].notna()].head(3)[['date','roe']].to_string()}")
    if old['roe'].notna().sum() > 0:
        print(f"  Backup v3 first non-NaN roe: {old[old['roe'].notna()].head(3)[['date','roe']].to_string()}")

# Check if the two files are actually different
print("\n=== File comparison ===")
print(f"Current v3 roe NaN count: {v3['roe'].isna().sum()}")
print(f"Backup v3 roe NaN count: {v3_old['roe'].isna().sum()}")
print(f"Difference: {v3['roe'].isna().sum() - v3_old['roe'].isna().sum()}")

# Check 2023 specifically
v3_2023 = v3[v3["date"].dt.year == 2023]
v3_old_2023 = v3_old[v3_old["date"].dt.year == 2023]
print(f"\n2023 current NaN: {v3_2023['roe'].isna().sum()} / {len(v3_2023)} = {v3_2023['roe'].isna().mean()*100:.1f}%")
print(f"2023 backup NaN:  {v3_old_2023['roe'].isna().sum()} / {len(v3_old_2023)} = {v3_old_2023['roe'].isna().mean()*100:.1f}%")

# Are the roe values actually the same?
merged = v3_2023[["symbol", "date", "roe"]].merge(
    v3_old_2023[["symbol", "date", "roe"]], on=["symbol", "date"], suffixes=("_new", "_old")
)
diff = (merged["roe_new"] != merged["roe_old"]).sum()
print(f"\nRows where roe differs (new vs old): {diff} / {len(merged)}")
