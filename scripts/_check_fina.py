#!/usr/bin/env python3
"""检查 fina_indicator 缓存结构."""

import os

import pandas as pd

fina_dir = "data/supply_cache/alt_data/fina_indicator"
for f in sorted(os.listdir(fina_dir))[:3]:
    fp = os.path.join(fina_dir, f)
    df = pd.read_parquet(fp)
    print(f"  {f}: 列={df.columns.tolist()}, 行={len(df)}")
    print(f"    前2行:\n{df.head(2).to_string()}")
    print()

# 检查 holdernumber
print("=== holdernumber ===")
hn_dir = "data/supply_cache/alt_data/holdernumber"
for f in sorted(os.listdir(hn_dir))[:3]:
    fp = os.path.join(hn_dir, f)
    df = pd.read_parquet(fp)
    print(f"  {f}: 列={df.columns.tolist()}, 行={len(df)}")
    print(f"    前2行:\n{df.head(2).to_string()}")
    print()

# 检查 holdertrade
print("=== holdertrade ===")
ht_dir = "data/supply_cache/alt_data/holdertrade"
for f in sorted(os.listdir(ht_dir))[:3]:
    fp = os.path.join(ht_dir, f)
    df = pd.read_parquet(fp)
    print(f"  {f}: 列={df.columns.tolist()}, 行={len(df)}")
    print(f"    前2行:\n{df.head(2).to_string()}")
    print()

# 检查 northbound
print("=== northbound ===")
nb_dir = "data/supply_cache/alt_data/northbound"
for f in sorted(os.listdir(nb_dir))[:3]:
    fp = os.path.join(nb_dir, f)
    df = pd.read_parquet(fp)
    print(f"  {f}: 列={df.columns.tolist()}, 行={len(df)}")
    print(f"    前2行:\n{df.head(2).to_string()}")
    print()

# 检查 lhb
print("=== lhb ===")
lhb_dir = "data/supply_cache/alt_data/lhb"
for f in sorted(os.listdir(lhb_dir))[:3]:
    fp = os.path.join(lhb_dir, f)
    df = pd.read_parquet(fp)
    print(f"  {f}: 列={df.columns.tolist()}, 行={len(df)}")
    print(f"    前2行:\n{df.head(2).to_string()}")
    print()
