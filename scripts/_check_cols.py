#!/usr/bin/env python3
"""快速检查 v3 当前列名, 查找 close/amount 是否被重命名."""
import pandas as pd
df = pd.read_parquet("data/panel_full_enriched_v3.parquet")
cols = df.columns.tolist()
print(f"总列数: {len(cols)}")
# 找 close/amount 相关
for c in cols:
    if "close" in c.lower() or "amount" in c.lower() or c.endswith("_x") or c.endswith("_y"):
        print(f"  {c}")
print()
# 打印所有列
for i, c in enumerate(cols):
    print(f"  {i:>3}: {c}")