#!/usr/bin/env python3
"""检查 v3 最新状态和数据覆盖率."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

V3_PATH = "data/panel_full_enriched_v3.parquet"

# 读取 parquet schema 和行数
meta = pq.read_metadata(V3_PATH)
print(f"行数: {meta.num_rows}")
print(f"列数: {meta.num_columns}")

# 分批读取列名
table = pq.read_table(V3_PATH, columns=[])
cols = table.schema.names
print(f"列名: {cols}")
print()

# 抽样检查最后 5 个交易日, 查看各列 NaN 率
df = pd.read_parquet(V3_PATH)
dates = sorted(df["date"].unique())
last_5 = dates[-5:]
print(f"最新 5 个交易日: {last_5}")
print()

# 各列非空率
total = len(df)
print(f"{'列名':<30} {'非空行':>10} {'非空率':>10} {'最新日非空':>10}")
print("-" * 65)
for c in df.columns:
    non_null = df[c].notna().sum()
    ratio = non_null / total * 100
    latest = df.loc[df["date"] == last_5[-1], c].notna().sum() if last_5[-1] in df["date"].values else 0
    if ratio < 100:
        print(f"{c:<30} {non_null:>10} {ratio:>9.1f}% {latest:>10}")

print(f"\n总行数: {total}")
print(f"总股票数: {df['symbol'].nunique()}")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")