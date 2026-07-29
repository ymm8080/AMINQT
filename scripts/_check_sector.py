#!/usr/bin/env python3
"""检查行业板块数据: v3 中的 industry 列 + sector_index 缓存详情."""

import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

V3_PATH = "data/panel_full_enriched_v3.parquet"

# 1. 检查 v3 中 industry 相关列
df = pd.read_parquet(V3_PATH)
print("=== v3 中 industry/sector 相关列 ===")
ind_cols = [
    c
    for c in df.columns
    if any(k in c.lower() for k in ["industr", "sector", "sw_", "board"])
]
print(f"列: {ind_cols}")
print()

# industry 列分布
if "industry" in df.columns:
    print("=== industry 列分布 ===")
    ind_counts = df.groupby("date")["industry"].apply(
        lambda x: x.notna().sum() / len(x) * 100
    )
    print("  非空率 (最新5日):")
    for d, v in ind_counts.tail(5).items():
        print(f"    {d.date()}: {v:.1f}%")
    print(f"  唯一行业数: {df['industry'].nunique()}")
    print(f"  前10行业: {df['industry'].value_counts().head(10).to_dict()}")
    print()

# board 列
if "board" in df.columns:
    print("=== board 列分布 ===")
    print(f"  {df['board'].value_counts().to_dict()}")
    print()

# sw_ret_1d 覆盖
if "sw_ret_1d" in df.columns:
    print("=== sw_ret_1d 覆盖 ===")
    sw_by_date = df.groupby("date")["sw_ret_1d"].apply(lambda x: x.notna().sum())
    print("  最新5日非空数:")
    for d, v in sw_by_date.tail(5).items():
        print(f"    {d.date()}: {v}")
    print()

# 2. 检查 sector_index 缓存详情
print("\n=== sector_index 缓存详情 ===")
sec_dir = "data/supply_cache/alt_data/sector_index"
for f in sorted(os.listdir(sec_dir)):
    fp = os.path.join(sec_dir, f)
    sdf = pd.read_parquet(fp)
    print(f"\n  文件: {f}")
    print(f"  行数: {len(sdf)}, 列: {sdf.columns.tolist()}")
    print(
        f"  日期范围: {sdf['date'].min() if 'date' in sdf.columns else '?'} ~ {sdf['date'].max() if 'date' in sdf.columns else '?'}"
    )
    if "index_name" in sdf.columns:
        print(f"  指数名称 (前10): {sdf['index_name'].unique()[:10].tolist()}")
    if "ret_pct" in sdf.columns:
        print(f"  ret_pct 非空: {sdf['ret_pct'].notna().sum()}/{len(sdf)}")
    print("  前3行:")
    print(sdf.head(3).to_string())
    print()
