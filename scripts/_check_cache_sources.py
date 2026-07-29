#!/usr/bin/env python3
"""检查所有缓存目录的可填充数据源."""
import os, sys
from pathlib import Path

ROOT = Path("d:/AMINQT/AMINQT CODES")

# 1) supply_cache/alt_data
alt_dir = ROOT / "data/supply_cache/alt_data"
if alt_dir.exists():
    print("=== supply_cache/alt_data ===")
    for src in sorted(os.listdir(alt_dir)):
        src_dir = alt_dir / src
        if src_dir.is_dir():
            parquet_files = list(src_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
            print(f"  {src:<25} {len(parquet_files):>4} 个文件, {total_size:>6.1f} MB")
else:
    print("=== supply_cache/alt_data 不存在 ===")

# 2) enrich_parts
enrich_dir = ROOT / "data/enrich_parts"
if enrich_dir.exists():
    print("\n=== enrich_parts ===")
    for src in sorted(os.listdir(enrich_dir)):
        src_dir = enrich_dir / src
        if src_dir.is_dir():
            parquet_files = list(src_dir.glob("*.parquet"))
            total_size = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
            print(f"  {src:<25} {len(parquet_files):>4} 个文件, {total_size:>6.1f} MB")

# 3) 列举 v3 应该有的数据源列前缀
print("\n=== v3 期望数据源列前缀 ===")
expected = {
    "daily_basic": ["turnover_rate_f", "pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv", "circ_mv"],
    "stk_limit": ["up_limit_raw", "down_limit_raw"],
    "margin": ["margin_", "short_"],
    "northbound": ["north_"],
    "lhb": ["lhb_"],
    "fina_indicator": ["roe", "roa", "gross_margin", "net_margin"],
    "holdernumber": ["holder_count"],
    "holdertrade": ["sh_"],
    "cyq_tushare": ["benefit_part", "pct_", "cost_"],
    "sector_index": ["sw_"],
}
for src, prefixes in expected.items():
    print(f"  {src:<25} -> {prefixes}")