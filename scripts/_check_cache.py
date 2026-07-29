"""检查缓存数据源的内容."""
import pandas as pd
import os

# Check enrich_parts
parts_dir = 'data/enrich_parts'
print("=== enrich_parts 缓存 ===")
for f in sorted(os.listdir(parts_dir)):
    fp = os.path.join(parts_dir, f)
    df = pd.read_parquet(fp)
    print(f'{f}: shape={df.shape}, cols={list(df.columns)[:15]}...')
    print(f'  date range: {df["date"].min()} ~ {df["date"].max()}')
    if "symbol" in df.columns:
        print(f'  symbols: {df["symbol"].nunique()}')
    # Count NaN
    nan_rates = (df.isna().mean() * 100).round(1)
    high_nan = nan_rates[nan_rates > 0].sort_values(ascending=False)
    if len(high_nan):
        print(f'  NaN rates: {dict(high_nan.head(5))}')
    print()

# Check supply_cache/alt_data
print("\n=== supply_cache/alt_data 缓存 ===")
alt_dir = 'data/supply_cache/alt_data'
for root, dirs, files in os.walk(alt_dir):
    rel = os.path.relpath(root, alt_dir)
    parquet_files = [f for f in files if f.endswith('.parquet')]
    if parquet_files:
        print(f'{rel}: {len(parquet_files)} files')
        for f in sorted(parquet_files)[:3]:
            fp = os.path.join(root, f)
            sz = os.path.getsize(fp) / 1024
            print(f'  - {f} ({sz:.0f} KB)')
        if len(parquet_files) > 3:
            print(f'  ... and {len(parquet_files)-3} more')
        # Check first file for schema
        first = os.path.join(root, sorted(parquet_files)[0])
        try:
            df = pd.read_parquet(first)
            print(f'  schema: {list(df.columns)[:8]}...')
            print(f'  rows: {len(df)}')
        except:
            pass
        print()