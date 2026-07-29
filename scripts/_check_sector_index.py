"""检查 sector_index 缓存和 fetch 方法."""
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from app.pipeline1.data_supply import DataSupplyChain

# Check supply_cache sector_index files
alt_dir = 'data/supply_cache/alt_data/sector_index'
print("=== sector_index cache files ===")
for f in sorted(os.listdir(alt_dir)):
    fp = os.path.join(alt_dir, f)
    df = pd.read_parquet(fp)
    print(f'{f}: shape={df.shape}, cols={list(df.columns)}')
    print(f'  date: {df["date"].min()} ~ {df["date"].max()}')
    print(f'  index_codes: {df["index_code"].nunique() if "index_code" in df.columns else "N/A"}')
    print()

# Try fetch_sector_index
supply = DataSupplyChain()
print("=== fetch_sector_index() ===")
try:
    si = supply.fetch_sector_index(start_date='20240101', end_date='20260728')
    print(f'Shape: {si.shape}')
    print(f'Cols: {list(si.columns)}')
    print(f'Date: {si["date"].min()} ~ {si["date"].max()}')
    print(f'Index codes: {si["index_code"].nunique() if "index_code" in si.columns else "N/A"}')
except Exception as e:
    print(f'Error: {e}')

print()

# Check enrich_parts sector_index
print("=== enrich_parts sector_index.parquet ===")
sector_ep = pd.read_parquet('data/enrich_parts/sector_index.parquet')
print(f'Shape: {sector_ep.shape}')
print(f'Cols: {list(sector_ep.columns)}')