"""Quick inspection of alt data cache files."""
import pandas as pd, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Margin
mg = pd.read_parquet("data/supply_cache/alt_data/margin/20240102_20260727.parquet")
print("=== MARGIN ===")
print(f"Rows: {len(mg)}, Symbols: {mg['symbol'].nunique()}, Dates: {mg['date'].nunique()}")
print(f"Date range: {mg['date'].min()} ~ {mg['date'].max()}")
print(f"Columns: {list(mg.columns)}")
print()

# LHB
lhb = pd.read_parquet("data/supply_cache/alt_data/lhb/all_20240102_20260727.parquet")
print("=== LHB ===")
print(f"Rows: {len(lhb)}, Symbols: {lhb['symbol'].nunique()}, Dates: {lhb['date'].nunique()}")
print(f"Date range: {lhb['date'].min()} ~ {lhb['date'].max()}")
print(f"Columns: {list(lhb.columns)}")
print()

# Northbound
nb = pd.read_parquet("data/supply_cache/alt_data/northbound/all_20240102_20260727.parquet")
print("=== NORTHBOUND ===")
print(f"Rows: {len(nb)}, Dates: {nb['date'].nunique()}")
print(f"Date range: {nb['date'].min()} ~ {nb['date'].max()}")
print(f"Columns: {list(nb.columns)}")
print(f"Has symbol column: {'symbol' in nb.columns}")
print()

# Holdernumber
hn_dir = "data/supply_cache/alt_data/holdernumber"
hn_files = [f for f in os.listdir(hn_dir) if f.endswith(".parquet")]
print(f"=== HOLDERNUMBER: {len(hn_files)} cache files ===")
hn = pd.read_parquet(os.path.join(hn_dir, hn_files[0]))
print(f"Sample file: {hn_files[0]}")
print(f"Rows: {len(hn)}, Columns: {list(hn.columns)}")
print()

# Holdertrade
ht_dir = "data/supply_cache/alt_data/holdertrade"
ht_files = [f for f in os.listdir(ht_dir) if f.endswith(".parquet")]
print(f"=== HOLDERTRADE: {len(ht_files)} cache files ===")
if ht_files:
    ht = pd.read_parquet(os.path.join(ht_dir, ht_files[0]))
    print(f"Sample file: {ht_files[0]}")
    print(f"Rows: {len(ht)}, Columns: {list(ht.columns)}")
else:
    print("No files!")
