"""Quick analysis of panel symbols vs fina_indicator cache coverage."""
import pandas as pd
import os
import re

panel = pd.read_parquet("D:/AMINQT/AMINQT CODES/data/panel_3y.parquet")
panel_symbols = set(panel["symbol"].unique())
print(f"Panel symbols: {len(panel_symbols)}")
print(f"Panel date range: {panel['date'].min()} ~ {panel['date'].max()}")

cache_dir = "D:/AMINQT/AMINQT CODES/data/supply_cache/alt_data/fina_indicator"
cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".parquet")]
print(f"Total cache files: {len(cache_files)}")

cached_tscodes = set()
for f in cache_files:
    m = re.match(r"(\d{6}\.(SZ|SH))__", f)
    if m:
        cached_tscodes.add(m.group(1))

cached_symbols = set()
missing_symbols = set()
for sym in panel_symbols:
    if sym.startswith(("0","3","1")):
        tsc = f"{sym}.SZ"
    else:
        tsc = f"{sym}.SH"
    if tsc in cached_tscodes:
        cached_symbols.add(sym)
    else:
        missing_symbols.add(sym)

print(f"Cached symbols: {len(cached_symbols)}")
print(f"Missing symbols: {len(missing_symbols)}")
if missing_symbols:
    print(f"Missing (first 20): {sorted(missing_symbols)[:20]}")

print("\n--- Sample cache data ---")
for f in sorted(os.listdir(cache_dir))[:3]:
    df = pd.read_parquet(os.path.join(cache_dir, f))
    print(f"{f}: {len(df)} rows, announce {df['announce_date'].min()} ~ {df['announce_date'].max()}, "
          f"report_period {df['report_period'].min()} ~ {df['report_period'].max()}")
