import os
import pandas as pd

path = "D:/AMINQT/AMINQT CODES/data/supply_cache/alt_data/fina_indicator/all_20240102_20260727.parquet"
if os.path.exists(path):
    df = pd.read_parquet(path)
    print(f"Output file exists! {len(df)} rows, {df['symbol'].nunique()} symbols")
else:
    print("Output file NOT yet created")
    cdir = "D:/AMINQT/AMINQT CODES/data/supply_cache/alt_data/fina_indicator"
    cached = [
        f for f in os.listdir(cdir) if f.endswith(".parquet") and "__20240102" in f
    ]
    print(f"Cache files with date range: {len(cached)}")
