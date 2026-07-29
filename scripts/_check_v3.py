"""快速检查 v3 面板状态."""
import pandas as pd
import numpy as np
import os

f = 'data/panel_full_enriched_v3.parquet'
print(f'Exists: {os.path.exists(f)}')
print(f'Size: {os.path.getsize(f) / 1024 / 1024:.1f} MB')

v3 = pd.read_parquet(f)
print(f'Shape: {v3.shape}')
print(f'Symbols: {v3["symbol"].nunique()}')
print(f'Date range: {v3["date"].min()} ~ {v3["date"].max()}')
print(f'Columns: {len(v3.columns)}')

nan_rates = (v3.isna().mean() * 100).round(1).sort_values(ascending=False)
print('\n=== 所有列 NaN 率 ===')
for col, rate in nan_rates.items():
    marker = ' <<<' if rate > 50 else ''
    print(f'  {col:40s}: {rate:>6.1f}%{marker}')