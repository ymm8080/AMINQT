#!/usr/bin/env python3
"""Simple fetch of fina_indicator for full range 2023-01-01 ~ 2026-07-28."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline1.data_supply import DataSupplyChain
import pandas as pd

panel = pd.read_parquet("data/panel_full_enriched_v3.parquet", columns=["date"])
start = panel["date"].min().strftime("%Y%m%d")
end = panel["date"].max().strftime("%Y%m%d")
print(f"Fetching fina_indicator: {start} ~ {end}")

supply = DataSupplyChain()
df = supply.fetch_fina_indicator(start_date=start, end_date=end, refresh=True)
print(f"Done: {len(df)} rows, {df['symbol'].nunique()} stocks")
