# -*- coding: utf-8 -*-
"""从当前生产面板重建迷你面板(symbol+date, 含0910行) — enrich专用."""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
MINI = r"D:/AMINQT/PARQUET/_panel_mini_0910.parquet"

pf = pd.read_parquet(PANEL, columns=["symbol", "date"])
print(f"[src] {len(pf)} rows, max_date={pf['date'].max().date()}")
pf.to_parquet(MINI, index=False)
print(f"[done] {MINI}")
