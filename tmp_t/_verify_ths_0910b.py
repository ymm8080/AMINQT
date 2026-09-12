# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

src = pd.read_parquet(
    r"D:/AMINQT/AMINQT CODES/data/supply_cache/ths_signal/bull_20260910.parquet")
codes = set(src["股票代码"].astype(str).str.zfill(6))
v = pd.read_parquet(
    r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
    columns=["symbol", "date", "ths_bull"],
    filters=[("date", "=", pd.Timestamp("2026-09-10"))])
inpanel = set(v.loc[v["ths_bull"] == 1.0, "symbol"].astype(str))
missing = sorted(codes - inpanel)
print(f"src={len(codes)} in_panel={len(inpanel)} missing={missing}")
