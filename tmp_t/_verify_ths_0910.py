# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

v = pd.read_parquet(
    r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
    columns=["symbol", "date", "ths_bull", "ths_bear", "ths_bear_pool",
             "ths_bull_tech_n"],
    filters=[("date", "=", pd.Timestamp("2026-09-10"))])
nb = int((v["ths_bull"] == 1.0).sum())
nr = int((v["ths_bear"] == 1.0).sum())
print(f"0910 rows={len(v)} ths_bull=1: {nb}  ths_bear=1: {nr}  "
      f"bear_pool_day={v['ths_bear_pool'].dropna().unique().tolist()[:1]}")
