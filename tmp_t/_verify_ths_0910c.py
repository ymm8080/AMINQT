# -*- coding: utf-8 -*-
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd

v = pd.read_parquet(
    r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
    columns=["symbol", "date"],
    filters=[("symbol", "=", "600818")])
v["date"] = pd.to_datetime(v["date"])
last5 = v.nlargest(5, "date")
print(f"600818 rows={len(v)} last_dates={last5['date'].dt.date.tolist()}")
