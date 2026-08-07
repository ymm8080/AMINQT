import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
cols = ["symbol", "date", "holder_count", "sh_change_vol", "sh_change_amt"]
df = pq.read_table(PANEL, columns=cols).to_pandas()
hist = df[df["date"] < pd.Timestamp("20260731")]
print(f"Total rows: {len(df):,}")
print(f"History (<20260731): {len(hist):,}")
for c in ["holder_count", "sh_change_vol", "sh_change_amt"]:
    print(f"  {c}: {hist[c].notna().sum():,} non-null in history")
# Check last known value per stock
last = hist.sort_values("date").drop_duplicates(subset=["symbol"], keep="last")
print(f"Last known per stock: {len(last):,}")
for c in ["holder_count", "sh_change_vol", "sh_change_amt"]:
    print(f"  {c}: {last[c].notna().sum():,} non-null")
