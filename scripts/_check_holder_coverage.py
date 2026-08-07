"""Measure coverage (non-null rate) of holder-related columns in current V3 panel."""

import os
import sys

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
print("panel mtime:", os.path.getmtime(PANEL))

pf = pq.ParquetFile(PANEL)
cols = pf.schema.names

holder_like = [
    c
    for c in cols
    if "holder" in c.lower() or c.startswith("sh_") or "holder_evt" in c.lower()
]
print(f"rows={pf.metadata.num_rows}  total_cols={len(cols)}")
print("holder-related cols:", holder_like)

# read only those columns to compute coverage
if holder_like:
    df = pf.read(columns=holder_like).to_pandas()
    for c in holder_like:
        nn = df[c].notna().mean()
        print(
            f"  {c:32s} coverage = {nn * 100:6.2f}%   nonnull={df[c].notna().sum():>9,d}"
        )
