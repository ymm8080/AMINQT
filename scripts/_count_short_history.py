# -*- coding: utf-8 -*-
"""Count panel rows belonging to symbols with short trading history (< 150 days in panel)."""

import sys
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
pf = pq.ParquetFile(PANEL)
df = pf.read(columns=["symbol", "date"]).to_pandas()
n_total = len(df)

cnt = df.groupby("symbol")["date"].count()
total_syms = len(cnt)

for th in (50, 100, 150, 200, 250):
    sub = cnt[cnt < th]
    n_rows = cnt.index.isin(sub.index).sum()
    rows_removed = n_total - df[df["symbol"].isin(cnt[cnt >= th].index)].shape[0]
    # rows belonging to short-history symbols
    syms_short = sub.index
    short_rows = df[df["symbol"].isin(syms_short)].shape[0]
    print(
        f"history<{th:3d} days: symbols={len(sub):4d} ({len(sub) / total_syms * 100:.1f}%)  "
        f"rows={short_rows:>8,d} ({short_rows / n_total * 100:.2f}%)"
    )

# distribution of min row-count (shortest histories)
print("\nshortest 10 symbols by row count:")
for s, c in cnt.nsmallest(10).items():
    print(f"  {s}  {c} rows")
