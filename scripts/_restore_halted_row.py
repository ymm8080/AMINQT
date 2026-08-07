"""Restore 300333 (兆日科技) 2026-07-29 row from the tailfix backup into the panel.

Background: the 07-29 backfill anchored the symbol universe to the panel's max
date (07-31), which excludes 300333 (halted 07-30 after a +20% limit-up on 07-29).
Its 07-29 row was therefore dropped. This script re-inserts that single row from
the WORM backup so a re-run of `_daily_fetch.py 20260729` re-enriches it fully.

WORM: backs up the current panel before writing.
"""

import os
import shutil
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
BACKUP = r"D:\AMINQT\PARQUET\panel_full_enriched_v3_tailfix_20260803_002500.parquet"
SYMBOL = "300333"
TARGET_DATE = pd.Timestamp("2026-07-29")


def main():
    cur = pq.ParquetFile(PANEL)
    cur_schema = cur.schema_arrow
    cur_names = cur.schema_arrow.names
    print(f"Current panel: {cur.metadata.num_rows:,} rows, {len(cur_names)} cols")
    cur.close()  # Windows: open handle locks the file and breaks os.remove below

    # 1. Extract the target row from backup
    bdf = pq.read_table(
        BACKUP,
        filters=[("symbol", "=", SYMBOL), ("date", "=", TARGET_DATE)],
    ).to_pandas()
    if not len(bdf):
        print("ERROR: row not found in backup!")
        return
    print(
        f"Backup row found: {SYMBOL} @ {TARGET_DATE.date()} (close={bdf.iloc[0]['close']})"
    )

    # 2. Confirm it is NOT already in the panel
    cur_df = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
    exists = ((cur_df["symbol"] == SYMBOL) & (cur_df["date"] == TARGET_DATE)).sum()
    print(f"Row already in panel: {exists}")
    if exists:
        print("Nothing to do.")
        return

    # 3. Backup current panel (WORM)
    ts = time.strftime("%Y%m%d_%H%M%S")
    bpath = PANEL.replace(".parquet", f"_prerestorehaltsym_{ts}.parquet")
    shutil.copy2(PANEL, bpath)
    print(f"WORM backup: {bpath}")

    # 4. Read full panel, align schema, concat, write
    full = pq.read_table(PANEL).to_pandas()
    print(f"Full panel: {len(full):,} rows")
    row = bdf.iloc[[0]].copy()
    row = row[list(cur_names)]
    # ensure dtypes align with panel columns
    for c in cur_names:
        if c in row.columns:
            try:
                row[c] = row[c].astype(full[c].dtype)
            except Exception:
                pass
    merged = pd.concat([full, row], ignore_index=True)
    del full
    print(f"After concat: {len(merged):,} rows")

    tmp = PANEL + ".tmp"
    table = pa.Table.from_pandas(merged, schema=cur_schema, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    del merged, table

    # Atomic replace
    if os.path.exists(PANEL):
        os.remove(PANEL)
    os.rename(tmp, PANEL)
    print("Panel replaced.")

    # 5. Verify
    pf = pq.ParquetFile(PANEL)
    print(
        f"Panel now: {pf.metadata.num_rows:,} rows, {len(pf.schema_arrow.names)} cols"
    )
    check = pq.read_table(
        PANEL,
        filters=[("symbol", "=", SYMBOL), ("date", "=", TARGET_DATE)],
    ).to_pandas()
    print(
        f"Restored row present: {len(check)} | close={check.iloc[0]['close'] if len(check) else 'N/A'}"
    )


if __name__ == "__main__":
    main()
