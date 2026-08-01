# -*- coding: utf-8 -*-
"""Restore 6 holdertrade columns from data/ backup into the 96-col PARQUET v3 panel."""

import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
BACKUP = r"d:\AMINQT\AMINQT CODES\data\panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp"

HOLDER_COLS = [
    "holder_count",
    "sh_change_vol",
    "sh_change_amt",
    "sh_change_amt_total",
    "sh_net_change_sign",
    "sh_net_sign",
]


def main():
    # Read current panel schema
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    cur_names = schema.names
    n_rows = pf.metadata.num_rows
    print(f"Current panel: {len(cur_names)} cols, {n_rows:,} rows")

    # Check which holder cols are missing
    missing = [c for c in HOLDER_COLS if c not in cur_names]
    present = [c for c in HOLDER_COLS if c in cur_names]
    print(f"  Missing: {missing}")
    if present:
        print(f"  Already present: {present}")

    if not missing:
        print("All holder columns already present. Nothing to do.")
        return

    # Read holder data from backup
    print(f"\nReading holder cols from backup: {BACKUP}")
    bpf = pq.ParquetFile(BACKUP)
    b_schema = bpf.schema_arrow
    b_names = b_schema.names
    avail = [c for c in missing if c in b_names]
    print(f"  Available in backup: {avail}")

    if not avail:
        print("ERROR: No holder columns found in backup!")
        return

    # Read holder data
    holder_df = pq.read_table(BACKUP, columns=["symbol", "date"] + avail).to_pandas()
    print(f"  Holder data: {len(holder_df):,} rows")
    print("  Non-null counts:")
    for c in avail:
        print(f"    {c}: {holder_df[c].notna().sum():,}")

    # Get field types from backup schema
    holder_fields = {}
    for c in avail:
        idx = b_names.index(c)
        holder_fields[c] = b_schema.field(idx)

    # Build new schema: insert holder cols after the last existing col
    # Actually, just append them at the end
    new_fields = list(schema)
    for c in avail:
        new_fields.append(holder_fields[c])
    new_schema = pa.schema(new_fields)
    new_names = cur_names + avail
    print(f"\nNew schema: {len(new_names)} cols")

    # Close readers
    pf.close()
    bpf.close()

    # Read full current panel
    print("\nReading full current panel...")
    full_df = pq.read_table(PANEL).to_pandas()
    print(f"  Shape: {full_df.shape}")

    # Merge holder data
    print("Merging holder columns...")
    full_df = full_df.merge(holder_df, on=["symbol", "date"], how="left")
    print(f"  After merge: {full_df.shape}")

    # Reorder: original cols + holder cols
    full_df = full_df[new_names]
    print(f"  Final shape: {full_df.shape}")

    del holder_df
    gc.collect()

    # Write
    print("\nWriting parquet...")
    table = pa.Table.from_pandas(full_df, schema=new_schema, preserve_index=False)
    pq.write_table(table, TMP, compression="snappy")
    del full_df, table
    gc.collect()

    # Atomic replace
    print("Replacing panel file...")
    os.remove(PANEL)
    os.rename(TMP, PANEL)

    # Verify
    pf2 = pq.ParquetFile(PANEL)
    print(f"\nDone: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    # Check holder col fill rates
    verify = pq.read_table(PANEL, columns=["symbol", "date"] + avail).to_pandas()
    for c in avail:
        print(f"  {c}: {verify[c].notna().sum():,} non-null")
    pf2.close()


if __name__ == "__main__":
    main()
