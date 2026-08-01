# -*- coding: utf-8 -*-
"""Restore holder_count column from backup into the 98-col panel."""
import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
BACKUP = r"data\panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp"

def main():
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    cur = schema.names
    print(f"Panel: {len(cur)} cols, {pf.metadata.num_rows:,} rows")
    pf.close()

    bpf = pq.ParquetFile(BACKUP)
    b_schema = bpf.schema_arrow
    b_names = b_schema.names
    bpf.close()

    # Get holder_count field type from backup
    hc_idx = b_names.index("holder_count")
    hc_field = b_schema.field(hc_idx)

    # Build new schema
    new_fields = list(schema) + [hc_field]
    new_schema = pa.schema(new_fields)
    new_names = cur + ["holder_count"]
    print(f"New: {len(new_names)} cols")

    # Read holder_count from backup
    print("Reading holder_count from backup...")
    hc = pq.read_table(BACKUP, columns=["symbol", "date", "holder_count"]).to_pandas()
    print(f"  holder_count: {hc['holder_count'].notna().sum():,} non-null")

    # Read full panel
    print("Reading full panel...")
    df = pq.read_table(PANEL).to_pandas()
    print(f"  Shape: {df.shape}")

    # Merge
    df = df.merge(hc, on=["symbol", "date"], how="left")
    df = df[new_names]
    print(f"  Final: {df.shape}")

    # Write
    print("Writing...")
    t = pa.Table.from_pandas(df, schema=new_schema, preserve_index=False)
    pq.write_table(t, TMP, compression="snappy")
    del df, t
    gc.collect()

    os.remove(PANEL)
    os.rename(TMP, PANEL)

    pf2 = pq.ParquetFile(PANEL)
    print(f"Done: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    pf2.close()

if __name__ == "__main__":
    main()
