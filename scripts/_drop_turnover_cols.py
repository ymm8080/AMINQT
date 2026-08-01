# -*- coding: utf-8 -*-
"""Drop turnover_rate and turnover_rate_f columns from V3 panel (102 -> 100 cols).

free_float_turnover_rate (= turnover_rate_f) and turn (= turnover_rate) are kept.
"""

import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp"
DROP_COLS = ["turnover_rate", "turnover_rate_f"]


def main():
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    old_names = schema.names
    n_rows = pf.metadata.num_rows
    print(f"Current: {len(old_names)} cols, {n_rows:,} rows")

    # Verify columns exist
    for c in DROP_COLS:
        assert c in old_names, f"{c} not in panel!"

    # Build new column list
    new_names = [n for n in old_names if n not in DROP_COLS]
    print(f"New: {len(new_names)} cols (dropped: {DROP_COLS})")

    # Build new schema
    new_fields = [schema.field(old_names.index(n)) for n in new_names]
    new_schema = pa.schema(new_fields)

    pf.close()

    # Read full panel, drop columns, write
    print("Reading full panel...")
    full_df = pq.read_table(PANEL).to_pandas()
    full_df = full_df.drop(columns=DROP_COLS)
    full_df = full_df[new_names]
    print(f"  Final shape: {full_df.shape}")

    print("Writing parquet...")
    table = pa.Table.from_pandas(full_df, schema=new_schema, preserve_index=False)
    pq.write_table(table, TMP, compression="snappy")
    del full_df, table
    gc.collect()

    print("Replacing panel file...")
    os.remove(PANEL)
    os.rename(TMP, PANEL)

    # Verify
    pf2 = pq.ParquetFile(PANEL)
    print(f"Done: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    pf2.close()


if __name__ == "__main__":
    main()
