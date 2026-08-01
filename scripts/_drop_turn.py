# -*- coding: utf-8 -*-
"""Drop 'turn' column from V3 panel (100 -> 99 cols). Keep free_float_turnover_rate only."""
import os, gc
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp"
DROP = ["turn"]

def main():
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    old = schema.names
    n = pf.metadata.num_rows
    print(f"Current: {len(old)} cols, {n:,} rows")
    assert "turn" in old, "turn not in panel!"
    new_names = [c for c in old if c not in DROP]
    new_fields = [schema.field(old.index(c)) for c in new_names]
    new_schema = pa.schema(new_fields)
    pf.close()
    print(f"New: {len(new_names)} cols (dropped: {DROP})")
    print("Reading full panel...")
    df = pq.read_table(PANEL).to_pandas()
    df = df.drop(columns=DROP)[new_names]
    print(f"  Final: {df.shape}")
    print("Writing...")
    t = pa.Table.from_pandas(df, schema=new_schema, preserve_index=False)
    pq.write_table(t, TMP, compression="snappy")
    del df, t; gc.collect()
    os.remove(PANEL)
    os.rename(TMP, PANEL)
    pf2 = pq.ParquetFile(PANEL)
    print(f"Done: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    pf2.close()

if __name__ == "__main__":
    main()
