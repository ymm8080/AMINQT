# -*- coding: utf-8 -*-
"""Rename benefit_part -> winner_ratio in V3 panel parquet."""

import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp"
OLD = "benefit_part"
NEW = "winner_ratio"


def main():
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    old_names = schema.names
    n_rows = pf.metadata.num_rows
    print(f"Current: {len(old_names)} cols, {n_rows:,} rows")
    assert OLD in old_names, f"{OLD} not in panel!"

    new_names = [NEW if n == OLD else n for n in old_names]
    new_fields = []
    for i, name in enumerate(new_names):
        if name == NEW:
            new_fields.append(pa.field(NEW, schema.field(i).type))
        else:
            new_fields.append(schema.field(i))
    new_schema = pa.schema(new_fields)
    pf.close()

    print("Reading full panel...")
    df = pq.read_table(PANEL).to_pandas()
    df = df.rename(columns={OLD: NEW})
    df = df[new_names]
    print(f"  Shape: {df.shape}")

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
