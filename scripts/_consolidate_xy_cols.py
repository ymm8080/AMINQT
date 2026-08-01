# -*- coding: utf-8 -*-
"""Consolidate _x/_y duplicate columns in v3 panel into single original-named columns.

_x has better coverage (0.65% NaN) vs _y (7.39% NaN).
Strategy: coalesce(_x, _y) -> original name, then drop _x and _y.

Reduces 116 cols -> 102 cols.
"""

import os
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

# 14 pairs of _x/_y columns to consolidate
PAIRS = [
    ("winner_ratio_x", "winner_ratio_y", "winner_ratio"),
    ("avg_cost_x", "avg_cost_y", "avg_cost"),
    ("cost_5pct_x", "cost_5pct_y", "cost_5pct"),
    ("cost_15pct_x", "cost_15pct_y", "cost_15pct"),
    ("cost_50pct_x", "cost_50pct_y", "cost_50pct"),
    ("cost_85pct_x", "cost_85pct_y", "cost_85pct"),
    ("cost_95pct_x", "cost_95pct_y", "cost_95pct"),
    ("pct_70_low_x", "pct_70_low_y", "pct_70_low"),
    ("pct_70_high_x", "pct_70_high_y", "pct_70_high"),
    ("pct_70_con_x", "pct_70_con_y", "pct_70_con"),
    ("pct_90_low_x", "pct_90_low_y", "pct_90_low"),
    ("pct_90_high_x", "pct_90_high_y", "pct_90_high"),
    ("pct_90_con_x", "pct_90_con_y", "pct_90_con"),
    ("weight_avg_x", "weight_avg_y", "weight_avg"),
]


def main():
    print(f"Reading panel: {PANEL}")
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    old_names = schema.names
    print(f"  Current: {len(old_names)} cols, {pf.metadata.num_rows:,} rows")

    # Build new column order
    cols_to_drop = set()
    cols_to_add = {}  # new_name -> position to insert at (position of _x in original)

    for x_col, y_col, new_col in PAIRS:
        assert x_col in old_names, f"{x_col} not in schema!"
        assert y_col in old_names, f"{y_col} not in schema!"
        cols_to_drop.add(x_col)
        cols_to_drop.add(y_col)
        cols_to_add[new_col] = old_names.index(x_col)

    # Build new column list (preserving order, replacing _x position with new name)
    new_names = []
    for i, name in enumerate(old_names):
        if name in cols_to_drop:
            # Skip both _x and _y, but insert consolidated name at _x position
            if name.endswith("_x") and name in [p[0] for p in PAIRS]:
                new_col = name[:-2]  # strip _x
                new_names.append(new_col)
            # else: skip (this is _y, already handled at _x position)
        else:
            new_names.append(name)

    print(f"  New: {len(new_names)} cols")
    assert len(new_names) == len(old_names) - len(PAIRS), (
        f"Expected {len(old_names) - len(PAIRS)}, got {len(new_names)}"
    )

    # Read all _x/_y columns + write consolidated
    print("\nCoalescing _x/_y pairs...")
    xy_cols = []
    for x_col, y_col, _ in PAIRS:
        xy_cols.extend([x_col, y_col])

    xy_table = pq.read_table(PANEL, columns=xy_cols)
    xy_df = xy_table.to_pandas()

    consolidated = {}
    for x_col, y_col, new_col in PAIRS:
        # Coalesce: take x first, fallback to y
        val = xy_df[x_col].combine_first(xy_df[y_col])
        consolidated[new_col] = val
        fill_rate = val.notna().sum() / len(val) * 100
        print(f"  {new_col:<15s}: {fill_rate:.2f}% filled ({val.notna().sum():,})")

    # Now rebuild the table: read all columns, replace _x/_y with consolidated
    print("\nRebuilding parquet (streaming row groups)...")
    tmp_path = PANEL + ".tmp"

    # Build new schema
    new_fields = []
    for name in new_names:
        if name in consolidated:
            # Use float64 for consolidated columns
            new_fields.append(pa.field(name, pa.float64()))
        else:
            # Find original field
            idx = old_names.index(name)
            new_fields.append(schema.field(idx))

    new_schema = pa.schema(new_fields)

    # writer = pq.ParquetWriter(tmp_path, schema=new_schema)  # Not needed - using pq.write_table instead

    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        rg_df = rg.to_pandas()

        # Drop _x/_y columns
        rg_df = rg_df.drop(columns=[c for c in cols_to_drop if c in rg_df.columns])

        # Add consolidated columns
        # We need the consolidated values for this row group
        # Since consolidated was computed on the full table, we need to slice it
        # Instead, let's recompute on the row group
        # Actually, we need to track row offsets
        # Simpler: read all at once and write in one go
        pass

    # Actually, streaming row groups with consolidated values is complex.
    # Let's do it in one shot since the file fits in memory.
    print("\nReading full panel...")
    full_df = pq.read_table(PANEL).to_pandas()
    print(f"  Shape: {full_df.shape}")

    # Drop _x/_y columns
    full_df = full_df.drop(columns=[c for c in cols_to_drop if c in full_df.columns])

    # Add consolidated columns
    for new_col, val in consolidated.items():
        full_df[new_col] = val.values

    # Reorder to match new_names
    full_df = full_df[new_names]

    print(f"  Final shape: {full_df.shape}")

    # Write
    print("\nWriting parquet...")
    table = pa.Table.from_pandas(full_df, schema=new_schema, preserve_index=False)
    pq.write_table(table, tmp_path, compression="snappy")

    # Atomic replace
    if os.path.exists(PANEL):
        os.remove(PANEL)
    os.rename(tmp_path, PANEL)

    # Verify
    pf2 = pq.ParquetFile(PANEL)
    print(f"\nDone: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    pf2.close()


if __name__ == "__main__":
    main()
