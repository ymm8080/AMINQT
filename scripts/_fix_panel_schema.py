"""Consolidate _x/_y duplicate columns + remove holder columns from v3 panel.

Operations:
  1. Coalesce 14 _x/_y pairs -> original names (28 cols -> 14)
  2. Drop 6 holder columns: holder_count, sh_change_vol, sh_change_amt,
     sh_change_amt_total, sh_net_change_sign, sh_net_sign

Result: 116 cols -> 96 cols.
"""

import gc
import os

import pyarrow as pa
import pyarrow.parquet as pq

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
TMP = PANEL + ".tmp"

# 14 pairs of _x/_y columns to consolidate (x first, y fallback)
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

# Holder columns to remove
HOLDER_COLS = [
    "holder_count",
    "sh_change_vol",
    "sh_change_amt",
    "sh_change_amt_total",
    "sh_net_change_sign",
    "sh_net_sign",
]


def main():
    # Clean up old .tmp if exists
    if os.path.exists(TMP):
        os.remove(TMP)

    # Read schema
    pf = pq.ParquetFile(PANEL)
    schema = pf.schema_arrow
    old_names = schema.names
    n_rows = pf.metadata.num_rows
    print(f"Current: {len(old_names)} cols, {n_rows:,} rows")

    # Identify columns to drop
    cols_to_drop = set()
    x_col_set = set()
    for x_col, y_col, _ in PAIRS:
        assert x_col in old_names, f"{x_col} not in schema!"
        assert y_col in old_names, f"{y_col} not in schema!"
        cols_to_drop.add(x_col)
        cols_to_drop.add(y_col)
        x_col_set.add(x_col)

    for h in HOLDER_COLS:
        if h in old_names:
            cols_to_drop.add(h)
            print(f"  Will remove holder col: {h}")
        else:
            print(f"  WARNING: {h} not in panel (already removed?)")

    # Build new column list
    new_names = []
    for name in old_names:
        if name in cols_to_drop:
            if name in x_col_set:
                new_names.append(name[:-2])  # strip _x -> original
            # else: skip _y or holder
        else:
            new_names.append(name)

    print(f"New: {len(new_names)} cols")
    assert len(new_names) == len(old_names) - len(PAIRS) - len(HOLDER_COLS), (
        f"Expected {len(old_names) - len(PAIRS) - len(HOLDER_COLS)}, got {len(new_names)}"
    )

    # Read _x/_y columns for coalescing
    print("\nCoalescing _x/_y pairs...")
    xy_cols = []
    for x_col, y_col, _ in PAIRS:
        xy_cols.extend([x_col, y_col])
    xy_df = pq.read_table(PANEL, columns=xy_cols).to_pandas()

    consolidated = {}
    for x_col, y_col, new_col in PAIRS:
        val = xy_df[x_col].combine_first(xy_df[y_col])
        consolidated[new_col] = val
        fill_rate = val.notna().sum() / len(val) * 100
        print(f"  {new_col:<15s}: {fill_rate:.2f}% filled")

    del xy_df
    gc.collect()

    # Close reader before writing
    pf.close()

    # Build new schema
    new_fields = []
    for name in new_names:
        if name in consolidated:
            new_fields.append(pa.field(name, pa.float64()))
        else:
            idx = old_names.index(name)
            new_fields.append(schema.field(idx))
    new_schema = pa.schema(new_fields)

    # Read full panel
    print("\nReading full panel...")
    full_df = pq.read_table(PANEL).to_pandas()
    print(f"  Shape: {full_df.shape}")

    # Drop old columns
    drop_present = [c for c in cols_to_drop if c in full_df.columns]
    full_df = full_df.drop(columns=drop_present)

    # Add consolidated columns
    for new_col, val in consolidated.items():
        full_df[new_col] = val.values

    # Reorder
    full_df = full_df[new_names]
    print(f"  Final shape: {full_df.shape}")

    # Write
    print("\nWriting parquet...")
    table = pa.Table.from_pandas(full_df, schema=new_schema, preserve_index=False)
    pq.write_table(table, TMP, compression="snappy")
    del full_df, table
    gc.collect()

    # Atomic replace
    print("\nReplacing panel file...")
    os.remove(PANEL)
    os.rename(TMP, PANEL)

    # Verify
    pf2 = pq.ParquetFile(PANEL)
    print(f"\nDone: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
    pf2.close()


if __name__ == "__main__":
    main()
