# -*- coding: utf-8 -*-
"""Replace SW code columns with SW name columns in V3 panel.

Remove: sw_l1_code, sw_l2_code, sw_l3_code
Add:    sw_l1_name, sw_l2_name, sw_l3_name (string)

Usage: python scripts/add_sw_cols_to_panel.py
"""

import os
import shutil
import sys
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import data_others_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
SW_CSV = str(data_others_path("data/processed/sw_stock_classification.csv"))

NEW_COLS = ["sw_l1_name", "sw_l2_name", "sw_l3_name"]
OLD_COLS_TO_REMOVE = [
    "sw_l1_code",
    "sw_l2_code",
    "sw_l3_code",
    "sw_index_close",
    "sw_index_vol",
    "sw_ret_1d",
]


def main():
    if not os.path.exists(PANEL):
        logger.error(f"Panel not found: {PANEL}")
        sys.exit(1)
    if not os.path.exists(SW_CSV):
        logger.error(f"SW CSV not found: {SW_CSV}")
        sys.exit(1)

    # ── 1. Load SW classification ──
    logger.info("Loading SW classification...")
    sw_df = pd.read_csv(SW_CSV, encoding="utf-8-sig", dtype=str)
    sw_map = {}
    for _, row in sw_df.iterrows():
        tc = str(row.get("ts_code", "")).strip()
        if tc:
            sw_map[tc] = {
                "sw_l1_name": row.get("sw_l1_name", ""),
                "sw_l2_name": row.get("sw_l2_name", ""),
                "sw_l3_name": row.get("sw_l3_name", ""),
            }
    logger.info(f"  SW CSV: {len(sw_map)} stocks")

    # ── 2. Build new schema: remove old cols, add new cols ──
    old_schema = pq.read_schema(PANEL)
    old_names = old_schema.names

    keep_fields = [f for f in old_schema if f.name not in OLD_COLS_TO_REMOVE]
    keep_names = [f.name for f in keep_fields]
    removed = [c for c in OLD_COLS_TO_REMOVE if c in old_names]
    logger.info(f"Removing: {removed}")

    # Find insert position: after 'industry' if present, else end
    insert_idx = len(keep_fields)
    if "industry" in keep_names:
        insert_idx = keep_names.index("industry") + 1

    # Build new field list
    new_fields = list(keep_fields[:insert_idx])
    for c in NEW_COLS:
        if c not in keep_names:
            new_fields.append(pa.field(c, pa.string()))
    new_fields.extend(keep_fields[insert_idx:])

    new_schema = pa.schema(new_fields)
    logger.info(f"Schema: {len(old_names)} -> {len(new_schema)} cols")
    logger.info(f"Added: {[c for c in NEW_COLS if c not in old_names]}")

    # ── 3. Stream rewrite ──
    logger.info("Rewriting panel...")
    pf = pq.ParquetFile(PANEL)
    tmp_path = PANEL + ".tmp"
    writer = pq.ParquetWriter(tmp_path, schema=new_schema)

    total_rows = 0
    new_col_names = [f.name for f in new_schema]

    for rg_idx in range(pf.metadata.num_row_groups):
        rg = pf.read_row_group(rg_idx)
        rg_df = rg.to_pandas()

        # Drop old columns
        for c in OLD_COLS_TO_REMOVE:
            if c in rg_df.columns:
                rg_df = rg_df.drop(columns=[c])

        # Build ts_code from symbol if needed
        if "ts_code" in rg_df.columns:
            ts_codes = rg_df["ts_code"].astype(str).tolist()
        elif "symbol" in rg_df.columns:
            ts_codes = (
                rg_df["symbol"]
                .apply(
                    lambda x: f"{x}.SZ" if str(x).startswith(("0", "3")) else f"{x}.SH"
                )
                .tolist()
            )
        else:
            logger.error(f"Row group {rg_idx}: no ts_code/symbol!")
            continue

        # Fill new columns
        for col in NEW_COLS:
            if col in rg_df.columns:
                rg_df = rg_df.drop(columns=[col])
            vals = []
            for tc in ts_codes:
                v = sw_map.get(tc.strip(), {}).get(col, "")
                vals.append(v if v and str(v) != "nan" else None)
            rg_df[col] = vals

        # Align to new_schema column order
        for c in new_col_names:
            if c not in rg_df.columns:
                rg_df[c] = pd.NA
        rg_df = rg_df[new_col_names]

        table = pa.Table.from_pandas(rg_df, schema=new_schema, preserve_index=False)
        writer.write_table(table)
        total_rows += len(rg_df)

        if (rg_idx + 1) % 5 == 0 or rg_idx == pf.metadata.num_row_groups - 1:
            logger.info(
                f"  Row group {rg_idx + 1}/{pf.metadata.num_row_groups}, {total_rows:,} rows"
            )

    writer.close()
    pf.close()

    # ── 4. WORM backup + atomic replace ──
    backup = PANEL.replace(".parquet", "_presw_{}.parquet".format(
        pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")))
    shutil.copy2(PANEL, backup)
    logger.info(f"备份: {backup}")
    os.replace(tmp_path, PANEL)

    # ── 5. Audit ──
    logger.info("\n" + "=" * 55)
    logger.info("DONE")
    logger.info("=" * 55)
    pf2 = pq.ParquetFile(PANEL)
    logger.info(
        f"Panel: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols"
    )
    pf2.close()

    t = pq.read_table(PANEL, columns=["date"] + NEW_COLS).to_pandas()
    max_date = t["date"].max()
    latest = t[t["date"] == max_date]
    logger.info(f"\nFill rates on {max_date.date()}:")
    for col in NEW_COLS:
        n = latest[col].notna().sum()
        logger.info(f"  {col}: {n}/{len(latest)} ({n / len(latest) * 100:.1f}%)")


if __name__ == "__main__":
    main()
