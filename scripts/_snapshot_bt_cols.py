"""Snapshot the 4 V3 panel block-trade upstream columns to a standalone WORM parquet.

Purpose: dim33_block_trade's 4 EWMA features (bt_act/disc/inst_abs/mv_ratio_ewma)
are computed at train time from these 4 panel columns. If a future panel rebuild
drops them, this snapshot is the faithful restore source (noise-filtered, full
history) — NOT the deduped daily cache `block_trade_full.parquet`, which keeps
only the last trade per (symbol, date) and cannot reproduce bt_count>1 or the
L1 noise filter.

WORM: each run writes a new dated file; never overwrites.
"""

import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import PANEL_V3_PATH

BT_COLS = [
    "bt_count",
    "bt_disc_raw",
    "bt_inst_absorb",
    "bt_amt_ratio_float_mv",
]
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "supply_cache"
    / "alt_data"
    / "block_trade"
)


def main():
    panel = Path(PANEL_V3_PATH)
    if not panel.exists():
        print(f"ERROR: panel not found: {panel}")
        return
    pf = pq.ParquetFile(panel)
    all_cols = pf.schema_arrow.names
    missing = [c for c in BT_COLS if c not in all_cols]
    if missing:
        print(f"ERROR: panel missing bt columns: {missing}")
        return
    print(f"Panel: {panel} | {pf.metadata.num_rows:,} rows, {len(all_cols)} cols")
    pf.close()

    df = pq.read_table(panel, columns=["symbol", "date"] + BT_COLS).to_pandas()
    print(f"Extracted: {len(df):,} rows, cols={list(df.columns)}")

    # Coverage report
    print("\nCoverage per bt column:")
    for c in BT_COLS:
        nn = df[c].notna().sum()
        rng = df.loc[df[c].notna(), "date"]
        rng = (rng.min().date(), rng.max().date()) if len(rng) else (None, None)
        print(
            f"  {c:24s} non-null={nn:>9,} ({nn / len(df) * 100:.2f}%)  {rng[0]} -> {rng[1]}"
        )

    # Write WORM snapshot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"bt_v3_snapshot_{ts}.parquet"
    df.to_parquet(out, index=False, compression="snappy")
    print(f"\nSnapshot: {out} ({out.stat().st_size / 1e6:.1f} MB)")

    # Verify
    v = pq.ParquetFile(out)
    vd = v.read().to_pandas()
    v.close()
    ok = all(vd[c].notna().sum() == df[c].notna().sum() for c in BT_COLS) and len(
        vd
    ) == len(df)
    print(f"Verify: {len(vd):,} rows, {len(vd.columns)} cols, values match={ok}")


if __name__ == "__main__":
    main()
