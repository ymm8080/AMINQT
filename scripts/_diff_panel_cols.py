"""Diff V3 panel column sets across WORM snapshots to explain column growth."""

import os
import sys

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = r"D:\AMINQT\PARQUET"
SNAPS = [
    (
        "cyqderiv  (08-02 晚备份)",
        "panel_full_enriched_v3_cyqderiv_20260802_202326.parquet",
    ),
    (
        "tailfix    (08-03 00:25)",
        "panel_full_enriched_v3_tailfix_20260803_002500.parquet",
    ),
    (
        "preholder_evt (08-03 02:15)",
        "panel_full_enriched_v3_preholder_evt_20260803_021534.parquet",
    ),
    (
        "preblock   (08-03 06:12)",
        "panel_full_enriched_v3_preblock_trade_20260803_061225.parquet",
    ),
    (
        "prelhb_ext (08-03 06:07)",
        "panel_full_enriched_v3_prelhb_extract_20260803_060747.parquet",
    ),
    (
        "preseats_v2 (08-03 09:42)",
        "panel_full_enriched_v3_preseats_v2_20260803_094255.parquet",
    ),
    (
        "gate_base  (08-03 10:45)",
        "panel_full_enriched_v3_gate_base_20260803_104548.parquet",
    ),
    ("CURRENT    (08-03 10:46)", "panel_full_enriched_v3.parquet"),
]
names = {}
for label, fn in SNAPS:
    p = os.path.join(BASE, fn)
    if not os.path.exists(p):
        names[label] = None
        continue
    pf = pq.ParquetFile(p)
    names[label] = set(pf.schema.names)
    print(f"{label:28s} -> {len(pf.schema.names):3d} cols")

# cumulative diff from earliest existing snapshot
prev = None
prev_label = None
for label, fn in SNAPS:
    cur = names[label]
    if cur is None:
        continue
    if prev is not None:
        added = sorted(cur - prev)
        removed = sorted(prev - cur)
        if added or removed:
            print(f"\n--- {prev_label} -> {label}: +{len(added)} -{len(removed)} ---")
            print("  ADDED:   ", ", ".join(added) if added else "(none)")
            if removed:
                print("  REMOVED: ", ", ".join(removed))
    prev = cur
    prev_label = label
