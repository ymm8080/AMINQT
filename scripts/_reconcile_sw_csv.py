# -*- coding: utf-8 -*-
"""Reconcile sw_stock_classification.csv at the DATA_OTHERS mapped path.

fetch_sw_classification.py --incremental wrote only the 9 new rows to the
mapped path (D:\\AMINQT\\DATA OTHERS\\processed\\sw_stock_classification.csv),
overwriting the previous full file. The intact 5869-row file still lives at
the literal data/processed/sw_stock_classification.csv. Merge both, dedupe on
ts_code, and write the full 5878-row file back to the mapped path that
add_sw_cols_to_panel.py reads.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

LITERAL = Path("data/processed/sw_stock_classification.csv")
MAPPED = Path(r"D:\AMINQT\DATA OTHERS\processed\sw_stock_classification.csv")

full = pd.read_csv(LITERAL, dtype=str)
inc = pd.read_csv(MAPPED, dtype=str)
print(f"literal={len(full)} mapped_inc={len(inc)}")
merged = pd.concat([full, inc], ignore_index=True).drop_duplicates(
    subset=["ts_code"], keep="first"
)
print(f"merged={len(merged)} (expect {len(full) + len(inc)})")
assert len(merged) == len(full) + len(inc), "overlap between literal and incremental rows?"
merged.to_csv(MAPPED, index=False)
print(f"wrote {len(merged)} rows to {MAPPED}")
