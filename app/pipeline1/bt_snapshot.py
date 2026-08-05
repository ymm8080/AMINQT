# -*- coding: utf-8 -*-
"""Rolling snapshot of the 4 V3 panel block-trade columns (dim33 upstream).

dim33_block_trade's 4 EWMA features are computed at train time from the 4
panel raw columns bt_count / bt_disc_raw / bt_inst_absorb / bt_amt_ratio_float_mv.
If a panel rebuild drops them, this rolling file is the faithful restore source
(NO Tushare re-fetch). It accumulates from each daily fetch's live
``pro.block_trade`` aggregation — NOT from the deduped raw cache
``block_trade_full.parquet``, which keeps only the last trade per (symbol, date)
and cannot reproduce bt_count>1 or the L1 noise filter. It also does not depend
on the panel, so it survives the very rebuild it protects against.

WORM: dated ``bt_v3_snapshot_*.parquet`` (scripts/_snapshot_bt_cols.py) are
immutable point-in-time backups; this ``_rolling`` file is the mutable daily
mirror, seeded from the panel on first creation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

BT_COLS = [
    "bt_count",
    "bt_disc_raw",
    "bt_inst_absorb",
    "bt_amt_ratio_float_mv",
]
SNAPSHOT_COLS = ["symbol", "date"] + BT_COLS


def refresh_rolling_snapshot(
    today: pd.DataFrame,
    path: str | Path,
    seed: pd.DataFrame | None = None,
    bt_cols: list[str] | None = None,
) -> dict[str, int]:
    """Append today's non-null bt_ rows into the rolling snapshot.

    Only (symbol, date) rows where at least one bt_ column is non-null are kept,
    so non-event days stay sparse. Dedup (symbol, date) keep=last makes re-runs
    crash-safe. On first creation the file is seeded from ``seed`` (the panel's
    bt_ history) so a fresh file starts with full history, not just today.
    Atomic replace. Returns {"appended": event rows from today, "total": file rows}.
    """
    path = Path(path)
    cols = bt_cols or BT_COLS
    have = [c for c in ["symbol", "date"] + cols if c in today.columns]
    bt_have = [c for c in cols if c in today.columns]
    if not bt_have:
        return {
            "appended": 0,
            "total": len(pd.read_parquet(path)) if path.exists() else 0,
        }

    today = today[have].copy()
    today = today[today[bt_have].notna().any(axis=1)]

    if path.exists():
        base = pd.read_parquet(path)
    elif seed is not None and len(seed):
        base = seed[[c for c in have if c in seed.columns]].copy()
        base = base[base[bt_have].notna().any(axis=1)]
    else:
        base = pd.DataFrame(columns=have)

    if not len(today) and not path.exists():
        return {"appended": 0, "total": 0}

    merged = today if len(base) == 0 else pd.concat([base, today], ignore_index=True)
    merged = merged.drop_duplicates(subset=["symbol", "date"], keep="last")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return {"appended": len(today), "total": len(merged)}
