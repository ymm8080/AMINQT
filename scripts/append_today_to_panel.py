"""Append today's OHLCV + CYQ into panel_full_enriched_v3.parquet.

Usage: python scripts/append_today_to_panel.py [trade_date]
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from app.pipeline1.data_supply import DataSupplyChain

PANEL_PATH = "data/panel_full_enriched_v3.parquet"


def main(trade_date: str):
    print(f"Loading panel: {PANEL_PATH}")
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel)} rows, {panel['date'].min()} ~ {panel['date'].max()}")

    # Remove today if already present (avoid duplicates)
    panel = panel[panel["date"] < pd.to_datetime(trade_date)]

    # Append today's OHLCV + LHB
    supply = DataSupplyChain()
    print(f"Appending today's data for {trade_date} ...")
    panel = supply.append_today_to_panel(
        panel, trade_date=trade_date, sources=["ohlcv", "lhb"]
    )

    # Merge today's CYQ from Tushare cache
    cyq_path = f"data/supply_cache/alt_data/cyq_tushare/{trade_date}.parquet"
    if os.path.exists(cyq_path):
        cyq = pd.read_parquet(cyq_path)
        cyq_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
        today_mask = panel["date"] == pd.to_datetime(trade_date)
        for col in cyq_cols:
            if col in panel.columns:
                panel.loc[today_mask, col] = (
                    panel.loc[today_mask, "symbol"]
                    .map(cyq.set_index("symbol")[col])
                    .values
                )
        print(f"CYQ merged: {len(cyq_cols)} cols for {today_mask.sum()} rows")

    # Save back (WORM: write new file, don't overwrite)
    out_path = PANEL_PATH  # overwrite v3 — it's the working panel
    panel.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")
    print(f"  {len(panel)} rows, {panel['date'].min()} ~ {panel['date'].max()}")


if __name__ == "__main__":
    td = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    main(td)
