"""Fast daily runner: bypass enrich_cyq (use cached panel), pass pre-built panel to production pipeline.

Usage: python scripts/_run_daily_fast.py [trade_date]
"""
from __future__ import annotations

import logging
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
import pyarrow.parquet as pq

from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.predict_runner import find_bundles

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_daily_fast")

PANEL_PATH = "data/panel_full_enriched_v3.parquet"


def build_panel_fast(trade_date: str) -> pd.DataFrame:
    """Load cached panel + append today's data + merge today's CYQ from Tushare cache.

    Bypasses enrich_cyq (slow full recompute) — panel_full_enriched_v3 already has CYQ.
    Only merges today's CYQ row from the Tushare cache file.
    """
    # 1. Load existing panel (filter to 2026+ for memory, 250d warmup needs ~1 year)
    logger.info("Loading panel (filtered to 2026-01-01+): %s", PANEL_PATH)
    table = pq.read_table(
        PANEL_PATH,
        filters=[("date", ">=", pd.Timestamp("2026-01-01"))],
    )
    panel = table.to_pandas()
    del table
    logger.info(
        "Panel: %d stocks, %d rows, %d cols, dates %s ~ %s",
        panel["symbol"].nunique(),
        len(panel),
        len(panel.columns),
        panel["date"].min(),
        panel["date"].max(),
    )

    # 2. Remove today's data if already present (avoid duplicates)
    panel = panel[panel["date"] < pd.to_datetime(trade_date)]

    # 3. Append today's OHLCV + LHB via DataSupplyChain (uses Tushare, cached)
    supply = DataSupplyChain()
    logger.info("Appending today's data for %s ...", trade_date)
    panel = supply.append_today_to_panel(
        panel, trade_date=trade_date, sources=["ohlcv", "lhb"]
    )
    logger.info(
        "After append: %d stocks, %d rows, dates %s ~ %s",
        panel["symbol"].nunique(),
        len(panel),
        panel["date"].min(),
        panel["date"].max(),
    )

    # 4. Merge today's CYQ from Tushare cache (dynamic date, not hardcoded)
    cyq_path = f"data/supply_cache/alt_data/cyq_tushare/{trade_date}.parquet"
    if os.path.exists(cyq_path):
        cyq = pd.read_parquet(cyq_path)
        cyq_cols = [c for c in cyq.columns if c not in ("symbol", "date")]
        today_dt = pd.to_datetime(trade_date)
        today_mask = panel["date"] == today_dt
        for col in cyq_cols:
            if col in panel.columns:
                panel.loc[today_mask, col] = (
                    panel.loc[today_mask, "symbol"]
                    .map(cyq.set_index("symbol")[col])
                    .values
                )
        logger.info("CYQ merge: %d cols updated for %d rows", len(cyq_cols), today_mask.sum())
    else:
        logger.warning("CYQ cache not found: %s — today's CYQ will be NaN", cyq_path)

    return panel


def main(trade_date: str):
    # 1. Build panel (fast: skip enrich_cyq, use cached)
    panel = build_panel_fast(trade_date)

    # 2. Find model bundles
    bundles = find_bundles(model_dir="models/pipeline1")
    logger.info("Bundles: %s", {k: os.path.basename(v) for k, v in bundles.items()})

    # 3. Run production pipeline with pre-built panel
    logger.info("Importing DailySelectionPipeline...")
    from app.pipeline1.daily_pipeline import DailySelectionPipeline
    logger.info("Constructing DailySelectionPipeline...")
    pipeline = DailySelectionPipeline(
        supply=DataSupplyChain(),
        bundle_paths=bundles,
    )
    logger.info("Running pipeline.run()...")
    result = pipeline.run(trade_date, panel=panel, market_state="range")

    lst = result.get("list")
    n = 0 if lst is None else len(lst)
    logger.info(
        "Done: mode=%s, %d stocks, schema=%s",
        result.get("mode"),
        n,
        result.get("schema_version", "-"),
    )
    if n:
        print(lst.to_string(index=False))
    return result


if __name__ == "__main__":
    td = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
    main(td)
