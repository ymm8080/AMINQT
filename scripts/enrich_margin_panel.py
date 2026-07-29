# -*- coding: utf-8 -*-
"""Step 1: Enrich panel cache with margin data — run once, persist result."""

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

import pandas as pd  # noqa: E402
from app.pipeline1.panel_builder import enrich_alt_data  # noqa: E402
from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

PANEL_PATH = r"D:\AMINQT\AMINQT CODES\data\panel_full_enriched.parquet"

# 1. Load panel
panel = pd.read_parquet(PANEL_PATH)
logger = logging.getLogger(__name__)
logger.info("Panel loaded: %d rows, %d cols", len(panel), len(panel.columns))

# 2. Date range
start = panel["date"].min().strftime("%Y%m%d")
end = panel["date"].max().strftime("%Y%m%d")
logger.info("Date range: %s — %s", start, end)

# 3-5. Supply chain + enrich margin
supply = DataSupplyChain()
panel = enrich_alt_data(
    panel, supply, sources=["margin"], start_date=start, end_date=end, refresh=True
)

# 6. Check columns
expected = ["margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol"]
for col in expected:
    if col in panel.columns:
        non_zero = (panel[col].notna() & (panel[col] != 0)).sum()
        total = panel[col].notna().sum()
        logger.info("  %s: non-NaN=%d, non-zero=%d", col, total, non_zero)
    else:
        logger.warning("  %s: NOT FOUND in panel", col)

logger.info("Post-enrich columns: %d total", len(panel.columns))

# 7. Save
panel.to_parquet(PANEL_PATH, index=False)
logger.info("Saved to %s", PANEL_PATH)
