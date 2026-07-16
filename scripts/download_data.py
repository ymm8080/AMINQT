# -*- coding: utf-8 -*-
"""Phase 1 — Download daily K-line for the stock pool to data/raw/.

Uses the configured data adapter (akshare by default; iFinD if creds set).
Saves one CSV per symbol. Raw Chinese columns are preserved; data_loader
canonicalizes them on read (PROMPT_CONTENT §1).
"""

import logging
import os
import sys

# Allow `python scripts/download_data.py` from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from data.adapters import get_adapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    """Download all pool symbols and save to data/raw/.

    Returns:
        0 if at least one symbol downloaded, else 1.
    """
    start = settings.DATA_START.isoformat()
    end = settings.DATA_END.isoformat()
    adapter = get_adapter()
    logger.info(
        "Downloading %d symbols [%s → %s] via %s",
        len(settings.STOCK_LIST),
        start,
        end,
        type(adapter).__name__,
    )

    ok = 0
    for sym in settings.STOCK_LIST:
        try:
            df = adapter.fetch_daily(sym, start, end)
            path = os.path.join(settings.RAW_DIR, f"{sym}.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info("Saved %s → %s (%d rows)", sym, path, len(df))
            ok += 1
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", sym, exc)

    logger.info(
        "Done: %d/%d symbols downloaded into %s",
        ok,
        len(settings.STOCK_LIST),
        settings.RAW_DIR,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
