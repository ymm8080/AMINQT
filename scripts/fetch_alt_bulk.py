"""
Bulk-fetch LHB (龙虎榜) and northbound (北向资金) data for the full date range,
caching into supply_cache/alt_data so it flows into the panel.

Strategy:
  - LHB: AKShare stock_lhb_detail_em supports date-range queries.
    Test with a short window (1 month) first, then expand to 2.5 years.
    AKShare paginates month by month internally (tqdm progress bar).
  - Northbound: AKShare stock_hsgt_hist_em returns all history in one call.

Usage:
    python scripts/fetch_alt_bulk.py
"""
import logging
import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_alt_bulk")

START = "20240102"
END = "20260727"
LHB_DIR = "data/supply_cache/alt_data/lhb"
NB_DIR = "data/supply_cache/alt_data/northbound"

# AKShare stock_lhb_detail_em known Chinese column names
LHB_DATE_COLS = ["上榜日", "日期", "上榜日期", "trade_date", "date"]
LHB_SYMBOL_COLS = ["代码", "stock_code", "ts_code", "symbol"]
LHB_NET_BUY_COLS = ["净买入额", "net_buy_amount", "net_buy"]
LHB_BUY_AMT_COLS = ["买入金额", "buy_amount", "买入成交额"]
LHB_SELL_AMT_COLS = ["卖出金额", "sell_amount", "卖出成交额"]


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    """Find the first candidate column that exists in cols."""
    for c in candidates:
        if c in cols:
            return c
    return None


def _stats(df: pd.DataFrame, label: str) -> None:
    """Print row count, date range, and stock coverage."""
    if df.empty:
        logger.warning("  [%s] EMPTY — no data returned", label)
        return
    rows = len(df)
    date_min = df["date"].min() if "date" in df.columns else "N/A"
    date_max = df["date"].max() if "date" in df.columns else "N/A"
    n_stocks = df["symbol"].nunique() if "symbol" in df.columns else "N/A"
    n_dates = df["date"].nunique() if "date" in df.columns else "N/A"
    logger.info(
        "  [%s] %d rows  |  %s stocks  |  %s dates  |  range [%s, %s]",
        label, rows, n_stocks, n_dates, date_min, date_max,
    )


def _parse_lhb_akshare(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse AKShare stock_lhb_detail_em output into uniform [symbol, date, lhb_*] columns."""
    if raw.empty:
        return raw

    out = pd.DataFrame()

    # Symbol
    sym_col = _find_col(list(raw.columns), LHB_SYMBOL_COLS)
    if sym_col:
        out["symbol"] = raw[sym_col].astype(str).str.zfill(6)
    else:
        logger.warning("  LHB: no symbol column found among %s", list(raw.columns)[:5])
        out["symbol"] = ""

    # Date
    date_col = _find_col(list(raw.columns), LHB_DATE_COLS)
    if date_col and date_col in raw.columns:
        out["date"] = pd.to_datetime(raw[date_col])
    else:
        logger.warning("  LHB: no date column found, columns=%s", list(raw.columns)[:8])

    # Net buy
    nb_col = _find_col(list(raw.columns), LHB_NET_BUY_COLS)
    if nb_col:
        out["lhb_net_buy"] = pd.to_numeric(raw[nb_col], errors="coerce")
    else:
        out["lhb_net_buy"] = 0.0

    # Buy amount
    ba_col = _find_col(list(raw.columns), LHB_BUY_AMT_COLS)
    if ba_col:
        out["lhb_buy_amt"] = pd.to_numeric(raw[ba_col], errors="coerce")
    else:
        out["lhb_buy_amt"] = 0.0

    # Sell amount
    sa_col = _find_col(list(raw.columns), LHB_SELL_AMT_COLS)
    if sa_col:
        out["lhb_sell_amt"] = pd.to_numeric(raw[sa_col], errors="coerce")
    else:
        out["lhb_sell_amt"] = 0.0

    return out


def fetch_lhb_akshare_direct() -> pd.DataFrame:
    """Fetch LHB directly via AKShare with proper column mapping.

    Uses stock_lhb_detail_em which supports start_date/end_date date ranges.
    The internal DataSupplyChain.fetch_lhb has incomplete Chinese column name
    mapping, so we handle it directly here.
    """
    import akshare as ak

    out_path = os.path.join(LHB_DIR, f"all_{START}_{END}.parquet")
    os.makedirs(LHB_DIR, exist_ok=True)

    if os.path.exists(out_path):
        df = pd.read_parquet(out_path)
        logger.info("  LHB: loaded from cached %s (%d rows)", out_path, len(df))
        return df

    # Phase 1: short-range smoke test (1 month)
    short_start, short_end = "20240102", "20240202"
    logger.info("LHB Phase 1 — smoke test [%s -> %s]", short_start, short_end)
    raw_short = ak.stock_lhb_detail_em(start_date=short_start, end_date=short_end)
    if raw_short is None or raw_short.empty:
        logger.error("  LHB short-range returned empty — aborting.")
        return pd.DataFrame()

    df_short = _parse_lhb_akshare(raw_short)
    _stats(df_short, "LHB-short")
    logger.info("  LHB short-range columns: %s", list(raw_short.columns))

    # Phase 2: full range
    logger.info("LHB Phase 2 — full range [%s -> %s]", START, END)
    raw_full = ak.stock_lhb_detail_em(start_date=START, end_date=END)
    if raw_full is None or raw_full.empty:
        logger.error("  LHB full-range returned empty — aborting.")
        return pd.DataFrame()

    df_full = _parse_lhb_akshare(raw_full)
    _stats(df_full, "LHB-full")
    df_full.to_parquet(out_path, index=False)
    logger.info("  LHB saved to %s  (%d rows, %.1f MB)",
                out_path, len(df_full), os.path.getsize(out_path) / 1024 / 1024)
    return df_full


def fetch_northbound(dsc: DataSupplyChain) -> pd.DataFrame:
    """Fetch northbound via DataSupplyChain method."""
    out_path = os.path.join(NB_DIR, f"all_{START}_{END}.parquet")
    os.makedirs(NB_DIR, exist_ok=True)

    if os.path.exists(out_path):
        df = pd.read_parquet(out_path)
        logger.info("  Northbound: loaded from cached %s (%d rows)", out_path, len(df))
        return df

    logger.info("Northbound — full range [%s -> %s]", START, END)
    df = dsc.fetch_northbound(start_date=START, end_date=END, refresh=True)
    _stats(df, "Northbound")
    df.to_parquet(out_path, index=False)
    logger.info("  Northbound saved to %s  (%d rows, %.1f MB)",
                out_path, len(df), os.path.getsize(out_path) / 1024 / 1024)
    return df


def main():
    t0 = datetime.now()
    logger.info("=" * 55)
    logger.info("Bulk alternative-data fetch started at %s", t0.isoformat())
    logger.info("=" * 55)

    dsc = DataSupplyChain(cache_dir="data/supply_cache")

    lhb_df = fetch_lhb_akshare_direct()
    nb_df = fetch_northbound(dsc)

    elapsed = (datetime.now() - t0).total_seconds()
    logger.info("=" * 55)
    logger.info("DONE in %.1f seconds.", elapsed)
    logger.info("  LHB:  %s rows  (%s stocks, %s dates)",
                len(lhb_df) if not lhb_df.empty else 0,
                lhb_df["symbol"].nunique() if not lhb_df.empty and "symbol" in lhb_df else "?",
                lhb_df["date"].nunique() if not lhb_df.empty and "date" in lhb_df else "?")
    logger.info("  Northbound: %s rows  (%s dates)",
                len(nb_df) if not nb_df.empty else 0,
                nb_df["date"].nunique() if not nb_df.empty and "date" in nb_df else "?")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
