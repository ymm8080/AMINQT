#!/usr/bin/env python3
"""Single-source enrichment driver — one agent per data source.

Usage:
  python scripts/enrich_one_source.py --source fina_indicator [--panel PATH] [--refresh]
  python scripts/enrich_one_source.py --source cyq_tushare
  python scripts/enrich_one_source.py --source northbound

Each invocation:
  1. Loads the base panel (panel_full_enriched_v3.parquet)
  2. Fetches/reads cache for ONE data source
  3. Merges into the panel
  4. Writes a partial output: data/enrich_parts/{source}.parquet

The final assembly step (scripts/assemble_enriched.py) left-joins all parts.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # ensure app.* imports work

DEFAULT_PANEL = str(ROOT / "data" / "panel_full_enriched_v3.parquet")
ALT_CACHE_DIR = str(ROOT / "data" / "supply_cache" / "alt_data")
PARTS_DIR = str(ROOT / "data" / "enrich_parts")
PROGRESS_FILE = str(ROOT / "data" / "enrich_progress.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enrich_one")


# ---------------------------------------------------------------------------
# Per-source merge functions (adapted from build_full_panel.py)
# ---------------------------------------------------------------------------


def _load_supply():
    from app.pipeline1.data_supply import DataSupplyChain

    return DataSupplyChain()


def _write_progress(msg: str) -> None:
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except OSError:
        pass


def merge_northbound(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    df = supply.fetch_northbound(start_date=start, end_date=end, refresh=refresh)
    if len(df) == 0:
        logger.warning("northbound: empty result")
        return panel
    date_cols = [
        c for c in df.columns if c not in ("symbol", "date") and not c.startswith("_")
    ]
    nb = df[["date"] + date_cols].drop_duplicates(subset=["date"])
    before = len(panel.columns)
    panel = panel.merge(nb, on="date", how="left")
    logger.info(
        "northbound: %d rows, +%d cols → panel", len(df), len(panel.columns) - before
    )
    return panel


def merge_margin(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    df = supply.fetch_margin(start_date=start, end_date=end, refresh=refresh)
    if len(df) == 0:
        logger.warning("margin: empty result")
        return panel
    merge_cols = ["symbol", "date"]
    avail = [c for c in df.columns if c not in merge_cols and not c.startswith("_")]
    before = len(panel.columns)
    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
    logger.info(
        "margin: %d rows, +%d cols → panel", len(df), len(panel.columns) - before
    )
    return panel


def merge_fina_indicator(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch fina_indicator for missing stocks only, merge via merge_asof."""
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    panel_symbols = set(panel["symbol"].unique())

    # ── 1. Try full-market query first ──
    try:
        df = supply.fetch_fina_indicator(
            start_date=start, end_date=end, refresh=refresh
        )
        if len(df) > 0:
            logger.info("fina_indicator: full-market returned %d rows", len(df))
            return _merge_fina_asof(panel, df)
    except Exception as exc:
        logger.info(
            "fina_indicator: full-market failed (%s), using per-stock cache", exc
        )

    # ── 2. Load from per-stock cache ──
    cache_dir = os.path.join(ALT_CACHE_DIR, "fina_indicator")
    caches = [
        f
        for f in glob.glob(os.path.join(cache_dir, "*.parquet"))
        if not os.path.basename(f).startswith("all_")
    ]
    cached_symbols: set[str] = set()
    frames = []
    for fpath in caches:
        try:
            df_one = pd.read_parquet(fpath)
            if len(df_one) and "symbol" in df_one.columns:
                frames.append(df_one)
                cached_symbols.update(df_one["symbol"].unique().tolist())
        except Exception as exc:
            logger.warning("fina_indicator: error reading %s: %s", fpath, exc)

    missing = panel_symbols - cached_symbols
    logger.info(
        "fina_indicator: %d cached symbols, %d missing → fetching",
        len(cached_symbols),
        len(missing),
    )

    # ── 3. Fetch missing stocks ──
    if missing:
        from app.pipeline1.panel_builder import _parallel_fetch

        def _fetch_one(sym: str):
            ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
            df_one = supply.fetch_fina_indicator(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                refresh=True,
            )
            return df_one if len(df_one) else None

        t0 = time.time()
        new_frames = _parallel_fetch(
            _fetch_one,
            sorted(missing),
            desc="fina_indicator",
            progress_file=PROGRESS_FILE,
        )
        elapsed = time.time() - t0
        if new_frames:
            frames.extend(new_frames)
            logger.info(
                "fina_indicator: fetched %d/%d missing stocks in %.0fs",
                len(new_frames),
                len(missing),
                elapsed,
            )
        else:
            logger.warning(
                "fina_indicator: no new data for %d missing stocks", len(missing)
            )

    if not frames:
        logger.warning("fina_indicator: no data at all")
        return panel

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "announce_date"])
    logger.info("fina_indicator: %d rows from %d stock files", len(df), len(frames))
    return _merge_fina_asof(panel, df)


def _merge_fina_asof(panel: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    if "announce_date" not in df.columns:
        logger.warning("fina_indicator: no announce_date column")
        return panel
    fin_cols = [
        c
        for c in df.columns
        if c not in ("symbol", "report_period", "_ts_code", "announce_date")
    ]
    f = df[["symbol", "announce_date"] + fin_cols].copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    f = f.dropna(subset=["announce_date"])
    f = f.sort_values("announce_date")
    if len(f) == 0:
        return panel
    panel_p = panel.sort_values("date").copy()
    before = len(panel_p.columns)
    panel_p = pd.merge_asof(
        panel_p,
        f,
        left_on="date",
        right_on="announce_date",
        by="symbol",
        direction="backward",
    )
    logger.info(
        "fina_indicator (merge_asof): +%d cols → panel", len(panel_p.columns) - before
    )
    return panel_p


def merge_lhb(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    try:
        df = supply.fetch_lhb(start_date=start, end_date=end, refresh=refresh)
        if len(df) > 0 and "date" in df.columns and "symbol" in df.columns:
            merge_cols = ["symbol", "date"]
            avail = [
                c for c in df.columns if c not in merge_cols and not c.startswith("_")
            ]
            before = len(panel.columns)
            panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
            logger.info(
                "lhb: %d rows, +%d cols → panel", len(df), len(panel.columns) - before
            )
            return panel
    except Exception as exc:
        logger.info("lhb: fetch failed (%s), trying cache", exc)

    # Fallback: read from cache
    cache_dir = os.path.join(ALT_CACHE_DIR, "lhb")
    caches = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    if not caches:
        logger.warning("lhb: no cache files")
        return panel
    raw = pd.read_parquet(caches[-1])
    logger.info("lhb: read cache %s: %d rows", os.path.basename(caches[-1]), len(raw))
    if "date" in raw.columns and "symbol" in raw.columns:
        raw = raw.dropna(subset=["date"])
        merge_cols = ["symbol", "date"]
        avail = [
            c for c in raw.columns if c not in merge_cols and not str(c).startswith("_")
        ]
        before = len(panel.columns)
        panel = panel.merge(raw[merge_cols + avail], on=merge_cols, how="left")
        logger.info("lhb: +%d cols → panel", len(panel.columns) - before)
    return panel


def merge_sector_index(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    df = supply.fetch_sector_index(start_date=start, end_date=end, refresh=refresh)
    if len(df) == 0:
        logger.warning("sector_index: empty result")
        return panel
    if "industry" not in panel.columns:
        logger.warning("sector_index: no industry column")
        return panel

    name_to_code = {
        name: code
        for code, name in df[["index_code", "index_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    ind_map: dict[str, str] = {}
    for ind_name in panel["industry"].dropna().unique():
        ind_clean = str(ind_name).strip()
        if ind_clean in name_to_code:
            ind_map[ind_clean] = ind_clean
        else:
            for sw_name in name_to_code:
                if ind_clean in str(sw_name) or str(sw_name) in ind_clean:
                    ind_map[ind_clean] = str(sw_name)
                    break
    if not ind_map:
        logger.warning("sector_index: no industry→SW mapping")
        return panel

    panel["_sw_name"] = panel["industry"].map(ind_map)
    sw_data = df.rename(
        columns={
            "ret_pct": "sw_ret_1d",
            "close": "sw_index_close",
            "volume": "sw_index_vol",
        }
    )
    avail = [c for c in sw_data.columns if c not in ("index_code", "date")]
    before = len(panel.columns)
    panel = panel.merge(
        sw_data[["index_name", "date"] + avail],
        left_on=["_sw_name", "date"],
        right_on=["index_name", "date"],
        how="left",
    )
    panel = panel.drop(columns=["_sw_name", "index_name"], errors="ignore")
    logger.info("sector_index: +%d cols → panel", len(panel.columns) - before)
    return panel


def merge_holdernumber(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    panel_symbols = set(panel["symbol"].unique())

    # Load from cache
    cache_dir = os.path.join(ALT_CACHE_DIR, "holdernumber")
    caches = glob.glob(os.path.join(cache_dir, "*.parquet"))
    cached_symbols: set[str] = set()
    frames = []
    for fpath in caches:
        try:
            df_one = pd.read_parquet(fpath)
            if len(df_one) and "symbol" in df_one.columns:
                frames.append(df_one)
                cached_symbols.update(df_one["symbol"].unique().tolist())
        except Exception as exc:
            logger.warning("holdernumber: error reading %s: %s", fpath, exc)

    missing = panel_symbols - cached_symbols
    logger.info(
        "holdernumber: %d cached symbols, %d missing",
        len(cached_symbols),
        len(missing),
    )

    if missing:
        from app.pipeline1.panel_builder import _parallel_fetch

        def _fetch_one(sym: str):
            ts_code = f"{sym}.{'SZ' if sym.startswith(('0', '3', '1')) else 'SH'}"
            df_one = supply.fetch_holdernumber(
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                refresh=True,
            )
            return df_one if len(df_one) else None

        t0 = time.time()
        new_frames = _parallel_fetch(
            _fetch_one,
            sorted(missing),
            desc="holdernumber",
            progress_file=PROGRESS_FILE,
        )
        elapsed = time.time() - t0
        if new_frames:
            frames.extend(new_frames)
            logger.info(
                "holdernumber: fetched %d/%d in %.0fs",
                len(new_frames),
                len(missing),
                elapsed,
            )

    if not frames:
        logger.warning("holdernumber: no data")
        return panel

    df = pd.concat(frames, ignore_index=True)
    logger.info("holdernumber: %d rows from %d files", len(df), len(frames))

    if "announce_date" not in df.columns:
        logger.warning("holdernumber: no announce_date")
        return panel

    hn_cols = [
        c
        for c in df.columns
        if c not in ("symbol", "date", "_ts_code", "announce_date")
    ]
    f = df[["symbol", "announce_date"] + hn_cols].copy()
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    f = f.dropna(subset=["announce_date"])
    f = f.sort_values("announce_date")
    if len(f) == 0:
        return panel

    panel_p = panel.sort_values("date").copy()
    before = len(panel_p.columns)
    panel_p = pd.merge_asof(
        panel_p,
        f,
        left_on="date",
        right_on="announce_date",
        by="symbol",
        direction="backward",
    )
    logger.info("holdernumber: +%d cols → panel", len(panel_p.columns) - before)
    return panel_p


def merge_holdertrade(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    try:
        df = supply.fetch_holdertrade(start_date=start, end_date=end, refresh=refresh)
    except Exception:
        # Fallback: cache
        cache_dir = os.path.join(ALT_CACHE_DIR, "holdertrade")
        caches = glob.glob(os.path.join(cache_dir, "*.parquet"))
        if caches:
            frames = [pd.read_parquet(f) for f in caches]
            df = pd.concat(frames, ignore_index=True)
            logger.info("holdertrade: %d rows from cache", len(df))
        else:
            logger.warning("holdertrade: no data")
            return panel

    if len(df) and "announce_date" in df.columns and "sh_net_sign" in df.columns:
        daily_net = (
            df.groupby(["symbol", "announce_date"])
            .agg(
                sh_net_change_sign=("sh_net_sign", "sum"),
                sh_change_amt_total=("sh_change_amt", "sum"),
            )
            .reset_index()
        )
        daily_net = daily_net.rename(columns={"announce_date": "date"})
        daily_net["date"] = pd.to_datetime(daily_net["date"])
        before = len(panel.columns)
        panel = panel.merge(daily_net, on=["symbol", "date"], how="left")
        logger.info("holdertrade: +%d cols → panel", len(panel.columns) - before)
    return panel


def merge_daily_basic(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch daily_basic per date, merge by symbol+date."""
    supply = _load_supply()
    panel["date"].min().strftime("%Y%m%d")
    panel["date"].max().strftime("%Y%m%d")
    dates = panel["date"].drop_duplicates().sort_values()

    from app.pipeline1.panel_builder import _parallel_fetch

    def _fetch_one_date(d):
        ds = d.strftime("%Y%m%d")
        df_one = supply.fetch_daily_basic(trade_date=ds, refresh=refresh)
        return df_one if len(df_one) else None

    frames = _parallel_fetch(
        _fetch_one_date,
        dates.tolist(),
        desc="daily_basic",
        progress_file=PROGRESS_FILE,
    )
    if not frames:
        logger.warning("daily_basic: no data")
        return panel

    df = pd.concat(frames, ignore_index=True)
    merge_cols = ["symbol", "date"]
    avail = [c for c in df.columns if c not in merge_cols and not c.startswith("_")]
    before = len(panel.columns)
    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
    logger.info(
        "daily_basic: %d rows, +%d cols → panel", len(df), len(panel.columns) - before
    )
    return panel


def merge_stk_limit(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch stk_limit per date, merge by symbol+date."""
    supply = _load_supply()
    dates = panel["date"].drop_duplicates().sort_values()

    from app.pipeline1.panel_builder import _parallel_fetch

    def _fetch_one_date(d):
        ds = d.strftime("%Y%m%d")
        df_one = supply.fetch_stk_limit(trade_date=ds, refresh=refresh)
        return df_one if len(df_one) else None

    frames = _parallel_fetch(
        _fetch_one_date,
        dates.tolist(),
        desc="stk_limit",
        progress_file=PROGRESS_FILE,
    )
    if not frames:
        logger.warning("stk_limit: no data")
        return panel

    df = pd.concat(frames, ignore_index=True)
    merge_cols = ["symbol", "date"]
    avail = [c for c in df.columns if c not in merge_cols and not c.startswith("_")]
    before = len(panel.columns)
    panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
    logger.info(
        "stk_limit: %d rows, +%d cols → panel", len(df), len(panel.columns) - before
    )
    return panel


def merge_cyq_tushare(panel: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Fetch Tushare cyq_perf chip distribution for all symbols."""
    supply = _load_supply()
    start = panel["date"].min().strftime("%Y%m%d")
    end = panel["date"].max().strftime("%Y%m%d")
    symbols = panel["symbol"].unique().tolist()

    _write_progress(f"cyq_tushare: starting {len(symbols)} symbols")
    t0 = time.time()
    df = supply.fetch_chip_distribution_batch(
        symbols,
        start_date=start,
        end_date=end,
        refresh=refresh,
    )
    elapsed = time.time() - t0
    if len(df):
        merge_cols = ["symbol", "date"]
        avail = [c for c in df.columns if c not in merge_cols and not c.startswith("_")]
        before = len(panel.columns)
        panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
        _write_progress(
            f"cyq_tushare: done {len(symbols)} symbols, "
            f"+{len(panel.columns) - before} cols in {elapsed:.0f}s"
        )
        logger.info(
            "cyq_tushare: %d rows, +%d cols → panel (%.0fs)",
            len(df),
            len(panel.columns) - before,
            elapsed,
        )
    else:
        logger.warning("cyq_tushare: no data returned")
    return panel


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
SOURCES = {
    "northbound": merge_northbound,
    "margin": merge_margin,
    "fina_indicator": merge_fina_indicator,
    "lhb": merge_lhb,
    "holdernumber": merge_holdernumber,
    "holdertrade": merge_holdertrade,
    "sector_index": merge_sector_index,
    "daily_basic": merge_daily_basic,
    "stk_limit": merge_stk_limit,
    "cyq_tushare": merge_cyq_tushare,
}


# ---------------------------------------------------------------------------
# Column lists for partial output (only keep NEW columns per source)
# ---------------------------------------------------------------------------
SOURCE_COL_PREFIXES: dict[str, list[str]] = {
    "northbound": ["north_"],
    "margin": ["margin_", "short_"],
    "fina_indicator": [
        "roe",
        "roe_deducted",
        "roa",
        "gross_margin",
        "net_margin",
        "eps_yoy",
        "rev_yoy",
        "profit_yoy",
        "op_cf_ratio",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
        "ar_turnover",
        "inventory_turnover",
        "ocf_to_or",
        "announce_date",
        "report_period",
    ],
    "lhb": ["lhb_"],
    "holdernumber": ["holder_count", "avg_shares_per_holder", "announce_date"],
    "holdertrade": ["sh_net_change_sign", "sh_change_amt_total"],
    "sector_index": ["sw_"],
    "daily_basic": [
        "pe_ttm",
        "pb",
        "total_mv",
        "circ_mv",
        "total_share",
        "float_share",
        "free_share",
    ],
    "stk_limit": ["up_limit", "down_limit"],
    "cyq_tushare": [
        "winner_rate",
        "his_low",
        "his_high",
        "cost_5pct",
        "cost_15pct",
        "cost_50pct",
        "cost_85pct",
        "cost_95pct",
        "weight_avg",
        "winner_ratio",
        "pct_90_con",
        "pct_70_con",
    ],
}


def _filter_new_cols(
    panel: pd.DataFrame, source: str, base_cols: set[str]
) -> pd.DataFrame:
    """Extract only the new columns this source added, plus symbol+date keys."""
    new_cols = [c for c in panel.columns if c not in base_cols]
    keep = ["symbol", "date"] + [c for c in new_cols if not c.startswith("_")]
    # Also check prefix patterns
    prefixes = SOURCE_COL_PREFIXES.get(source, [])
    for c in new_cols:
        if c in keep:
            continue
        for pfx in prefixes:
            if c.startswith(pfx) or pfx in c:
                keep.append(c)
                break
    keep = list(dict.fromkeys(keep))  # dedup preserving order
    result = panel[keep].copy()
    logger.info("Partial output for %s: %d cols: %s", source, len(keep) - 2, keep[2:])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Enrich one alt-data source")
    parser.add_argument(
        "--source", required=True, choices=list(SOURCES), help="Data source to enrich"
    )
    parser.add_argument(
        "--panel", default=DEFAULT_PANEL, help="Path to base panel parquet"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Force re-fetch (skip cache)"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Override ENRICH_WORKERS env var"
    )
    args = parser.parse_args()

    if args.workers:
        os.environ["ENRICH_WORKERS"] = str(args.workers)

    source = args.source
    panel_path = args.panel

    # ── Load base panel ──
    if not os.path.exists(panel_path):
        logger.error("Panel not found: %s", panel_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Loading panel: %s", panel_path)
    panel = pd.read_parquet(panel_path)
    logger.info(
        "Panel: %d rows, %d symbols, %d cols",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
    )

    # ── Deduplicate column names (parquet can carry duplicates from bad merges) ──
    dupes = panel.columns[panel.columns.duplicated()].tolist()
    if dupes:
        logger.info("Dropping %d duplicate columns: %s", len(dupes), dupes)
        panel = panel.loc[:, ~panel.columns.duplicated()]

    # ── Clean up stale _x/_y columns from previous partial merges ──
    stale_suffix_cols = [
        c for c in panel.columns if c.endswith("_x") or c.endswith("_y")
    ]
    if stale_suffix_cols:
        logger.info(
            "Dropping %d stale _x/_y columns: %s",
            len(stale_suffix_cols),
            stale_suffix_cols,
        )
        panel = panel.drop(columns=stale_suffix_cols)

    base_cols = set(panel.columns)

    # ── Check if source already present ──
    prefixes = SOURCE_COL_PREFIXES.get(source, [])
    existing_raw = [
        c
        for c in panel.columns
        for pfx in prefixes
        if (c.startswith(pfx) or pfx in c)
        and not c.endswith("_x")
        and not c.endswith("_y")
    ]
    existing = list(
        dict.fromkeys(existing_raw)
    )  # dedup (a col may match multiple prefixes)
    if existing and not args.refresh:
        logger.info(
            "Source %s already has columns: %s — skipping (use --refresh to force)",
            source,
            existing,
        )
        # Still write partial output for assembly
        os.makedirs(PARTS_DIR, exist_ok=True)
        out_path = os.path.join(PARTS_DIR, f"{source}.parquet")
        partial = panel[["symbol", "date"] + existing].copy()
        partial.to_parquet(out_path, index=False)
        logger.info("Wrote existing columns to %s", out_path)
        _write_progress(f"{source}: skipped (already present)")
        return

    # ── Run merge ──
    merge_fn = SOURCES[source]
    logger.info("Running enrich: %s (refresh=%s)", source, args.refresh)
    t0 = time.time()
    try:
        panel = merge_fn(panel, refresh=args.refresh)
    except Exception as exc:
        logger.error("%s failed: %s", source, exc, exc_info=True)
        _write_progress(f"{source}: FAILED — {exc}")
        sys.exit(1)
    elapsed = time.time() - t0

    new_cols = [c for c in panel.columns if c not in base_cols]
    logger.info(
        "%s: done in %.0fs, +%d new cols: %s",
        source,
        elapsed,
        len(new_cols),
        sorted(new_cols),
    )

    # ── Write partial output ──
    os.makedirs(PARTS_DIR, exist_ok=True)
    out_path = os.path.join(PARTS_DIR, f"{source}.parquet")
    partial = _filter_new_cols(panel, source, base_cols)
    partial.to_parquet(out_path, index=False)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    logger.info("Wrote partial: %s (%.1f MB, %d rows)", out_path, size_mb, len(partial))

    _write_progress(f"{source}: COMPLETE — +{len(new_cols)} cols in {elapsed:.0f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
