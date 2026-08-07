#!/usr/bin/env python3
"""Build FULLY enriched panel with ALL alt data for 3227 stocks."""

import glob
import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PANEL_INPUT = "data/panel_full_enriched.parquet"
PANEL_OUTPUT = "data/panel_full_enriched_v3.parquet"
ALT_CACHE_DIR = "data/supply_cache/alt_data"


def load_panel(path):
    df = pd.read_parquet(path)
    logger.info(
        "Loaded panel: %d rows, %d symbols, %d columns, dates %s to %s",
        len(df),
        df["symbol"].nunique(),
        len(df.columns),
        df["date"].min(),
        df["date"].max(),
    )
    return df


def fix_industry(panel, refresh=False):
    from app.pipeline1.panel_builder import load_or_fetch_meta

    try:
        ind_map, name_map = load_or_fetch_meta(
            cache_dir="data/processed", refresh=refresh
        )
        if ind_map:
            before = (panel["industry"] == "UNKNOWN").sum()
            panel["industry"] = panel["symbol"].map(ind_map).fillna("UNKNOWN")
            after = (panel["industry"] == "UNKNOWN").sum()
            logger.info(
                "fix_industry: %d UNKNOWN -> %d UNKNOWN (fixed %d rows, %d industries)",
                before,
                after,
                before - after,
                panel["industry"].nunique(),
            )
        else:
            logger.warning("fix_industry: industry_map empty")
    except Exception as exc:
        logger.warning("fix_industry failed: %s", exc)
    return panel


def merge_northbound(panel, refresh=False):
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    try:
        df = supply.fetch_northbound(
            start_date=panel["date"].min().strftime("%Y%m%d"),
            end_date=panel["date"].max().strftime("%Y%m%d"),
            refresh=refresh,
        )
        if len(df) == 0:
            logger.warning("northbound: empty result")
            return panel
        date_cols = [
            c
            for c in df.columns
            if c not in ("symbol", "date") and not c.startswith("_")
        ]
        nb_by_date = df[["date"] + date_cols].drop_duplicates(subset=["date"])
        before = len(panel.columns)
        panel = panel.merge(nb_by_date, on="date", how="left")
        logger.info(
            "northbound: %d rows, +%d cols -> panel",
            len(df),
            len(panel.columns) - before,
        )
    except Exception as exc:
        logger.warning("northbound skipped: %s", exc)
    return panel


def merge_margin(panel, refresh=False):
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    try:
        df = supply.fetch_margin(
            start_date=panel["date"].min().strftime("%Y%m%d"),
            end_date=panel["date"].max().strftime("%Y%m%d"),
            refresh=refresh,
        )
        if len(df) == 0:
            logger.warning("margin: empty result")
            return panel
        merge_cols = ["symbol", "date"]
        avail = [c for c in df.columns if c not in merge_cols and not c.startswith("_")]
        before = len(panel.columns)
        panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
        logger.info(
            "margin: %d rows, %d symbols, +%d cols -> panel",
            len(df),
            df["symbol"].nunique(),
            len(panel.columns) - before,
        )
    except Exception as exc:
        logger.warning("margin skipped: %s", exc)
    return panel


def merge_fina_indicator(panel, refresh=False):
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    try:
        df = supply.fetch_fina_indicator(
            start_date=panel["date"].min().strftime("%Y%m%d"),
            end_date=panel["date"].max().strftime("%Y%m%d"),
            refresh=refresh,
        )
        if len(df) > 0:
            logger.info("fina_indicator: full-market query returned %d rows", len(df))
            return _merge_fina_panel(panel, df)
        else:
            logger.info("fina_indicator: full-market empty, using per-stock cache")
    except Exception as exc:
        logger.info(
            "fina_indicator: full-market failed (%s), using per-stock cache", exc
        )
    cache_dir = os.path.join(ALT_CACHE_DIR, "fina_indicator")
    caches = [
        f
        for f in glob.glob(os.path.join(cache_dir, "*.parquet"))
        if not os.path.basename(f).startswith("all_")
    ]
    if not caches:
        logger.warning("fina_indicator: no per-stock cache files")
        return panel
    frames = []
    for fpath in caches:
        try:
            df_one = pd.read_parquet(fpath)
            if len(df_one) and "symbol" in df_one.columns:
                frames.append(df_one)
        except Exception as exc:
            logger.warning("fina_indicator: error reading %s: %s", fpath, exc)
    if not frames:
        logger.warning("fina_indicator: no valid cache entries")
        return panel
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["symbol", "announce_date"])
    logger.info("fina_indicator: %d rows from %d per-stock files", len(df), len(caches))
    return _merge_fina_panel(panel, df)


def _merge_fina_panel(panel, df):
    if "announce_date" not in df.columns:
        logger.warning("fina_indicator: no announce_date column")
        return panel
    fin_cols = [
        c
        for c in df.columns
        if c not in ("symbol", "report_period", "_ts_code", "announce_date")
    ]
    f = df[["symbol", "announce_date"] + fin_cols].copy()
    f = f.sort_values("announce_date")
    f["announce_date"] = pd.to_datetime(f["announce_date"])
    f = f.dropna(subset=["announce_date"])
    if len(f) == 0:
        logger.warning("fina_indicator: no valid announce_date after dropna")
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
        "fina_indicator (merge_asof): +%d cols -> panel", len(panel_p.columns) - before
    )
    return panel_p


def merge_lhb(panel, refresh=False):
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    try:
        df = supply.fetch_lhb(
            start_date=panel["date"].min().strftime("%Y%m%d"),
            end_date=panel["date"].max().strftime("%Y%m%d"),
            refresh=refresh,
        )
        if len(df) > 0 and "date" in df.columns and "symbol" in df.columns:
            merge_cols = ["symbol", "date"]
            avail = [
                c for c in df.columns if c not in merge_cols and not c.startswith("_")
            ]
            before = len(panel.columns)
            panel = panel.merge(df[merge_cols + avail], on=merge_cols, how="left")
            logger.info(
                "lhb: %d rows, %d symbols, +%d cols -> panel",
                len(df),
                df["symbol"].nunique(),
                len(panel.columns) - before,
            )
            return panel
        else:
            logger.info("lhb: no usable data from fetch, direct cache read")
    except Exception as exc:
        logger.info("lhb: fetch failed (%s), direct cache read", exc)
    cache_dir = os.path.join(ALT_CACHE_DIR, "lhb")
    caches = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    if not caches:
        logger.warning("lhb: no cache files")
        return panel
    main_cache = None
    for f in caches:
        if "20240102_20260727" in f:
            main_cache = f
            break
    if main_cache is None:
        main_cache = caches[-1]
    try:
        raw = pd.read_parquet(main_cache)
        logger.info(
            "lhb: read cache %s: %d rows, %d cols",
            os.path.basename(main_cache),
            len(raw),
            len(raw.columns),
        )
        if "date" not in raw.columns:
            for col in raw.columns:
                if col == "symbol" or col.startswith("_"):
                    continue
                try:
                    sample = raw[col].dropna().iloc[0] if len(raw) > 0 else None
                    if sample is not None:
                        s = str(sample).strip()
                        if len(s) == 10 and s.count("-") == 2:
                            raw["date"] = pd.to_datetime(raw[col], errors="coerce")
                            break
                        if len(s) == 8 and s.isdigit():
                            raw["date"] = pd.to_datetime(
                                raw[col], format="%Y%m%d", errors="coerce"
                            )
                            break
                except (IndexError, KeyError, TypeError):
                    pass
            if "date" not in raw.columns and len(raw.columns) > 3:
                col = raw.columns[3]
                raw["date"] = pd.to_datetime(raw[col], errors="coerce")
                logger.info("lhb: mapped date from column[3]")
        if "date" in raw.columns and "symbol" in raw.columns:
            raw = raw.dropna(subset=["date"])
            merge_cols = ["symbol", "date"]
            exclude = set(merge_cols + list(raw.columns[raw.columns.duplicated()]))
            avail = [
                c
                for c in raw.columns
                if c not in exclude and not str(c).startswith("_")
            ]
            before = len(panel.columns)
            panel = panel.merge(raw[merge_cols + avail], on=merge_cols, how="left")
            logger.info(
                "lhb: direct merge: +%d cols -> panel", len(panel.columns) - before
            )
        else:
            logger.warning("lhb: cannot determine date column")
    except Exception as exc:
        logger.warning("lhb: direct cache read failed: %s", exc)
    return panel


def merge_sector_index(panel, refresh=False):
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    try:
        df = supply.fetch_sector_index(
            start_date=panel["date"].min().strftime("%Y%m%d"),
            end_date=panel["date"].max().strftime("%Y%m%d"),
            refresh=refresh,
        )
        if len(df) == 0:
            logger.warning("sector_index: empty result")
            return panel
        if "industry" not in panel.columns:
            logger.warning("sector_index: no industry column")
            return panel
        if panel["industry"].nunique() <= 1 and panel["industry"].iloc[0] == "UNKNOWN":
            logger.warning("sector_index: industry all UNKNOWN, skip")
            return panel
        name_to_code = {
            name: code
            for code, name in df[["index_code", "index_name"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        ind_map = {}
        for ind_name in panel["industry"].dropna().unique():
            ind_clean = str(ind_name).strip()
            if ind_clean in name_to_code:
                ind_map[ind_clean] = ind_clean
            else:
                for sw_name in name_to_code:
                    sw_clean = str(sw_name).strip()
                    if ind_clean in sw_clean or sw_clean in ind_clean:
                        ind_map[ind_clean] = sw_clean
                        break
        if not ind_map:
            logger.warning("sector_index: no industry->SW mapping")
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
        logger.info(
            "sector_index: %d rows, +%d cols -> panel",
            len(df),
            len(panel.columns) - before,
        )
    except Exception as exc:
        logger.warning("sector_index skipped: %s", exc)
    return panel


def merge_holdernumber(panel, refresh=False):
    cache_dir = os.path.join(ALT_CACHE_DIR, "holdernumber")
    caches = glob.glob(os.path.join(cache_dir, "*.parquet"))
    if not caches:
        logger.warning("holdernumber: no cache files")
        return panel
    frames = []
    for fpath in caches:
        try:
            df_one = pd.read_parquet(fpath)
            if len(df_one) and "symbol" in df_one.columns:
                frames.append(df_one)
        except Exception as exc:
            logger.warning("holdernumber: error reading %s: %s", fpath, exc)
    if not frames:
        logger.warning("holdernumber: no valid cache data")
        return panel
    df = pd.concat(frames, ignore_index=True)
    logger.info("holdernumber: %d rows from %d cache files", len(df), len(caches))
    if "announce_date" in df.columns:
        hn_cols = [
            c
            for c in df.columns
            if c not in ("symbol", "date", "_ts_code", "announce_date")
        ]
        f = df[["symbol", "announce_date"] + hn_cols].copy()
        f = f.sort_values("announce_date")
        f["announce_date"] = pd.to_datetime(f["announce_date"])
        f = f.dropna(subset=["announce_date"])
        if len(f) == 0:
            logger.warning("holdernumber: no valid announce_date after dropna")
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
        logger.info("holdernumber: +%d cols -> panel", len(panel_p.columns) - before)
        return panel_p
    logger.warning("holdernumber: no announce_date column")
    return panel


def merge_holdertrade(panel, refresh=False):
    cache_dir = os.path.join(ALT_CACHE_DIR, "holdertrade")
    caches = glob.glob(os.path.join(cache_dir, "*.parquet"))
    if caches:
        frames = []
        for fpath in caches:
            try:
                df_one = pd.read_parquet(fpath)
                if len(df_one) and "symbol" in df_one.columns:
                    frames.append(df_one)
            except Exception as exc:
                logger.warning("holdertrade: error reading %s: %s", fpath, exc)
        if frames:
            df = pd.concat(frames, ignore_index=True)
            logger.info("holdertrade: %d rows from cache", len(df))
            if "announce_date" in df.columns and "sh_net_sign" in df.columns:
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
                logger.info(
                    "holdertrade: +%d cols -> panel", len(panel.columns) - before
                )
            return panel
    logger.warning("holdertrade: no cache available")
    return panel


def report_coverage(panel, alt_cols):
    logger.info("=" * 70)
    logger.info("COVERAGE REPORT")
    logger.info("=" * 70)
    logger.info(
        "Panel: %d rows, %d symbols, %d columns",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
    )
    dims = {
        "northbound": [c for c in alt_cols if c.startswith("north_")],
        "margin": [
            c for c in alt_cols if c.startswith("margin_") or c.startswith("short_")
        ],
        "fina_indicator": [
            c
            for c in alt_cols
            if c
            in (
                "roe",
                "roe_deducted",
                "roa",
                "net_margin",
                "gross_margin",
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
            )
        ],
        "lhb": [c for c in alt_cols if c.startswith("lhb_")],
        "holdernumber": [
            c for c in alt_cols if c in ("holder_count", "avg_shares_per_holder")
        ],
        "holdertrade": [c for c in alt_cols if c.startswith("sh_")],
        "sector_index": [c for c in alt_cols if c.startswith("sw_")],
    }
    for dim, cols in dims.items():
        if not cols:
            logger.info("  %-20s: NO columns present", dim)
            continue
        total = len(panel) * len(cols)
        cell_nan = panel[cols].isna().sum().sum() / total * 100 if total > 0 else 0
        nan_counts = panel[cols].isna().sum(axis=1)
        rows_with_data = (nan_counts < len(cols)).sum()
        row_pct = rows_with_data / len(panel) * 100
        logger.info(
            "  %-20s: %2d cols, %5.1f%% NaN, %5.1f%% rows have some data",
            dim,
            len(cols),
            cell_nan,
            row_pct,
        )


def main():
    logger.info("=" * 70)
    logger.info("BUILD FULLY ENRICHED PANEL (v2)")
    logger.info("=" * 70)
    panel = load_panel(PANEL_INPUT)
    # 清除 base panel 中已有的 alt 数据列, 避免 merge 时产生 _x/_y 重复列
    DROP_ALT_PATTERNS = (
        "margin_balance",
        "short_balance",
        "margin_buy_amt",
        "short_sell_vol",
        "sh_net_change_sign",
        "sh_change_amt_total",
        "north_net_buy",
        "north_buy_amt",
        "north_sell_amt",
        "lhb_net_buy",
        "lhb_buy_amt",
        "lhb_sell_amt",
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
        "holder_count",
        "avg_shares_per_holder",
        "announce_date",  # 由 merge_asof 重新引入
    )
    dropped = [c for c in panel.columns if c in DROP_ALT_PATTERNS]
    if dropped:
        panel = panel.drop(columns=dropped)
        logger.info("Dropped %d stale alt columns from base: %s", len(dropped), dropped)
    base_cols = set(panel.columns)
    panel = fix_industry(panel)
    logger.info("\n--- Merging alt data sources ---")
    panel = merge_northbound(panel)
    panel = merge_margin(panel)
    panel = merge_fina_indicator(panel)
    panel = merge_lhb(panel)
    panel = merge_sector_index(panel)
    panel = merge_holdernumber(panel)
    panel = merge_holdertrade(panel)
    new_cols = [c for c in panel.columns if c not in base_cols]
    logger.info("\n--- Summary ---")
    logger.info(
        "Final panel: %d rows, %d symbols, %d columns",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
    )
    logger.info("New columns added (%d): %s", len(new_cols), sorted(new_cols))
    report_coverage(panel, new_cols)
    os.makedirs(os.path.dirname(PANEL_OUTPUT) or ".", exist_ok=True)
    panel.to_parquet(PANEL_OUTPUT, index=False)
    size_mb = os.path.getsize(PANEL_OUTPUT) / 1024 / 1024
    logger.info("\nSaved: %s (%.1f MB)", PANEL_OUTPUT, size_mb)


if __name__ == "__main__":
    main()
