"""Rebuild full enriched panel with ALL alt data properly merged.

Handles edge cases:
- fina_indicator: filter null announce_dates before merge_asof
- sector_index: avoid duplicate index_name column, use industry name mapping
- northbound: date-level broadcast merge (data only covers early 2024)
- margin: per-stock date merge
- lhb: use clean-cache with English col names
"""

import warnings

warnings.filterwarnings("ignore")
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.pipeline1.cleaning_pipeline import board_of  # noqa: E402

# ── 1. Load base panel ──
panel = pd.read_parquet("data/panel_full_enriched.parquet")
logger.info(
    "Base panel: %d stocks, %d rows, %d cols",
    panel["symbol"].nunique(),
    len(panel),
    len(panel.columns),
)

# ── 2. Fix industry using Tushare ──
pro = None
try:
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    pro = supply._tushare_pro()
except Exception:
    pass

industry_fixed = False
if pro is not None:
    try:
        basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
        if basic is not None and len(basic):
            basic["symbol"] = (
                basic["ts_code"]
                .str.replace(".SZ", "", regex=False)
                .str.replace(".SH", "", regex=False)
            )
            ind_map = dict(
                zip(basic["symbol"], basic["industry"].fillna("综合"), strict=False)
            )
            panel["industry"] = panel["symbol"].map(ind_map).fillna("综合")
            industry_fixed = True
            logger.info(
                "Industry fixed via Tushare: %d unique", panel["industry"].nunique()
            )
    except Exception as e:
        logger.warning("Tushare industry failed: %s", e)

if not industry_fixed:
    from app.pipeline1.data_supply import _ak_call

    try:
        import akshare as ak

        boards = _ak_call(ak.stock_board_industry_name_em)
        ind_map = {}
        for board_name in boards["板块名称"].astype(str):
            try:
                cons = _ak_call(ak.stock_board_industry_cons_em, symbol=board_name)
                for code in cons["代码"].astype(str).str[-6:]:
                    ind_map[code] = board_name
            except Exception:
                pass
            time.sleep(0.2)
        if ind_map:
            panel["industry"] = panel["symbol"].map(ind_map).fillna("综合")
            logger.info(
                "Industry fixed via AKShare: %d unique", panel["industry"].nunique()
            )
    except Exception as e:
        logger.warning("AKShare industry failed: %s", e)

# ── 3. Ensure board ──
if "board" not in panel.columns:
    panel["board"] = panel["symbol"].map(board_of)
    logger.info("Board column added")

# ── 4. CYQ enrich (skip if already present) ──
CYQ_COLS = [
    "winner_ratio",
    "avg_cost",
    "pct_70_low",
    "pct_70_high",
    "pct_70_con",
    "pct_90_low",
    "pct_90_high",
    "pct_90_con",
    "cost_5pct",
    "cost_15pct",
    "cost_50pct",
    "cost_85pct",
    "cost_95pct",
    "weight_avg",
]
has_cyq = all(c in panel.columns for c in CYQ_COLS)
if has_cyq:
    logger.info("CYQ already in base (%d cols) -- skipping enrich", len(CYQ_COLS))
else:
    from app.pipeline1.panel_builder import enrich_cyq  # noqa: E402

    panel = enrich_cyq(panel, cyq_cache="data/cyq_panel.parquet")
    logger.info("After CYQ enrich: %d cols", len(panel.columns))

date_min = panel["date"].min().strftime("%Y%m%d")
date_max = panel["date"].max().strftime("%Y%m%d")
logger.info("Panel date range: %s - %s", date_min, date_max)

# ── 5. Alt data: manual merge with cached data ──
CACHE = "data/supply_cache/alt_data"


def safe_date(df, col="date"):
    if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ─── 5a. Margin (两融) ───
margin_path = os.path.join(CACHE, "margin", f"{date_min}_{date_max}.parquet")
if os.path.exists(margin_path):
    margin = safe_date(pd.read_parquet(margin_path))
    margin = margin.drop_duplicates(subset=["symbol", "date"])
    margin_cols = ["symbol", "date"] + [
        c
        for c in margin.columns
        if c not in ("symbol", "date") and not c.startswith("_")
    ]
    panel = panel.merge(margin[margin_cols], on=["symbol", "date"], how="left")
    mc = (
        "margin_balance"
        if "margin_balance" in panel.columns
        else (margin_cols[2] if len(margin_cols) > 2 else None)
    )
    logger.info(
        "  margin: merged %d rows, %d cols -> NaN=%.1f%%",
        len(margin),
        len(margin_cols) - 2,
        panel[mc].isna().mean() * 100 if mc else 0,
    )
else:
    logger.warning("  margin: NO CACHE")

# ─── 5b. Northbound (北向) ───
nb_path = os.path.join(CACHE, "northbound", "20240102_20260727.parquet")
for cand in [
    nb_path,
    os.path.join(CACHE, "northbound", "all_20240102_20260727.parquet"),
]:
    if os.path.exists(cand):
        nb_path = cand
        break
if os.path.exists(nb_path):
    nb = safe_date(pd.read_parquet(nb_path))
    date_cols = [
        c for c in nb.columns if c not in ("symbol", "date") and not c.startswith("_")
    ]
    nb_by_date = nb[["date"] + date_cols].drop_duplicates(subset=["date"])
    # Report source data quality before merge
    for c in date_cols:
        src_nan = nb_by_date[c].isna().mean()
        logger.info(
            "  northbound source %s: NaN=%.1f%% (%d/%d dates have data)",
            c,
            src_nan * 100,
            nb_by_date[c].notna().sum(),
            len(nb_by_date),
        )
    panel = panel.merge(nb_by_date, on="date", how="left")
    for c in date_cols:
        if c in panel.columns:
            nan_rate = panel[c].isna().mean()
            logger.info("  northbound merged %s: NaN=%.1f%%", c, nan_rate * 100)
else:
    logger.warning("  northbound: NO CACHE")

# ─── 5c. LHB (龙虎榜) ───
# Use clean-cache "all_..." (English column names) instead of garbled "lhb_..." cache
lhb_path = os.path.join(CACHE, "lhb", f"all_{date_min}_{date_max}.parquet")
if os.path.exists(lhb_path):
    lhb = safe_date(pd.read_parquet(lhb_path))
    safe_cols = ["symbol", "date", "lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]
    avail = [c for c in safe_cols if c in lhb.columns]
    lhb_sub = lhb[avail].drop_duplicates(subset=["symbol", "date"])
    panel = panel.merge(lhb_sub, on=["symbol", "date"], how="left")
    logger.info(
        "  lhb: merged %d rows, %d cols -> NaN=%.1f%%",
        len(lhb_sub),
        len(avail) - 2,
        panel["lhb_net_buy"].isna().mean() * 100
        if "lhb_net_buy" in panel.columns
        else 0,
    )
else:
    logger.warning("  lhb: NO CACHE at %s, checking other paths...")
    # Fallback to lhb_ cache (may have garbled columns)
    lhb_path2 = os.path.join(CACHE, "lhb", f"lhb_{date_min}_{date_max}.parquet")
    if os.path.exists(lhb_path2):
        logger.warning("  lhb: found %s but may have garbled columns", lhb_path2)

# ─── 5d. Fina_indicator (基本面PIT) ───
fina_path = os.path.join(CACHE, "fina_indicator", f"all__{date_min}_{date_max}.parquet")
if not os.path.exists(fina_path):
    fina_path = os.path.join(
        CACHE, "fina_indicator", f"all_{date_min}_{date_max}.parquet"
    )
if os.path.exists(fina_path):
    fina = safe_date(pd.read_parquet(fina_path))
    if "announce_date" in fina.columns:
        fina["announce_date"] = pd.to_datetime(fina["announce_date"], errors="coerce")
        before = len(fina)
        fina = fina.dropna(subset=["announce_date"])
        dropped = before - len(fina)
        if dropped:
            logger.warning(
                "  fina_indicator: dropped %d rows with null announce_date", dropped
            )
        fina_cols = [
            c
            for c in fina.columns
            if c not in ("symbol", "announce_date", "report_period", "_ts_code")
        ]
        f = fina[["symbol", "announce_date"] + fina_cols].sort_values("announce_date")
        panel = panel.sort_values("date")
        panel = pd.merge_asof(
            panel,
            f,
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )
        roe_col = (
            "roe" if "roe" in panel.columns else (fina_cols[0] if fina_cols else None)
        )
        if roe_col:
            logger.info(
                "  fina_indicator: merged %d rows, %d cols -> NaN=%.1f%%",
                len(f),
                len(fina_cols),
                panel[roe_col].isna().mean() * 100,
            )
    else:
        logger.warning("  fina_indicator: cache has no announce_date")
else:
    logger.warning("  fina_indicator: NO CACHE")

# ─── 5e. Sector Index (申万行业指数) ───
sw_paths = [
    os.path.join(CACHE, "sector_index", f"sw_{date_min}_{date_max}.parquet"),
    os.path.join(CACHE, "sector_index", "sw_all_20260725.parquet"),
]
sw_path = next((p for p in sw_paths if os.path.exists(p)), None)
if sw_path and "industry" in panel.columns:
    sw = safe_date(pd.read_parquet(sw_path))
    name_to_code = dict(zip(sw["index_name"], sw["index_code"], strict=False))
    ind_map = {}
    for ind_name in panel["industry"].dropna().unique():
        if ind_name in name_to_code:
            ind_map[ind_name] = ind_name
        else:
            for sw_name in name_to_code:
                if ind_name in sw_name or sw_name in ind_name:
                    ind_map[ind_name] = sw_name
                    break
    if ind_map:
        panel["_sw_name"] = panel["industry"].map(ind_map)
        sw_data = sw.rename(
            columns={
                "ret_pct": "sw_ret_1d",
                "close": "sw_index_close",
                "volume": "sw_index_vol",
            }
        )
        sw_avail = [
            c
            for c in ["sw_ret_1d", "sw_index_close", "sw_index_vol"]
            if c in sw_data.columns
        ]
        panel = panel.merge(
            sw_data[["index_name", "date"] + sw_avail],
            left_on=["_sw_name", "date"],
            right_on=["index_name", "date"],
            how="left",
        )
        # Deduplicate columns (merge may add duplicate index_name)
        panel = panel.loc[:, ~panel.columns.duplicated()]
        panel = panel.drop(columns=["_sw_name"], errors="ignore")
        logger.info(
            "  sector_index: %d/%d industries mapped, %d cols -> sw_ret_1d NaN=%.1f%%",
            len(ind_map),
            panel["industry"].nunique(),
            len(sw_avail),
            panel["sw_ret_1d"].isna().mean() * 100
            if "sw_ret_1d" in panel.columns
            else 0,
        )
    else:
        logger.warning("  sector_index: no industry mapping found")
else:
    logger.warning("  sector_index: NO CACHE or no industry column")

# ── 6. Report per-dimension coverage ──
dimensions = [
    ("margin_", "DIM24 两融"),
    ("north_", "DIM25 北向"),
    ("lhb_", "DIM26 龙虎榜"),
    ("roe", "DIM22 基本面"),
    ("sw_ret_1d", "DIM28 行业指数"),
    ("benefit_", "DIM23 筹码(CYQ)"),
]
for prefix, name in dimensions:
    if prefix in ("roe", "sw_ret_1d"):
        # Single column check
        cols = [c for c in panel.columns if c == prefix]
    else:
        cols = [c for c in panel.columns if c.startswith(prefix)]
    if cols:
        nan_rate = panel[cols[0]].isna().mean()
        logger.info("  %s: %d cols, NaN=%.1f%%", name, len(cols), nan_rate * 100)
    else:
        logger.info("  %s: NO COLUMNS", name)

# Additional detail for fina_indicator
fina_dim_cols = [
    "roe",
    "roe_deducted",
    "roa",
    "gross_margin",
    "rev_yoy",
    "debt_ratio",
    "current_ratio",
    "asset_turnover",
    "ocf_to_or",
]
present = [c for c in fina_dim_cols if c in panel.columns]
if present:
    nan_rates = {c: panel[c].isna().mean() * 100 for c in present}
    avg_nan = np.mean(list(nan_rates.values()))
    logger.info(
        "  DIM22 基本面细项: %d/%d cols present, avg NaN=%.1f%%",
        len(present),
        len(fina_dim_cols),
        avg_nan,
    )
    for c, r in sorted(nan_rates.items()):
        logger.info("    %s: NaN=%.1f%%", c, r)

# ── 7. Save ──
out_path = "data/panel_full_enriched_v3.parquet"
panel.to_parquet(out_path, index=False)
file_mb = os.path.getsize(out_path) / 1024 / 1024
logger.info("=" * 60)
logger.info("SAVED: %s", out_path)
logger.info("  Size: %.1f MB", file_mb)
logger.info("  Stocks: %d", panel["symbol"].nunique())
logger.info("  Rows: %d", len(panel))
logger.info("  Cols: %d", len(panel.columns))
logger.info("=" * 60)
