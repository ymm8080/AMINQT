# -*- coding: utf-8 -*-
"""
DIM21-DIM29 特征 IC 评估脚本 (v1.0)
====================================
- 对 DIM21 至 DIM28 所有特征以及时序变化 (_chgN) 做逐因子 IC 评估
- 输出到 data/factor_registry/dim21_29_eval_{timestamp}.json
- 同时更新 data/feature_evaluation_log.json

用法:
    python scripts/eval_dim21_29.py

输出结构:
    {
      "timestamp": "...",
      "panel_info": {"stocks": N, "dates": N, "total_cols": N},
      "dimensions": {
        "dim21_cyq": {
          "status": "INCLUDE|SKIP|PARTIAL",
          "best_feature": "conc_90",
          "best_ic": {"1d": 0.15, "3d": 0.14, "5d": 0.15},
          "nan_pct": 29.6,
          "features": [...]
        },
        ...
      },
      "timeseries_analysis": {
        "chg_win_rate": "N/M",
        "features_with_positive_chg": [...]
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PANEL_PATH = "data/panel_3y.parquet"
PANEL_ENRICHED_PATH = "data/panel_full_enriched.parquet"
CYQ_CACHE = "data/cyq_panel.parquet"
OUTPUT_DIR = "data/factor_registry"
EVAL_LOG_PATH = "data/feature_evaluation_log.json"
SUPPLY_CACHE_DIR = "data/supply_cache"

# Rank IC 参数
MIN_X_UNIQUE = 5
MIN_Y_UNIQUE = 2
MIN_CROSS_SECTION = 10  # 单截面最小股票数
MIN_TOTAL_SAMPLES = 500  # 因子全局最小样本数

# ──────────────────────────────────────────────
# Dimension -> Feature mapping (from FeatureEngineV35 source)
# ──────────────────────────────────────────────
DIM_FEATURES: dict[str, list[str]] = {
    "dim21_cyq": [
        "conc_90",
        "benefit_part",
        "cost_bias",
        "cost_spread",
        "chip_skew",
        "conc_trend_20d",
        "benefit_trend_5d",
        "conc_streak",
        "conc_streak_3d",
        "conc70_streak",
        "conc70_streak_3d",
        "conc_reversal",
        "cost50_rank",
        "benefit_vs_ma60",
        "benefit_dir_5d",
    ],
    "dim22_finPIT": [
        "roe_qoq",
        "roa_qoq",
        "margin_chg",
        "growth_accel",
        "profit_accel",
        "debt_leveraging",
        "efficiency_chg",
        "ocf_stability",
        "roe_trend_4q",
        "margin_trend_4q",
        "rev_yoy_trend",
        "quality_momentum",
    ],
    "dim23_shareholder": [
        "holder_count_log",
        "holder_count_qoq",
        "holder_count_yoy",
        "holder_qoq_accel",
        "avg_shares_log",
        "avg_shares_qoq",
        "avg_shares_yoy",
        "holder_concentration_zscore",
    ],
    "dim24_margin": [
        "margin_balance_chg_1d",
        "margin_balance_chg_5d",
        "short_balance_ratio",
        "margin_buy_ratio",
        "margin_balance_ma20_dev",
        "margin_balance_yoy",
        "margin_pressure_score",
    ],
    "dim25_northbound": [
        "north_net_buy_5d",
        "north_net_buy_20d",
        "north_net_buy_streak",
        "north_buy_ratio",
        "north_sh_sz_divergence",
        "north_momentum_5d",
        "north_flow_zscore",
    ],
    "dim26_lhb": [
        "lhb_inst_net_buy_5d",
        "lhb_inst_net_buy_20d",
        "lhb_inst_count_5d",
        "lhb_inst_buy_ratio",
        "lhb_abnormal_score",
    ],
    "dim27_indflow": [
        "ind_margin_chg_5d",
        "ind_margin_accel",
        "ind_holder_trend_20d",
        "ind_north_chg_5d",
        "ind_lhb_net_flow_5d",
        "ind_capital_flow",
    ],
    "dim28_sector": [
        "sw_ret_1d",
        "sw_ret_5d",
        "sw_ret_20d",
        "sw_vol_20d",
        "sw_relative_strength",
        "sw_rotation_position",
        "sw_momentum_accel",
        "sw_turnover_anomaly",
    ],
    # dim29 - reserved (not yet implemented)
    "dim29_reserved": [],
}

# ──────────────────────────────────────────────
# Rank IC computation (lightweight, using scipy.stats.spearmanr)
# ──────────────────────────────────────────────
def _daily_ic_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
) -> pd.Series:
    """按 date 分组计算每日横截面 Spearman Rank IC."""
    sub = df[[date_col, x_col, y_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _ic_single(group: pd.DataFrame) -> float:
        x = group[x_col]
        y = group[y_col]
        if len(x) < 2 or x.nunique() < MIN_X_UNIQUE or y.nunique() < MIN_Y_UNIQUE:
            return float("nan")
        try:
            from scipy.stats import spearmanr

            return float(spearmanr(x, y).statistic)
        except (ValueError, TypeError):
            return float("nan")

    ics = (
        sub.groupby(date_col, observed=True)
        .apply(_ic_single, include_groups=False)
        .dropna()
    )
    return ics


def mean_rank_ic(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
    abs_mean: bool = True,
) -> float:
    """日度 Rank IC 序列均值 (abs_mean=True 为强度 IC)."""
    ics = _daily_ic_series(df, x_col, y_col, date_col)
    if ics.empty:
        return 0.0
    vals = ics.abs().values if abs_mean else ics.values
    return float(np.nanmean(vals))


# ──────────────────────────────────────────────
# Loading & enrichment
# ──────────────────────────────────────────────
def load_panel(path: str | None = None) -> pd.DataFrame:
    """Load base panel from parquet."""
    path = path or PANEL_PATH
    logger.info("Loading panel from %s ...", path)
    df = pd.read_parquet(path)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(
        "Panel loaded: %d stocks, %d dates, %d rows",
        df["symbol"].nunique(),
        df["date"].nunique(),
        len(df),
    )
    return df


def add_industry_board(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add industry classification and board info."""
    from app.pipeline1.cleaning_pipeline import board_of
    from app.pipeline1.data_supply import DataSupplyChain

    supply = DataSupplyChain()
    pro = supply._tushare_pro()

    try:
        basic = pro.stock_basic(
            exchange="", list_status="L", fields="ts_code,industry"
        )
        basic["symbol"] = basic["ts_code"].str.replace(".SZ", "").str.replace(
            ".SH", ""
        )
        ind_map = dict(zip(basic["symbol"], basic["industry"].fillna("综合")))
        df["industry"] = df["symbol"].map(ind_map).fillna("综合")
    except Exception as exc:
        logger.warning("Industry fetch failed: %s", exc)
        df["industry"] = "UNKNOWN"

    df["board"] = df["symbol"].map(board_of)
    logger.info(
        "Board distribution: %s",
        {k: int(v) for k, v in df["board"].value_counts().items()},
    )
    return df


def try_enrich_cyq(df: pd.DataFrame) -> pd.DataFrame:
    """Add CYQ chip distribution data via enrich_cyq()."""
    from app.pipeline1.panel_builder import enrich_cyq

    try:
        t0 = time.time()
        df = enrich_cyq(df, cyq_cache=CYQ_CACHE)
        cyq_cols = [
            c
            for c in df.columns
            if c.startswith("pct_") or c.startswith("cost_")
            or c in ("benefit_part", "weight_avg")
        ]
        logger.info(
            "CYQ enriched: +%d cols, %.1fs", len(cyq_cols), time.time() - t0
        )
    except Exception as exc:
        logger.warning("CYQ enrichment skipped: %s", exc)
        for c in DIM_FEATURES["dim21_cyq"]:
            if c not in df.columns:
                df[c] = np.nan
    return df


# ──────────────────────────────────────────────
# Feature building
# ──────────────────────────────────────────────
def _ensure_scaffold(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required scaffolding columns needed by FeatureEngine."""
    from app.pipeline1.cleaning_pipeline import get_limit_pct

    if "is_st" not in df.columns:
        df["is_st"] = False
    if "is_suspended" not in df.columns:
        df["is_suspended"] = False
    if "list_days" not in df.columns:
        df["list_days"] = df.groupby("symbol").cumcount() + 1
    if "limit_pct" not in df.columns:
        df["limit_pct"] = [
            get_limit_pct(b, d)
            for b, d in zip(df["board"], df["date"])
        ]
    return df


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal DIM01-DIM20 features required as upstream for DIM21-DIM28.

    This avoids building the full FeatureEngine (which OOMs in constrained env).
    """
    from app.pipeline1.feature_engine_v35 import (
        FeatureEngineV35,
        _apply_per_stock,
    )

    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Essential upstream features needed by dim21-28
    # dim20_chip_proxy: produces chip_concentration (OHLCV-derived, no API)
    df = FeatureEngineV35.dim20_chip_proxy(df)
    # dim21 needs board/limit info for cost50_rank (already in df)
    # dim27 needs industry (already in df)
    # dim28 needs industry + sw_ret (computed from OHLCV in dim28 itself)

    return df


def build_dim21_29_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build only DIM21-DIM28 features + labels (skip dim01-dim20 to save memory)."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine

    df = _ensure_scaffold(df)

    # Build upstream support features (minimal set)
    t0 = time.time()
    df = _add_basic_features(df)
    logger.info("Basic features: %d cols in %.1fs", len(df.columns), time.time() - t0)

    # Build DIM21-DIM28 features
    t0 = time.time()
    fe = FeatureEngineV35()
    # Call each dim21-28 directly (they're all @staticmethod)
    df = FeatureEngineV35.dim21_chip_tushare(df)
    df = FeatureEngineV35.dim22_fundamental_pit(df)
    df = FeatureEngineV35.dim23_shareholder_structure(df)
    df = FeatureEngineV35.dim24_margin_trading(df)
    df = FeatureEngineV35.dim25_northbound(df)
    df = FeatureEngineV35.dim26_lhb_enhanced(df)
    df = FeatureEngineV35.dim27_industry_flow(df)
    df = FeatureEngineV35.dim28_sector_index(df)
    df = FeatureEngineV35.industry_neutralize(df)
    df = FeatureEngineV35.add_missingness_flags(df)
    df = FeatureEngineV35._add_time_series_changes(df)
    logger.info(
        "DIM21-28 features: +%d cols in %.1fs",
        len(df.columns),
        time.time() - t0,
    )

    # Build labels
    t0 = time.time()
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)
    logger.info(
        "Labels: +%d cols in %.1fs",
        len(df.columns),
        time.time() - t0,
    )

    return df


# ──────────────────────────────────────────────
# IC evaluation per dimension
# ──────────────────────────────────────────────
def determine_dim_status(
    features: list[dict],
) -> tuple[str, dict | None, float]:
    """Determine dimension status + best feature from evaluated feature list.

    Returns (status, best_info, nan_pct_best).
    """
    valid = [
        f
        for f in features
        if f.get("ic_1d", 0) > 0 or f.get("ic_3d", 0) > 0 or f.get("ic_5d", 0) > 0
    ]

    if not valid:
        return "SKIP", None, 100.0

    # Score each feature by average IC across horizons
    for f in valid:
        f["_avg_ic"] = np.mean(
            [f.get("ic_1d", 0), f.get("ic_3d", 0), f.get("ic_5d", 0)]
        )
    valid.sort(key=lambda x: x["_avg_ic"], reverse=True)

    best = valid[0]
    best_ic = {"1d": best["ic_1d"], "3d": best["ic_3d"], "5d": best["ic_5d"]}
    max_ic = max(best_ic.values())
    best_nan = best["nan_pct"]

    if max_ic >= 0.02:
        status = "INCLUDE"
    elif max_ic >= 0.01:
        status = "WATCH"
    elif len(valid) >= 2:
        status = "PARTIAL"
    else:
        status = "SKIP"

    return status, best, best_nan


def evaluate_dimension(
    df: pd.DataFrame,
    dim_name: str,
    feature_names: list[str],
    label_cols: dict[str, str],
) -> dict:
    """Evaluate all features in one dimension.

    Args:
        df: full panel with features + labels
        dim_name: e.g. "dim21_cyq"
        feature_names: list of expected feature column names
        label_cols: {"1d": "label_1d_net", "3d": "label_3d_net", "5d": "label_5d_net"}

    Returns:
        dict with dimension evaluation results.
    """
    available = [c for c in feature_names if c in df.columns]
    if not available:
        return {
            "status": "SKIP",
            "best_feature": None,
            "best_ic": {"1d": 0.0, "3d": 0.0, "5d": 0.0},
            "nan_pct": 100.0,
            "features": [],
        }

    feature_results = []
    for f in available:
        nan_pct = float(df[f].isna().mean() * 100)
        if nan_pct > 95:
            feature_results.append(
                {
                    "name": f,
                    "ic_1d": 0.0,
                    "ic_3d": 0.0,
                    "ic_5d": 0.0,
                    "nan_pct": nan_pct,
                    "best_chg": None,
                    "timeseries_positive": False,
                }
            )
            continue

        ic1 = mean_rank_ic(df, f, label_cols["1d"])
        ic3 = mean_rank_ic(df, f, label_cols["3d"])
        ic5 = mean_rank_ic(df, f, label_cols["5d"])

        # Evaluate _chgN variants for this feature
        best_chg = _evaluate_chg_variants(df, f, label_cols)

        # Determine if any _chgN beats level IC
        timeseries_positive = False
        level_ic_avg = np.mean([ic1, ic3, ic5])
        if best_chg and level_ic_avg > 0:
            if best_chg["ic"] > level_ic_avg * 1.1:
                timeseries_positive = True

        feature_results.append(
            {
                "name": f,
                "ic_1d": round(float(ic1), 5),
                "ic_3d": round(float(ic3), 5),
                "ic_5d": round(float(ic5), 5),
                "nan_pct": round(float(nan_pct), 2),
                "best_chg": best_chg,
                "timeseries_positive": timeseries_positive,
            }
        )

    status, best, best_nan = determine_dim_status(feature_results)
    best_ic = {"1d": 0.0, "3d": 0.0, "5d": 0.0}
    best_feat_name = None
    if best:
        best_feat_name = best["name"]
        best_ic = {"1d": best["ic_1d"], "3d": best["ic_3d"], "5d": best["ic_5d"]}

    logger.info(
        "  %s: %d/%d features, best=%s ic=%.4f/%.4f/%.4f -> %s",
        dim_name,
        len(available),
        len(feature_names),
        best_feat_name or "N/A",
        best_ic["1d"],
        best_ic["3d"],
        best_ic["5d"],
        status,
    )

    return {
        "status": status,
        "best_feature": best_feat_name,
        "best_ic": best_ic,
        "nan_pct": round(best_nan, 2) if best_nan is not None else 100.0,
        "features": feature_results,
    }


def _evaluate_chg_variants(
    df: pd.DataFrame,
    base_name: str,
    label_cols: dict[str, str],
) -> dict | None:
    """Evaluate _chgN variants for a base feature.

    Returns the best chg variant {"window": N, "ic": max_ic} or None.
    """
    chg_windows = [1, 3, 5, 10, 20]
    best_ic_val = 0.0
    best_window = None

    for w in chg_windows:
        chg_col = f"{base_name}_chg{w}"
        if chg_col not in df.columns:
            continue
        nan_pct = df[chg_col].isna().mean()
        if nan_pct > 0.95:
            continue

        # Average IC across all three label horizons
        ic1 = mean_rank_ic(df, chg_col, label_cols["1d"])
        ic3 = mean_rank_ic(df, chg_col, label_cols["3d"])
        ic5 = mean_rank_ic(df, chg_col, label_cols["5d"])
        avg_ic = np.mean([ic1, ic3, ic5])

        if avg_ic > best_ic_val:
            best_ic_val = avg_ic
            best_window = w

    if best_window is None:
        return None

    # Recompute best window's IC precisely
    chg_col = f"{base_name}_chg{best_window}"
    ic1 = mean_rank_ic(df, chg_col, label_cols["1d"])
    ic3 = mean_rank_ic(df, chg_col, label_cols["3d"])
    ic5 = mean_rank_ic(df, chg_col, label_cols["5d"])
    best_avg = np.mean([ic1, ic3, ic5])

    return {
        "window": best_window,
        "ic": round(float(best_avg), 5),
        "ic_1d": round(float(ic1), 5),
        "ic_3d": round(float(ic3), 5),
        "ic_5d": round(float(ic5), 5),
    }


# ──────────────────────────────────────────────
# Time-series analysis (chgN across all features)
# ──────────────────────────────────────────────
def compute_timeseries_analysis(
    dim_results: dict[str, dict],
) -> dict:
    """Compute cross-dimension time-series analysis summary."""
    all_features = []
    positive_chg_features = []

    for dim_name, dim_data in dim_results.items():
        for feat in dim_data.get("features", []):
            all_features.append(feat)
            if feat.get("timeseries_positive"):
                positive_chg_features.append(f"{dim_name}/{feat['name']}")

    total = len(all_features)
    positive = len(positive_chg_features)
    chg_win_rate = f"{positive}/{total}" if total > 0 else "0/0"

    return {
        "chg_win_rate": chg_win_rate,
        "chg_positive_count": positive,
        "chg_total_count": total,
        "features_with_positive_chg": positive_chg_features,
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()

    # 1. Load panel (enriched panel has alt data pre-merged)
    #    Use sampling to fit memory constraints
    base_path = PANEL_ENRICHED_PATH if os.path.exists(PANEL_ENRICHED_PATH) else PANEL_PATH
    df = load_panel(base_path)

    # Sample to control memory usage (environment is memory-constrained)
    MAX_STOCKS = 300
    rng = np.random.RandomState(42)
    syms = rng.choice(df["symbol"].unique(), min(MAX_STOCKS, df["symbol"].nunique()), replace=False)
    df = df[df["symbol"].isin(syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(
        "Sampled to %d stocks (%d rows) for memory",
        df["symbol"].nunique(),
        len(df),
    )

    # 2. Add industry + board
    df = add_industry_board(df)

    # 3. Enrich CYQ (skip if already in enriched panel — avoids column collisions)
    #    enrich_cyq() merges columns like benefit_part, pct_90_con, cost_50pct.
    #    If these already exist (from panel_full_enriched.parquet), the merge
    #    creates suffixed duplicates (e.g. benefit_part_x/y), breaking downstream
    #    feature computation. Check by exact column name, not "conc_" prefix.
    CYQ_REQUIRED = {"pct_90_con", "benefit_part", "cost_50pct"}
    if not CYQ_REQUIRED.issubset(df.columns):
        df = try_enrich_cyq(df)
    else:
        logger.info("CYQ raw columns already present in panel (skip CYQ re-enrichment)")

    # 4. Enrich alt data via panel_builder (cache-backed, API fallback when needed)
    #    Uses existing caches: northbound (full range), margin (full range),
    #    fina_indicator (per-stock fallback for sampled stocks), LHB (partial).
    #    Excludes holdernumber/holdertrade per user request.
    from app.pipeline1.data_supply import DataSupplyChain
    from app.pipeline1.panel_builder import enrich_alt_data as panel_enrich_alt

    alt_markers = {"margin_balance", "north_net_buy", "lhb_net_buy", "roe"}
    if not alt_markers.intersection(df.columns):
        supply = DataSupplyChain(cache_dir=SUPPLY_CACHE_DIR)
        start_date = df["date"].min().strftime("%Y%m%d")
        end_date = df["date"].max().strftime("%Y%m%d")
        refresh = False  # prefer cache
        # 逐源 enrich 以便处理超时/失败不互相干扰
        for src in ["northbound", "margin", "fina_indicator", "lhb"]:
            try:
                t_src = time.time()
                df = panel_enrich_alt(
                    df, supply,
                    sources=[src],
                    start_date=start_date, end_date=end_date,
                    refresh=refresh,
                )
                elapsed = time.time() - t_src
                logger.info("  enrich[%s] done in %.1fs", src, elapsed)
            except Exception as exc:
                logger.warning("  enrich[%s] failed: %s — skipping", src, exc)
        # Log what we got
        new_markers = alt_markers.intersection(df.columns)
        logger.info("Alt enrichment complete. Markers present: %s", new_markers)
    else:
        logger.info("Alt data columns already present (skip enrichment)")
        logger.info("  Sample cols: %s", alt_markers.intersection(df.columns))

    # 5. Build DIM21-29 features + labels (skip heavy dim01-dim20 to save memory)
    df = build_dim21_29_features(df)

    # 6. Filter to main board only for IC computation
    board_counts_before = {k: int(v) for k, v in df["board"].value_counts().items()}
    df_main = df[df["board"] == "main"].copy()
    logger.info(
        "Main board filter: %d/%d rows (board dist: %s)",
        len(df_main),
        len(df),
        board_counts_before,
    )

    # 7. Determine label columns
    label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    label_3d = "label_3d_net" if "label_3d_net" in df.columns else "label_3d"
    label_5d = "label_5d_net" if "label_5d_net" in df.columns else "label_5d"
    label_cols = {"1d": label_1d, "3d": label_3d, "5d": label_5d}
    logger.info("Labels: %s", label_cols)

    # 8. Panel info
    panel_info = {
        "stocks": int(df_main["symbol"].nunique()),
        "dates": int(df_main["date"].nunique()),
        "rows": len(df_main),
        "total_cols": int(len(df.columns)),
        "sampled": int(df_main["symbol"].nunique()) < df["symbol"].nunique(),
        "date_range": {
            "start": str(df_main["date"].min()),
            "end": str(df_main["date"].max()),
        },
    }

    # 9. Evaluate each dimension
    dim_results: dict[str, dict] = {}
    logger.info("Evaluating dimensions ...")
    for dim_name, feats in DIM_FEATURES.items():
        if not feats:
            dim_results[dim_name] = {
                "status": "NOT_IMPLEMENTED",
                "best_feature": None,
                "best_ic": {"1d": 0.0, "3d": 0.0, "5d": 0.0},
                "nan_pct": 100.0,
                "features": [],
            }
            logger.info("  %s: NOT_IMPLEMENTED (no feature definitions)", dim_name)
            continue

        dim_results[dim_name] = evaluate_dimension(
            df_main, dim_name, feats, label_cols
        )

    # 10. Time-series analysis
    ts_analysis = compute_timeseries_analysis(dim_results)

    # 11. Assemble output
    timestamp = pd.Timestamp.now().isoformat()
    output = {
        "timestamp": timestamp,
        "script_version": "1.0",
        "panel_path": base_path,
        "panel_info": panel_info,
        "dimensions": dim_results,
        "timeseries_analysis": ts_analysis,
    }

    # 12. Save output
    timestamp_suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"dim21_29_eval_{timestamp_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)

    # 13. Update evaluation log
    _update_eval_log(output)

    # 14. Print summary
    elapsed = time.time() - overall_start
    print()
    print("=" * 100)
    print("DIM21-29 IC EVALUATION SUMMARY")
    print("=" * 100)
    print(f"  Panel: {panel_info['stocks']} stocks x {panel_info['dates']} dates = {panel_info['rows']} rows")
    print(f"  Features: {panel_info['total_cols']} total columns")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    print(
        f"{'Dimension':<22s} {'Status':<14s} {'Best Feature':<30s}"
        f" {'IC_1d':<8s} {'IC_3d':<8s} {'IC_5d':<8s} {'NaN%':<7s}"
    )
    print("-" * 100)
    for dim_name, dim_data in dim_results.items():
        bf = dim_data.get("best_feature") or "-"
        ic = dim_data["best_ic"]
        npct = dim_data["nan_pct"]
        mark = ""
        if dim_data["status"] == "INCLUDE":
            mark = " <<"
        elif dim_data["status"] == "WATCH":
            mark = " ?"
        print(
            f"  {dim_name:<22s} {dim_data['status']:<14s} {bf:<30s} "
            f"{ic['1d']:<8.4f} {ic['3d']:<8.4f} {ic['5d']:<8.4f} {npct:<7.1f}{mark}"
        )
    print("-" * 100)
    print(f"  chg win: {ts_analysis['chg_win_rate']}")
    if ts_analysis["features_with_positive_chg"]:
        print("  Positive chg features:")
        for f in ts_analysis["features_with_positive_chg"][:10]:
            print(f"    - {f}")
        if len(ts_analysis["features_with_positive_chg"]) > 10:
            print(f"    ... +{len(ts_analysis['features_with_positive_chg']) - 10} more")
    print("=" * 100)


def _update_eval_log(output: dict) -> None:
    """Append a summary entry to the evaluation log."""
    if not os.path.exists(EVAL_LOG_PATH):
        log = {"entries": []}
    else:
        try:
            with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
                log = json.load(f)
        except (json.JSONDecodeError, Exception):
            log = {"entries": []}

    summary = {
        "timestamp": output["timestamp"],
        "type": "dim21_29_eval",
        "script_version": output["script_version"],
        "panel_info": output["panel_info"],
        "dimension_summary": {
            dim: {
                "status": data["status"],
                "best_feature": data.get("best_feature"),
                "best_ic": data.get("best_ic"),
            }
            for dim, data in output["dimensions"].items()
        },
        "chg_win_rate": output["timeseries_analysis"]["chg_win_rate"],
        "chg_positive_count": output["timeseries_analysis"]["chg_positive_count"],
    }
    log.setdefault("entries", []).append(summary)
    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    logger.info("Updated evaluation log: %s", EVAL_LOG_PATH)


if __name__ == "__main__":
    main()
