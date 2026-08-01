# -*- coding: utf-8 -*-
"""
DIM21-DIM29 特征 IC 评估脚本 (v2.0 — Full Panel)
====================================================
面向 panel_full_enriched_v2.parquet (3227 stocks, 77 cols with alt data).
- 跳过 CYQ/alt enrichment (已预合并)
- 跳过采样 (全量 3227)
- 丢弃乱码列 + 空稠列 (>95% NaN) 以节内存
- 全量 Rank IC 评估 dim21-28 所有特征 + _chgN 时序变化

用法:
    python scripts/eval_dim21_29_full.py

输出:
    data/factor_registry/dim21_29_full_{timestamp}.json
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

from config.settings import data_others_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PANEL_PATH = "data/panel_full_enriched_v3.parquet"
OUTPUT_DIR = str(data_others_path("data/factor_registry"))
EVAL_LOG_PATH = str(data_others_path("data/feature_evaluation_log.json"))

# Rank IC 参数
MIN_X_UNIQUE = 5
MIN_Y_UNIQUE = 2

# ──────────────────────────────────────────────
# Dimension -> Feature mapping (from FeatureEngineV35 source)
# ──────────────────────────────────────────────
DIM_FEATURES: dict[str, list[str]] = {
    "dim21_cyq": [
        "conc_90",
        "winner_ratio",
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
    "dim29_reserved": [],
}

# 需要丢弃的空稠列 (>95% NaN, 无有效信号)
HIGH_NAN_COLS = [
    "north_net_buy_sh",
    "north_buy_amt_sh",
    "north_sell_amt_sh",
    "north_net_buy_sz",
    "north_buy_amt_sz",
    "north_sell_amt_sz",
    "margin_balance",
    "short_balance",
    "margin_buy_amt",
    "short_sell_vol",
    "holder_count",
]

# 乱码列名 (AKShare 中文名称被编码破坏, >98% NaN)
GARBLED_COL_PATTERNS = []  # 将通过名称探测


# ──────────────────────────────────────────────
# Rank IC computation (lightweight)
# ──────────────────────────────────────────────
def _daily_ic_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    date_col: str = "date",
) -> pd.Series:
    """按 date 分组计算每日横截面 Spearman Rank IC."""
    from scipy.stats import spearmanr

    sub = df[[date_col, x_col, y_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _ic_single(group: pd.DataFrame) -> float:
        x = group[x_col]
        y = group[y_col]
        if len(x) < 2 or x.nunique() < MIN_X_UNIQUE or y.nunique() < MIN_Y_UNIQUE:
            return float("nan")
        try:
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


def _stock_ic_series(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    symbol_col: str = "symbol",
) -> pd.Series:
    """时序 IC: 逐股计算 Spearman Rank IC (特征 vs label 在时间序列上)."""
    from scipy.stats import spearmanr

    sub = df[[symbol_col, x_col, y_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _ic_single(g: pd.DataFrame) -> float:
        x = g[x_col]
        y = g[y_col]
        if len(x) < 10 or x.nunique() < MIN_X_UNIQUE or y.nunique() < MIN_Y_UNIQUE:
            return float("nan")
        try:
            return float(spearmanr(x, y).statistic)
        except (ValueError, TypeError):
            return float("nan")

    ics = (
        sub.groupby(symbol_col, observed=True)
        .apply(_ic_single, include_groups=False)
        .dropna()
    )
    return ics


def mean_timeseries_ic(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    symbol_col: str = "symbol",
    abs_mean: bool = True,
) -> float:
    """时序 IC 均值 (abs_mean=True 为强度)."""
    ics = _stock_ic_series(df, x_col, y_col, symbol_col)
    if ics.empty:
        return 0.0
    vals = ics.abs().values if abs_mean else ics.values
    return float(np.nanmean(vals))


def timeseries_ic_details(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    symbol_col: str = "symbol",
) -> dict:
    """时序 IC 详情: mean, std, pos_ratio, n_stocks."""
    ics = _stock_ic_series(df, x_col, y_col, symbol_col)
    if ics.empty:
        return {
            "mean": 0.0,
            "abs_mean": 0.0,
            "std": 0.0,
            "pos_ratio": 0.0,
            "n_stocks": 0,
        }
    return {
        "mean": round(float(ics.mean()), 5),
        "abs_mean": round(float(ics.abs().mean()), 5),
        "std": round(float(ics.std()), 5),
        "pos_ratio": round(float((ics > 0).mean()), 4),
        "n_stocks": len(ics),
    }


# ──────────────────────────────────────────────
# Panel cleaning & preparation
# ──────────────────────────────────────────────
def clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Drop garbled columns and near-empty alt data columns."""
    orig_cols = len(df.columns)

    # 1. Drop garbled Chinese columns (>95% NaN, unknown meaning)
    garbled = [
        c
        for c in df.columns
        if any(b > 127 for b in c.encode("utf-8", errors="replace"))
        and df[c].isna().mean() > 0.95
    ]
    if garbled:
        df = df.drop(columns=garbled)
        logger.info("Dropped %d garbled columns: %s", len(garbled), garbled[:5])

    # 2. Drop known high-NaN alt columns
    existing_high_nan = [c for c in HIGH_NAN_COLS if c in df.columns]
    if existing_high_nan:
        df = df.drop(columns=existing_high_nan)
        logger.info("Dropped %d high-NaN alt columns", len(existing_high_nan))

    # 3. Drop any other column with >98% NaN (likely unsalvageable)
    extra_drop = [
        c
        for c in df.columns
        if df[c].isna().mean() > 0.98
        and c not in ("symbol", "date", "industry", "board")
    ]
    if extra_drop:
        df = df.drop(columns=extra_drop)
        logger.info("Dropped %d extra >98%% NaN columns", len(extra_drop))

    # 4. Ensure scaffolding columns
    if "is_suspended" not in df.columns:
        df["is_suspended"] = False
    if "is_st" not in df.columns:
        df["is_st"] = False
    if "list_days" not in df.columns:
        df["list_days"] = df.groupby("symbol").cumcount() + 1

    # 5. Ensure limit_pct
    if "limit_pct" not in df.columns:
        from app.pipeline1.cleaning_pipeline import get_limit_pct

        df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])]

    logger.info("Cleaned panel: %d cols -> %d cols", orig_cols, len(df.columns))
    return df


def maybe_drop_sparse_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where ALL remaining alt+derived columns are NaN (memory save).

    Defines 'alt columns' as financial/CYQ/north/margin/holder columns.
    If a row has no alt data at all, it's not useful for dim21-28 eval.
    """
    alt_markers = {
        "winner_ratio",
        "pct_90_con",
        "roe",
        "gross_margin",
        "margin_balance_chg_1d",
        "north_net_buy_5d",
    }
    present_markers = [c for c in alt_markers if c in df.columns]

    if not present_markers:
        return df

    # Row is sparse if ALL present marker columns are NaN
    all_nan = df[present_markers].isna().all(axis=1)
    n_sparse = all_nan.sum()

    if n_sparse > 0:
        df = df[~all_nan].copy()
        logger.info(
            "Dropped %d sparse rows (all alt data NaN), remaining: %d",
            n_sparse,
            len(df),
        )

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
        df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])]
    return df


def _add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build minimal upstream features needed by DIM21-DIM28.

    dim20_chip_proxy 已合并到 dim21_chip_tushare (CYQ NaN 时自动 OHLCV 补位),
    不需要单独调用.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def build_dim21_29_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build DIM21-DIM28 features + labels (skip dim01-dim20 to save memory)."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine

    df = _ensure_scaffold(df)
    logger.info(
        "Starting feature build with %d rows, %d cols", len(df), len(df.columns)
    )

    # Build upstream support features (minimal set)
    t0 = time.time()
    df = _add_basic_features(df)
    logger.info("Basic features: %d cols in %.1fs", len(df.columns), time.time() - t0)

    # Build DIM21-DIM28 features
    t0 = time.time()
    df = FeatureEngineV35.dim21_chip_tushare(df)
    logger.info("dim21 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim22_fundamental_pit(df)
    logger.info("dim22 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim23_shareholder_structure(df)
    logger.info("dim23 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim24_margin_trading(df)
    logger.info("dim24 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim25_northbound(df)
    logger.info("dim25 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim26_lhb_enhanced(df)
    logger.info("dim26 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim27_industry_flow(df)
    logger.info("dim27 done: %d cols", len(df.columns))
    df = FeatureEngineV35.dim28_sector_index(df)
    logger.info("dim28 done: %d cols", len(df.columns))
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
    logger.info("Labels built in %.1fs", time.time() - t0)

    return df


# ──────────────────────────────────────────────
# IC evaluation per dimension
# ──────────────────────────────────────────────
def determine_dim_status(features: list[dict]) -> tuple[str, dict | None, float]:
    """Determine dimension status + best feature."""
    valid = [
        f
        for f in features
        if f.get("ic_1d", 0) > 0 or f.get("ic_3d", 0) > 0 or f.get("ic_5d", 0) > 0
    ]
    if not valid:
        return "SKIP", None, 100.0

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


def _evaluate_chg_variants(
    df: pd.DataFrame,
    base_name: str,
    label_cols: dict[str, str],
) -> dict | None:
    """Evaluate _chgN / _pct_chgN variants: returns best {kind, window, ic, ic_1d, ic_3d, ic_5d} or None."""
    chg_windows = [1, 3, 5, 10, 20]
    variants = [("_chg", "abs"), ("_pct_chg", "pct")]
    best_ic_val = 0.0
    best_window = None
    best_kind = "abs"

    for suffix, kind in variants:
        for w in chg_windows:
            chg_col = f"{base_name}{suffix}{w}"
            if chg_col not in df.columns:
                continue
            nan_pct = df[chg_col].isna().mean()
            if nan_pct > 0.95:
                continue

            ic1 = mean_rank_ic(df, chg_col, label_cols["1d"])
            ic3 = mean_rank_ic(df, chg_col, label_cols["3d"])
            ic5 = mean_rank_ic(df, chg_col, label_cols["5d"])
            avg_ic = np.mean([ic1, ic3, ic5])
            if avg_ic > best_ic_val:
                best_ic_val = avg_ic
                best_window = w
                best_kind = kind

    if best_window is None:
        return None

    best_suffix = "_chg" if best_kind == "abs" else "_pct_chg"
    chg_col = f"{base_name}{best_suffix}{best_window}"
    ic1 = mean_rank_ic(df, chg_col, label_cols["1d"])
    ic3 = mean_rank_ic(df, chg_col, label_cols["3d"])
    ic5 = mean_rank_ic(df, chg_col, label_cols["5d"])
    best_avg = np.mean([ic1, ic3, ic5])

    return {
        "kind": best_kind,
        "window": best_window,
        "ic": round(float(best_avg), 5),
        "ic_1d": round(float(ic1), 5),
        "ic_3d": round(float(ic3), 5),
        "ic_5d": round(float(ic5), 5),
    }


def evaluate_dimension(
    df: pd.DataFrame,
    dim_name: str,
    feature_names: list[str],
    label_cols: dict[str, str],
) -> dict:
    """Evaluate all features in one dimension (截面IC + 时序IC)."""
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
                    "ts_ic": {"abs_mean": 0.0, "pos_ratio": 0.0, "n_stocks": 0},
                    "nan_pct": nan_pct,
                    "best_chg": None,
                    "chg_vs_level": "level_wins",
                }
            )
            continue

        # === 截面 IC (cross-sectional per date) ===
        ic1 = mean_rank_ic(df, f, label_cols["1d"])
        ic3 = mean_rank_ic(df, f, label_cols["3d"])
        ic5 = mean_rank_ic(df, f, label_cols["5d"])

        # === 时序 IC (time-series per stock) ===
        ts_ic = timeseries_ic_details(df, f, label_cols["3d"])

        # === Evaluate _chgN / _pct_chgN variants ===
        best_chg = _evaluate_chg_variants(df, f, label_cols)

        # === chg vs level comparison ===
        level_ic_avg = np.mean([ic1, ic3, ic5])
        if best_chg and level_ic_avg > 0 and best_chg["ic"] > level_ic_avg * 1.1:
            chg_vs_level = "chg_wins"
        elif best_chg and best_chg["ic"] > 0:
            chg_vs_level = "level_wins"
        else:
            chg_vs_level = "no_chg_signal"

        feature_results.append(
            {
                "name": f,
                "ic_1d": round(float(ic1), 5),
                "ic_3d": round(float(ic3), 5),
                "ic_5d": round(float(ic5), 5),
                "ts_ic": ts_ic,
                "nan_pct": round(float(nan_pct), 2),
                "best_chg": best_chg,
                "chg_vs_level": chg_vs_level,
            }
        )

    status, best, best_nan = determine_dim_status(feature_results)
    best_ic = {"1d": 0.0, "3d": 0.0, "5d": 0.0}
    best_feat_name = None
    if best:
        best_feat_name = best["name"]
        best_ic = {"1d": best["ic_1d"], "3d": best["ic_3d"], "5d": best["ic_5d"]}

    logger.info(
        "  %s: %d/%d features, best=%s ic=%.4f/%.4f/%.4f -> %s  (NaN%%=%.1f)",
        dim_name,
        len(available),
        len(feature_names),
        best_feat_name or "N/A",
        best_ic["1d"],
        best_ic["3d"],
        best_ic["5d"],
        status,
        best_nan if best_nan else 100.0,
    )

    return {
        "status": status,
        "best_feature": best_feat_name,
        "best_ic": best_ic,
        "nan_pct": round(best_nan, 2) if best_nan is not None else 100.0,
        "features": feature_results,
    }


def compute_timeseries_analysis(dim_results: dict[str, dict]) -> dict:
    """Cross-dimension analysis: chg_vs_level + time-series IC summary."""
    all_features = []
    chg_wins_features = []
    ts_ic_pos_features = []
    for dim_name, dim_data in dim_results.items():
        for feat in dim_data.get("features", []):
            all_features.append(feat)
            if feat.get("chg_vs_level") == "chg_wins":
                chg_wins_features.append(f"{dim_name}/{feat['name']}")
            ts_ic = feat.get("ts_ic", {})
            if ts_ic.get("pos_ratio", 0) > 0.55 and ts_ic.get("abs_mean", 0) > 0.02:
                ts_ic_pos_features.append(f"{dim_name}/{feat['name']}")

    total = len(all_features)
    chg_win_rate = f"{len(chg_wins_features)}/{total}" if total > 0 else "0/0"
    ts_win_rate = f"{len(ts_ic_pos_features)}/{total}" if total > 0 else "0/0"

    return {
        "chg_win_rate": chg_win_rate,
        "chg_wins_count": len(chg_wins_features),
        "chg_total_count": total,
        "features_with_chg_win": chg_wins_features,
        "ts_ic_win_rate": ts_win_rate,
        "ts_ic_positive_count": len(ts_ic_pos_features),
        "features_with_ts_ic_positive": ts_ic_pos_features,
    }


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
        "type": "dim21_29_full_eval_v3",
        "script_version": output["script_version"],
        "panel_info": output["panel_info"],
        "boards": {
            bk: {
                "stocks": br["panel_info"]["stocks"],
                "dimensions": {
                    dim: {
                        "status": data["status"],
                        "best_feature": data.get("best_feature"),
                        "best_ic": data.get("best_ic"),
                    }
                    for dim, data in br["dimensions"].items()
                },
                "chg_win_rate": br["timeseries_analysis"]["chg_win_rate"],
                "ts_ic_win_rate": br["timeseries_analysis"]["ts_ic_win_rate"],
            }
            for bk, br in output.get("boards", {}).items()
        },
    }
    log.setdefault("entries", []).append(summary)
    with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    logger.info("Updated evaluation log: %s", EVAL_LOG_PATH)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    overall_start = time.time()

    # 1. Load the enriched panel
    logger.info("Loading panel from %s ...", PANEL_PATH)
    df = pd.read_parquet(PANEL_PATH)
    logger.info(
        "Panel loaded: %d stocks, %d dates, %d rows, %d cols",
        df["symbol"].nunique(),
        df["date"].nunique(),
        len(df),
        len(df.columns),
    )

    # 2. Clean panel: drop garbled/near-empty columns
    df = clean_panel(df)

    # 3. Drop rows where ALL alt data columns are NaN
    df = maybe_drop_sparse_rows(df)

    # 4. Build DIM21-29 features + labels
    df = build_dim21_29_features(df)

    # 5. Evaluate per-board (main/GEM/STAR)
    board_counts = {k: int(v) for k, v in df["board"].value_counts().items()}
    logger.info("Board distribution: %s", board_counts)

    # 6. Determine label columns
    label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    label_3d = "label_3d_net" if "label_3d_net" in df.columns else "label_3d"
    label_5d = "label_5d_net" if "label_5d_net" in df.columns else "label_5d"
    label_cols = {"1d": label_1d, "3d": label_3d, "5d": label_5d}
    logger.info("Label columns: %s", label_cols)

    # Panel info (full, pre-filter)
    full_panel_info = {
        "stocks": int(df["symbol"].nunique()),
        "dates": int(df["date"].nunique()),
        "rows": len(df),
        "total_cols": int(len(df.columns)),
        "full_panel": True,
        "pre_clean_cols": 77,
        "date_range": {
            "start": str(df["date"].min()),
            "end": str(df["date"].max()),
        },
        "board_distribution": {
            k: {
                "stocks": int(df[df["board"] == k]["symbol"].nunique()),
                "rows": int((df["board"] == k).sum()),
            }
            for k in sorted(board_counts)
        },
    }

    # 7. Evaluate each dimension per board group
    # main 单独; GEM+STAR (双创 20% 涨跌停) 合并
    board_groups = [
        ("main", df[df["board"] == "main"].copy()),
        ("gem_star", df[df["board"].isin(["GEM", "STAR"])].copy()),
    ]
    all_results: dict[str, dict] = {}
    for board_key, df_board in board_groups:
        n_stocks = df_board["symbol"].nunique()
        n_rows = len(df_board)
        if n_stocks < 10:
            logger.info(
                "Board group %s: %d stocks, %d rows — SKIP", board_key, n_stocks, n_rows
            )
            continue
        logger.info("=== %s: %d stocks, %d rows ===", board_key, n_stocks, n_rows)

        dim_results: dict[str, dict] = {}
        for dim_name, feats in DIM_FEATURES.items():
            if not feats:
                dim_results[dim_name] = {
                    "status": "NOT_IMPLEMENTED",
                    "best_feature": None,
                    "best_ic": {"1d": 0.0, "3d": 0.0, "5d": 0.0},
                    "nan_pct": 100.0,
                    "features": [],
                }
                continue
            dim_results[dim_name] = evaluate_dimension(
                df_board, dim_name, feats, label_cols
            )

        ts_analysis = compute_timeseries_analysis(dim_results)
        all_results[board_key] = {
            "panel_info": {
                "stocks": n_stocks,
                "dates": int(df_board["date"].nunique()),
                "rows": n_rows,
            },
            "dimensions": dim_results,
            "timeseries_analysis": ts_analysis,
        }

    # 8. Assemble output
    timestamp = pd.Timestamp.now().isoformat()
    output = {
        "timestamp": timestamp,
        "script_version": "3.0",
        "panel_path": PANEL_PATH,
        "panel_info": full_panel_info,
        "boards": all_results,
    }

    # 9. Save output
    timestamp_suffix = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"dim21_29_full_{timestamp_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)

    # 10. Update evaluation log
    _update_eval_log(output)

    # 11. Print summary — one table per board group
    elapsed = time.time() - overall_start
    print()
    print("=" * 130)
    print("DIM21-29 FULL PANEL IC EVALUATION (PER-BOARD)")
    print("=" * 130)
    print(
        f"  Total: {full_panel_info['stocks']} stocks x {full_panel_info['dates']} dates = {full_panel_info['rows']} rows"
    )
    print(
        f"  Columns: {full_panel_info['total_cols']} (from {full_panel_info['pre_clean_cols']} pre-clean)"
    )
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Boards: {full_panel_info['board_distribution']}")

    for board_key in ["main", "gem_star"]:
        if board_key not in all_results:
            continue
        br = all_results[board_key]
        pi = br["panel_info"]
        dims = br["dimensions"]
        ts = br["timeseries_analysis"]
        print()
        print(f"  ── {board_key.upper()} ({pi['stocks']} stocks, {pi['rows']} rows) ──")
        print(
            f"  {'Dimension':<22s} {'Status':<10s} {'Best Feature':<28s} {'CS_1d':<8s} {'CS_3d':<8s} {'CS_5d':<8s} {'TS_IC':<8s} {'NaN%':<7s}"
        )
        print("-" * 130)
        for dim_name, dim_data in dims.items():
            if dim_data["status"] == "SKIP" and dim_data.get("nan_pct", 0) >= 100:
                continue  # 全量 NaN 的维度不展示
            bf = dim_data.get("best_feature") or "-"
            ic = dim_data["best_ic"]
            npct = dim_data["nan_pct"]
            ts_ic_str = "-"
            for feat in dim_data.get("features", []):
                if feat["name"] == bf:
                    ts_ic_str = f"{feat.get('ts_ic', {}).get('abs_mean', 0):.4f}"
                    break
            mark = (
                " <<"
                if dim_data["status"] == "INCLUDE"
                else (" ?" if dim_data["status"] == "WATCH" else "")
            )
            print(
                f"  {dim_name:<22s} {dim_data['status']:<10s} {bf:<28s} "
                f"{ic['1d']:<8.4f} {ic['3d']:<8.4f} {ic['5d']:<8.4f} {ts_ic_str:<8s} {npct:<7.1f}{mark}"
            )
        print("-" * 130)
        print(f"  chg wins (abs/pct): {ts['chg_win_rate']}")
        print(f"  TS_IC positive:     {ts['ts_ic_win_rate']} (pos>0.55 & abs>0.02)")
        pos_chg = ts.get("features_with_chg_win", [])
        if pos_chg:
            print(
                f"  chg>level: {', '.join(pos_chg[:8])}{' ...' if len(pos_chg) > 8 else ''}"
            )
        ts_pos = ts.get("features_with_ts_ic_positive", [])
        if ts_pos:
            print(
                f"  TS_IC stable: {', '.join(ts_pos[:8])}{' ...' if len(ts_pos) > 8 else ''}"
            )

    print()
    print("=" * 130)
    # Cross-board comparison: key dims
    print("CROSS-BOARD IC COMPARISON (best feature per dim)")
    print("-" * 80)
    header = f"  {'Dim':<20s}"
    for bk in ["main", "gem_star"]:
        if bk in all_results:
            header += f" {bk + ' CS_3d':>14s} {bk + ' TS':>10s}"
    print(header)
    for dim_name in DIM_FEATURES:
        # 全量 SKIP 的维度不展示 (两板都没有数据)
        all_skip = True
        for bk in all_results:
            dd = all_results[bk]["dimensions"].get(dim_name, {})
            if dd.get("status") != "SKIP" or dd.get("nan_pct", 0) < 100:
                all_skip = False
                break
        if all_skip:
            continue
        parts = [f"  {dim_name:<20s}"]
        for bk in ["main", "gem_star"]:
            if bk not in all_results:
                continue
            dd = all_results[bk]["dimensions"].get(dim_name, {})
            cs3 = dd.get("best_ic", {}).get("3d", 0)
            bf = dd.get("best_feature")
            ts_val = 0.0
            if bf:
                for f in dd.get("features", []):
                    if f["name"] == bf:
                        ts_val = f.get("ts_ic", {}).get("abs_mean", 0)
                        break
            parts.append(f" {cs3:>14.4f} {ts_val:>10.4f}")
        print("".join(parts))
    print("=" * 130)

    # 12. Training readiness per board
    for board_key in ["main", "gem_star"]:
        if board_key not in all_results:
            continue
        dims = all_results[board_key]["dimensions"]
        print()
        print(f"TRAINING READINESS — {board_key.upper()}:")
        print("-" * 50)
        ready, watch, skip = [], [], []
        for dim_name, dim_data in dims.items():
            s = dim_data["status"]
            nf = len(dim_data.get("features", []))
            if s == "SKIP" and dim_data.get("nan_pct", 0) >= 100:
                continue  # 全量 NaN 的维度不展示
            best_ic_str = (
                f"IC={max(dim_data['best_ic'].values()):.4f}"
                if dim_data.get("best_ic")
                else "IC=0"
            )
            if s == "INCLUDE":
                ready.append(
                    f"  {dim_name}: {dim_data['best_feature']} ({best_ic_str}, {nf} feats)"
                )
            elif s in ("WATCH", "PARTIAL"):
                watch.append(
                    f"  {dim_name}: {dim_data['best_feature']} ({best_ic_str}, {nf} feats)"
                )
            else:
                skip.append(f"  {dim_name}: ({nf} feats)")
        print("READY:")
        for r in ready:
            print(r)
        if not ready:
            print("  None")
        print("WATCH:")
        for w in watch:
            print(w)
        if not watch:
            print("  None")
        print("SKIP:")
        for s in skip:
            print(s)
        if not skip:
            print("  None")
        print("=" * 50)


if __name__ == "__main__":
    main()
