"""IC 评估 — 按 DIM 分组, 单板块, 单 dim 组.

用法: python scripts/ic_eval_dim.py main dim01,dim02,dim03
      python scripts/ic_eval_dim.py dual dim21,dim22,dim23

轻量: 只计算指定 dim 的特征, IC 评估后输出部分结果 JSON.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.cleaning_pipeline import board_of
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.label_engine import LabelEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ic_dim")

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
REGISTRY_DIR = "data/factor_registry"
MIN_AMOUNT = 200_000
CACHE_DIR = "data/supply_cache/alt_data"


def merge_from_caches(df: pd.DataFrame) -> pd.DataFrame:
    """Merge enrichment from consolidated cache files using merge(on symbol+date)."""
    logger = logging.getLogger("ic_dim.cache")

    # Drop columns known to cause merge conflicts (will be replaced from cache)
    def _drop_if_exists(df, cols):
        return df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    # 1. Margin — consolidated (228k rows, 4911 symbols - good coverage)
    mg_path = os.path.join(CACHE_DIR, "margin_panel.parquet")
    if os.path.exists(mg_path):
        mg = pd.read_parquet(mg_path)
        df = _drop_if_exists(
            df, ["margin_balance", "margin_buy_amt", "short_balance", "short_sell_vol"]
        )
        df = df.merge(mg, on=["symbol", "date"], how="left")
        logger.info("margin: merged %d rows", len(mg))

    # 2. LHB — consolidated (53k rows, 5152 symbols)
    lhb_path = os.path.join(CACHE_DIR, "lhb", "all_20240102_20260727.parquet")
    if os.path.exists(lhb_path):
        lhb = pd.read_parquet(lhb_path)
        df = _drop_if_exists(df, ["lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"])
        df = df.merge(lhb, on=["symbol", "date"], how="left")
        logger.info("lhb: merged %d rows", len(lhb))

    # 3. Northbound — SKIPPED: cache is market-level data (no per-stock symbol)
    # The V3 panel columns (north_net_buy_sh etc.) have 0.2% coverage — insufficient

    return df


# Map dim names to FeatureEngineV35 method names
DIM_METHODS = {
    "dim01": "dim01_price_volume",
    "dim02": "dim02_volatility",
    "dim03": "dim03_fundamentals",
    "dim04": "dim04_sector_effect",
    "dim05": "dim05_turnover_liquidity",
    "dim06": "dim06_valuation_size",
    "dim07": "dim07_limit_gene",
    "dim08": "dim08_calendar_month",
    "dim09": "dim09_custom_formulas",
    "dim10": "dim10_money_flow",
    "dim11": "dim11_float_limits",
    "dim12": "dim12_ma_system",
    "dim13": "dim13_holiday",
    "dim14": "dim14_market_sentiment",
    "dim15": "dim15_alpha_factors",
    "dim16": "dim16_candlestick",
    "dim17": "dim17_extended_factors",
    "dim18": "dim18_lhb",
    "dim19": "dim19_amihud",
    "dim21": "dim21_chip_tushare",
    "dim22": "dim22_fundamental_pit",
    "dim23": "dim23_shareholder_structure",
    "dim24": "dim24_margin_trading",
    "dim25": "dim25_northbound",
    "dim26": "dim26_lhb_enhanced",
    "dim27": "dim27_industry_flow",
    "dim28": "dim28_sector_index",
    "dim29": "dim29_holdertrade",
    "dim30": "dim30_kline_geometry",
}

# dims that need float_shares_map
DIMS_NEED_FLOAT_SHARES = {"dim09"}

# dim groups for workload distribution (5 groups)
DIM_GROUPS = {
    # Smaller groups to avoid OOM; dependency-aware (dim07 must precede dim14)
    "g1_price_vol": ["dim01", "dim02", "dim03"],
    "g2_sector_val": ["dim04", "dim05", "dim06", "dim08"],
    "g3_limit_flow": ["dim07", "dim14", "dim09", "dim10"],  # dim14 needs dim07
    "g4_ma_alpha_moment": ["dim11", "dim12", "dim13", "dim15", "dim16", "dim17"],
    "g5_lhb_amihud": ["dim18", "dim19"],
    "g6_chip_fina_holder": ["dim21", "dim22", "dim23"],
    "g7_margin_north_lhb": ["dim24", "dim25", "dim26"],
    "g8_ind_kline": ["dim27", "dim28", "dim29", "dim30"],
}

# Dims that depend on other dims (prerequisites)
DIM_PREREQS = {
    "dim14": ["dim07"],  # market_sentiment needs is_limit_up from limit_gene
}


def compute_dim(
    df: pd.DataFrame,
    dim_name: str,
    fe: FeatureEngineV35,
    float_shares_map: dict | None = None,
) -> pd.DataFrame:
    """Compute a single dim on df."""
    method_name = DIM_METHODS[dim_name]
    method = getattr(fe, method_name)
    if dim_name in DIMS_NEED_FLOAT_SHARES and float_shares_map is not None:
        return method(df, float_shares_map)
    return method(df)


def main():
    if len(sys.argv) < 3:
        print("用法: python scripts/ic_eval_dim.py <board> <dim_group>")
        print("  board: main|dual")
        print(
            "  dim_group: g1_price_vol|g2_sector_turnover|g3_flow_ma_alpha|g4_candle_lhb_amihud|g5_chip_alt"
        )
        print("  OR dims: dim01,dim02,dim03")
        sys.exit(1)

    board = sys.argv[1]
    dim_spec = sys.argv[2]

    # Resolve dims
    if dim_spec in DIM_GROUPS:
        dims = DIM_GROUPS[dim_spec]
        group_name = dim_spec
    else:
        dims = [d.strip() for d in dim_spec.split(",")]
        group_name = "custom"

    invalid = [d for d in dims if d not in DIM_METHODS]
    if invalid:
        logger.error("Unknown dims: %s", invalid)
        sys.exit(1)

    t0 = time.time()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_id = f"{board}_{group_name}_{tag}"
    logger.info(
        "=== [%s] DIM IC Eval: %s (%s) ===", board.upper(), group_name, ",".join(dims)
    )

    # ---- Load & clean panel ----
    panel = pd.read_parquet(PANEL_PATH)
    if "board" not in panel.columns:
        panel = panel.copy()
        panel["board"] = panel["symbol"].map(board_of)

    # Merge from cache directories (fina/northbound/lhb/holder etc.)
    logger.info("Merging enrichment from caches...")
    panel = merge_from_caches(panel)
    logger.info("Panel after cache merge: %d cols", len(panel.columns))

    clean = panel[~panel["is_suspended"].astype(bool)]
    clean = clean.dropna(
        subset=["open", "high", "low", "close", "close_hfq", "volume", "amount"]
    )
    clean = clean[clean["amount"] >= MIN_AMOUNT]

    if board == "main":
        board_df = clean[clean["board"] == "main"].copy()
    else:
        board_df = clean[clean["board"].isin(["GEM", "STAR"])].copy()

    logger.info(
        "[%s] %d rows, %d stocks", board, len(board_df), board_df["symbol"].nunique()
    )

    if len(board_df) < 1000:
        logger.error("Too few samples")
        sys.exit(1)

    # ---- Compute ONLY assigned dims ----
    fe = FeatureEngineV35()
    df = board_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Snapshot columns BEFORE dim computation
    set(df.columns)

    # Ensure limit_pct exists (dim07 needs it); derive from up_limit_raw/pre_close
    if "limit_pct" not in df.columns:
        if "up_limit_raw" in df.columns and "pre_close" in df.columns:
            df["limit_pct"] = (df["up_limit_raw"] / df["pre_close"] - 1).clip(
                0.05, 0.30
            )
        else:
            from app.pipeline1.cleaning_pipeline import get_limit_pct

            df["limit_pct"] = [
                get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])
            ]
        logger.info("  (computed limit_pct: mean=%.3f)", df["limit_pct"].mean())

    logger.info("Computing %d dims: %s", len(dims), dims)

    # Resolve prerequisites — ensure dependency dims are computed first
    all_dims = list(dims)
    for d in dims:
        for prereq in DIM_PREREQS.get(d, []):
            if prereq not in all_dims:
                all_dims.insert(0, prereq)
                logger.info("  (auto-added prerequisite: %s for %s)", prereq, d)

    # Track per-dim added columns
    dim_columns = {}
    for dim_name in all_dims:
        pre_cols = set(df.columns)
        t_dim = time.time()
        df = compute_dim(df, dim_name, fe)
        added = set(df.columns) - pre_cols
        dim_columns[dim_name] = sorted(added)
        logger.info("  %s: %.1fs, +%d cols", dim_name, time.time() - t_dim, len(added))

    # ---- Labels ----
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)
    df = df.replace([np.inf, -np.inf], np.nan)

    if board != "main":
        df = fe._add_cross_sectional_ranks(df)

    # ---- ONLY screen columns added by the dims we computed ----
    cols_added = set()
    for added in dim_columns.values():
        cols_added.update(added)

    # Also allow feature_columns to identify numeric-only valid candidates
    candidates = [
        c
        for c in cols_added
        if c in df.columns
        and df[c].isna().mean() < 0.95
        and df[c].dtype != object
        and not c.startswith("label_")
    ]

    logger.info(
        "Dim-added columns: %d total, valid candidates: %d",
        len(cols_added),
        len(candidates),
    )

    if len(candidates) == 0:
        logger.error(
            "[%s] %s: 0 dim-added candidates! dim_columns=%s",
            board,
            group_name,
            {k: len(v) for k, v in dim_columns.items()},
        )
        sys.exit(1)

    # 标签覆盖检查
    label_col = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    nn = df[label_col].notna().mean()
    logger.info("%s non-null: %.1f%%", label_col, nn * 100)

    if nn < 0.01:
        logger.error("Label coverage too low!")
        sys.exit(1)

    # ---- IC screening (only dim-added columns) ----
    screener = ICScreener(registry_path=REGISTRY_DIR)
    result = screener.screen(df, candidates, window_id=window_id)

    strong = sum(1 for v in result["detail"].values() if v["grade"] == "strong")
    weak = sum(1 for v in result["detail"].values() if v["grade"] == "weak")
    dead = sum(1 for v in result["detail"].values() if v["grade"] == "dead")

    # ---- Per-dim IC breakdown ----
    per_dim = {}
    for dim_name, cols in dim_columns.items():
        dim_candidates = [c for c in cols if c in result["detail"]]
        if not dim_candidates:
            continue
        best_ic = max(
            max(
                result["detail"][c].get("ic_1d", 0),
                result["detail"][c].get("ic_3d", 0),
                result["detail"][c].get("ic_5d", 0),
            )
            for c in dim_candidates
        )
        best_factor = max(
            dim_candidates,
            key=lambda c: max(
                result["detail"][c].get("ic_1d", 0),
                result["detail"][c].get("ic_3d", 0),
                result["detail"][c].get("ic_5d", 0),
            ),
        )
        n_strong = sum(
            1 for c in dim_candidates if result["detail"][c]["grade"] == "strong"
        )
        n_weak = sum(
            1 for c in dim_candidates if result["detail"][c]["grade"] == "weak"
        )
        n_dead = sum(
            1 for c in dim_candidates if result["detail"][c]["grade"] == "dead"
        )
        per_dim[dim_name] = {
            "n_cols": len(cols),
            "n_candidates": len(dim_candidates),
            "n_strong": n_strong,
            "n_weak": n_weak,
            "n_dead": n_dead,
            "best_factor": best_factor,
            "best_ic": round(best_ic, 4),
        }

    # ---- Output ----
    elapsed = time.time() - t0

    # Per-dim summary
    print(f"\n{'=' * 85}")
    print(f"  [{board.upper()}] {group_name} — Per-Dim IC ({elapsed:.0f}s)")
    print(f"  candidates={len(candidates)} strong={strong} weak={weak} dead={dead}")
    print(f"{'=' * 85}")
    print(
        f"{'Dim':<12s} {'Cols':>5s} {'Cand':>5s} {'Str':>4s} {'Wk':>4s} {'Dead':>5s} {'Best IC':>8s} {'Best Factor'}"
    )
    print("-" * 85)
    for dim_name in all_dims:
        if dim_name in per_dim:
            p = per_dim[dim_name]
            print(
                f"{dim_name:<12s} {p['n_cols']:>5d} {p['n_candidates']:>5d} {p['n_strong']:>4d} {p['n_weak']:>4d} {p['n_dead']:>5d} {p['best_ic']:>8.4f}  {p['best_factor']}"
            )
        else:
            print(f"{dim_name:<12s} (no candidates)")

    # Top-15 dim-added only
    top = sorted(
        [(k, v) for k, v in result["detail"].items()],
        key=lambda kv: max(
            kv[1].get("ic_1d", 0), kv[1].get("ic_3d", 0), kv[1].get("ic_5d", 0)
        ),
        reverse=True,
    )[:15]
    print(
        f"\n{'Factor':<38s} {'IC_1d':>7s} {'IC_3d':>7s} {'IC_5d':>7s} {'AUC':>7s} {'RollM':>7s} {'ICIR':>6s} {'Grade'}"
    )
    print("-" * 85)
    for fname, detail in top:
        print(
            f"{fname:<38s} {detail.get('ic_1d', 0):>7.4f} {detail.get('ic_3d', 0):>7.4f} "
            f"{detail.get('ic_5d', 0):>7.4f} {detail.get('auc', 0):>7.4f} "
            f"{detail.get('rolling_mean', 0):>7.4f} {detail.get('icir', 0):>6.4f} "
            f"{detail['grade']:>7s}"
        )

    # Save partial results
    out = {
        "board": board,
        "dim_group": group_name,
        "dims": dims,
        "window_id": window_id,
        "elapsed_s": round(elapsed, 1),
        "n_rows": len(board_df),
        "n_stocks": int(board_df["symbol"].nunique()),
        "n_candidates": len(candidates),
        "n_strong": strong,
        "n_weak": weak,
        "n_dead": dead,
        "n_selected": len(result["factors"]),
        "top_20": [
            {
                "factor": f,
                "ic_1d": d.get("ic_1d", 0),
                "ic_3d": d.get("ic_3d", 0),
                "ic_5d": d.get("ic_5d", 0),
                "grade": d.get("grade", ""),
            }
            for f, d in top
        ],
    }
    out_path = os.path.join(REGISTRY_DIR, f"ic_dim_{window_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"\n[DONE] {out_path}")
    print(f"  strong={strong} weak={weak} dead={dead}")


if __name__ == "__main__":
    main()
