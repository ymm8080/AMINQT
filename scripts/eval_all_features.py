# -*- coding: utf-8 -*-
"""Evaluate ALL V3 feature columns (not just top-20) on sampled data.
Small sample (100 stocks, latest 250d) to complete in reasonable time.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
REGISTRY_DIR = "data/factor_registry"
N_STOCKS = 120
RANDOM_SEED = 42


def clean_light(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def eval_all(board_name: str, board_df: pd.DataFrame) -> dict:
    """Evaluate EVERY feature column for one board."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine
    from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

    features = FeatureEngineV35()
    cross_sectional_rank = board_name != "main"

    # Latest ~250 trading days only
    max_date = board_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=400)
    board_df = board_df[board_df["date"] >= cutoff].copy()
    logger.info(
        "[%s] %s -> %s, %d rows, %d stocks",
        board_name,
        str(cutoff.date()),
        str(max_date.date()),
        len(board_df),
        board_df["symbol"].nunique(),
    )

    logger.info("[%s] Building features (this is the slow part)...", board_name)
    t0 = datetime.now()
    df = features.build(board_df, cross_sectional_rank=cross_sectional_rank)
    logger.info(
        "[%s] Feature engine: %d cols in %.0fs",
        board_name,
        len(df.columns),
        (datetime.now() - t0).total_seconds(),
    )

    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)

    feature_cols = FeatureEngineV35.feature_columns(df)
    feature_cols = [
        c for c in feature_cols if c in df.columns and df[c].dtype != object
    ]
    feature_cols = [c for c in feature_cols if df[c].isna().mean() < 0.95]

    label_col = "label_1d_net"
    if label_col not in df.columns:
        label_col = "label_1d"
    logger.info(
        "[%s] Evaluating %d features vs %s...", board_name, len(feature_cols), label_col
    )

    t1 = datetime.now()
    results = []
    first_error = None
    for i, col in enumerate(feature_cols):
        valid = df[[col, label_col, "date"]].dropna()
        if len(valid) < 50:
            continue
        try:
            ic_series = daily_rank_ic_series(valid, col, label_col)
            if len(ic_series) < 5:
                continue
            ic_mean = mean_rank_ic(valid, col, label_col)
            ic_abs = mean_rank_ic(valid, col, label_col, abs_mean=True)
            ic_std = float(np.nanstd(ic_series.values)) if len(ic_series) else 0.0
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            pos_ratio = float((ic_series > 0).mean())
            nan_rate = (
                float(board_df[col].isna().mean()) if col in board_df.columns else 1.0
            )
            results.append(
                {
                    "factor": col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_abs": round(float(ic_abs), 6),
                    "ic_std": round(float(ic_std), 6),
                    "icir": round(float(icir), 4),
                    "pos_ratio": round(float(pos_ratio), 4),
                    "nan_rate": round(float(nan_rate), 4),
                    "n_dates": len(ic_series),
                }
            )
        except Exception as e:
            if first_error is None:
                first_error = (col, str(e))
        if (i + 1) % 500 == 0:
            elapsed = (datetime.now() - t1).total_seconds()
            logger.info(
                "[%s] %d/%d (%.0fs, %d results)",
                board_name,
                i + 1,
                len(feature_cols),
                elapsed,
                len(results),
            )

    if not results:
        logger.error("[%s] NO results. First error: %s", board_name, first_error)
        return {"board": board_name, "n_features": 0, "error": str(first_error)}

    results_df = pd.DataFrame(results).sort_values("ic_abs", ascending=False)
    elapsed = (datetime.now() - t1).total_seconds()
    logger.info("[%s] Done: %d features in %.0fs", board_name, len(results_df), elapsed)

    # --- PRINT ---
    top_n = 80
    top = results_df.head(top_n)

    print(f"\n{'=' * 100}")
    print(f"  {board_name.upper()} — ALL FEATURES ranked by |IC| (top {top_n})")
    print(
        f"  {len(board_df['symbol'].unique())} stocks, {len(board_df)} rows, {len(feature_cols)} features evaluated"
    )
    print(f"{'=' * 100}")
    print(
        f"{'Rank':<5s} {'Factor':<50s} {'IC_mean':>8s} {'|IC|':>8s} {'ICIR':>7s} {'Pos%':>7s} {'NaN%':>7s}"
    )
    print("-" * 100)
    for idx, (_, r) in enumerate(top.iterrows(), 1):
        print(
            f"{idx:<5d} {r['factor']:<50s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>7.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}"
        )

    # --- STATS ---
    strong = (results_df["ic_abs"] >= 0.05).sum()  # higher threshold for small sample
    weak = ((results_df["ic_abs"] >= 0.02) & (results_df["ic_abs"] < 0.05)).sum()
    noise = (results_df["ic_abs"] < 0.02).sum()
    print(
        f"\n  Summary: {strong} strong (|IC|>=0.05) | {weak} weak (0.02-0.05) | {noise} noise (<0.02)"
    )
    print(
        f"  IC range: [{results_df['ic_mean'].min():+.4f}, {results_df['ic_mean'].max():+.4f}]"
    )

    return {
        "board": board_name,
        "n_features": len(results_df),
        "n_strong": int(strong),
        "n_weak": int(weak),
        "n_noise": int(noise),
        "top_50": results_df.head(50).to_dict(orient="records"),
        "ic_range": [
            float(results_df["ic_mean"].min()),
            float(results_df["ic_mean"].max()),
        ],
    }


def main():
    logger.info("Loading panel...")
    panel = pd.read_parquet(PANEL_PATH)
    logger.info(
        "Panel: %d stocks, %d rows, %d cols",
        panel["symbol"].nunique(),
        len(panel),
        len(panel.columns),
    )

    from app.pipeline1.cleaning_pipeline import board_of

    if "board" not in panel.columns:
        panel["board"] = panel["symbol"].map(board_of)

    rng = np.random.RandomState(RANDOM_SEED)

    # Sample separately per board
    for board_val, label, n in [
        ("main", "main", N_STOCKS),
        ("GEM", "dual", N_STOCKS // 2),
        ("STAR", "dual", N_STOCKS // 2),
    ]:
        pass  # sampling inline below

    # Main
    main_syms = panel[panel["board"] == "main"]["symbol"].unique()
    main_pick = rng.choice(main_syms, min(N_STOCKS, len(main_syms)), replace=False)
    main_df = clean_light(panel[panel["symbol"].isin(main_pick)].copy())

    # Dual (GEM + STAR)
    dual_mask = panel["board"].isin(["GEM", "STAR"])
    dual_syms = panel[dual_mask]["symbol"].unique()
    dual_pick = rng.choice(dual_syms, min(N_STOCKS, len(dual_syms)), replace=False)
    dual_df = clean_light(panel[panel["symbol"].isin(dual_pick)].copy())

    logger.info(
        "Main: %d stocks, %d rows | Dual: %d stocks, %d rows",
        main_df["symbol"].nunique(),
        len(main_df),
        dual_df["symbol"].nunique(),
        len(dual_df),
    )

    output = {"timestamp": datetime.now().isoformat(), "boards": {}}

    for board_name, board_df in [("main", main_df), ("dual", dual_df)]:
        if len(board_df) == 0:
            logger.warning("[%s] No data", board_name)
            continue
        result = eval_all(board_name, board_df)
        output["boards"][board_name] = result

    # Save
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    out_path = os.path.join(REGISTRY_DIR, f"feature_eval_all_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Full results saved: %s", out_path)


if __name__ == "__main__":
    main()
