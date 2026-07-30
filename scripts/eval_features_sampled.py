# -*- coding: utf-8 -*-
"""Sampled feature evaluation for main + dual boards (2024-now data).

Avoids OOM by sampling stocks per board and handling inf replacement column-wise.
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
# Sampling: use ~600 main + ~400 dual to stay within memory
N_SAMPLE_MAIN = 300
N_SAMPLE_DUAL = 200
RANDOM_SEED = 42


def sample_board(panel: pd.DataFrame, board: str, n: int) -> pd.DataFrame:
    """Sample n stocks from the given board."""
    from app.pipeline1.cleaning_pipeline import board_of

    if "board" not in panel.columns:
        panel = panel.copy()
        panel["board"] = panel["symbol"].map(board_of)
    board_syms = panel[panel["board"] == board]["symbol"].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    sampled = rng.choice(board_syms, min(n, len(board_syms)), replace=False)
    sub = panel[panel["symbol"].isin(sampled)].copy()
    logger.info("[%s] sampled %d stocks, %d rows", board, len(sampled), len(sub))
    return sub


def eval_board(board_name: str, board_df: pd.DataFrame, top_n: int = 50) -> dict:
    """Evaluate all features for one board."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine
    from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

    features = FeatureEngineV35()
    cross_sectional_rank = board_name != "main"

    # Filter to latest ~250 trading days for speed
    max_date = board_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=400)  # ~250 trading days
    board_df = board_df[board_df["date"] >= cutoff].copy()
    logger.info("[%s] Last ~250d rows (%s to %s): %d, stocks: %d",
                board_name, str(cutoff.date()), str(max_date.date()),
                len(board_df), board_df["symbol"].nunique())

    logger.info("[%s] Building features...", board_name)
    df = features.build(board_df, cross_sectional_rank=cross_sectional_rank)

    logger.info("[%s] Feature engine produced %d columns", board_name, len(df.columns))

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
    logger.info("[%s] %d features vs %s", board_name, len(feature_cols), label_col)

    results = []
    first_error = None
    for i, col in enumerate(feature_cols):
        valid = df[[col, label_col, "date"]].dropna()
        if len(valid) < 100:
            continue
        try:
            ic_series = daily_rank_ic_series(valid, col, label_col)
            if len(ic_series) < 5:
                continue
            ic_mean = mean_rank_ic(ic_series)
            ic_abs = mean_rank_ic(ic_series, abs_mean=True)
            ic_std = float(np.nanstd(ic_series.values)) if len(ic_series) else 0.0
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            pos_ratio = float((ic_series > 0).mean())
            nan_rate = float(board_df[col].isna().mean()) if col in board_df.columns else 1.0
            results.append({
                "factor": col,
                "ic_mean": round(float(ic_mean), 6),
                "ic_abs": round(float(ic_abs), 6),
                "ic_std": round(float(ic_std), 6),
                "icir": round(float(icir), 4),
                "pos_ratio": round(float(pos_ratio), 4),
                "nan_rate": round(float(nan_rate), 4),
                "n_dates": len(ic_series),
            })
        except Exception as e:
            if first_error is None:
                first_error = (col, str(e))
                logger.warning("[%s] First IC error on '%s': %s", board_name, col, e)
        if (i + 1) % 200 == 0:
            logger.info("[%s] %d/%d features (got %d results so far)",
                        board_name, i + 1, len(feature_cols), len(results))

    if not results:
        logger.error("[%s] NO features passed IC computation. First error: %s", board_name, first_error)
        return {"board": board_name, "n_features": 0, "n_strong": 0, "n_weak": 0, "n_noise": 0, "error": str(first_error)}

    results_df = pd.DataFrame(results).sort_values("ic_abs", ascending=False)
    logger.info("[%s] Done: %d features with valid IC", board_name, len(results_df))

    # Print top/bottom
    top = results_df.head(top_n)
    bottom = results_df.tail(top_n).iloc[::-1]

    print(f"\n{'=' * 90}")
    print(f"  {board_name.upper()} — Top {top_n} features by |IC| (2024-now, sampled)")
    print(f"{'=' * 90}")
    print(f"{'Rank':<5s} {'Factor':<45s} {'IC':>8s} {'|IC|':>8s} {'ICIR':>7s} {'Pos%':>7s} {'NaN%':>7s}")
    print("-" * 90)
    for idx, (_, r) in enumerate(top.iterrows(), 1):
        print(f"{idx:<5d} {r['factor']:<45s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>7.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}")

    print(f"\n{'=' * 90}")
    print(f"  {board_name.upper()} — Bottom {top_n} features by |IC| (noise/dead)")
    print(f"{'=' * 90}")
    print(f"{'Rank':<5s} {'Factor':<45s} {'IC':>8s} {'|IC|':>8s} {'ICIR':>7s} {'Pos%':>7s} {'NaN%':>7s}")
    print("-" * 90)
    for idx, (_, r) in enumerate(bottom.iterrows(), 1):
        print(f"{idx:<5d} {r['factor']:<45s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>7.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}")

    strong = (results_df["ic_abs"] >= 0.03).sum()
    weak = ((results_df["ic_abs"] >= 0.01) & (results_df["ic_abs"] < 0.03)).sum()
    noise = (results_df["ic_abs"] < 0.01).sum()
    print(f"\n  Summary: {strong} strong (|IC|>=0.03) | {weak} weak (0.01-0.03) | {noise} noise (<0.01)")

    return {
        "board": board_name,
        "n_features": len(results_df),
        "n_strong": int(strong),
        "n_weak": int(weak),
        "n_noise": int(noise),
        "top_20": results_df.head(20)["factor"].tolist(),
        "top_50": results_df.head(50).to_dict(orient="records"),
    }


def main():
    logger.info("Loading panel: %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    logger.info("Panel: %d stocks, %d rows, %d cols",
                panel["symbol"].nunique(), len(panel), len(panel.columns))

    from app.pipeline1.cleaning_pipeline import board_of

    if "board" not in panel.columns:
        panel["board"] = panel["symbol"].map(board_of)

    # Sample from each board separately
    main_sample = sample_board(panel, "main", N_SAMPLE_MAIN)
    dual_sample = sample_board(panel, "GEM", N_SAMPLE_DUAL)
    star_sample = sample_board(panel, "STAR", N_SAMPLE_DUAL // 2)
    dual_combined = pd.concat([dual_sample, star_sample], ignore_index=True)
    logger.info("Dual combined: %d stocks, %d rows",
                dual_combined["symbol"].nunique(), len(dual_combined))

    # Minimal cleaning: remove ST and short-history only, skip liquidity/top-N filters
    def clean_light(df: pd.DataFrame) -> pd.DataFrame:
        df = df[~df["is_st"].astype(bool)].copy()
        df = df[df["list_days"] >= 250].copy()
        return df

    main_clean = clean_light(main_sample)
    dual_clean = clean_light(dual_combined)
    logger.info("Cleaned: main=%d rows, dual=%d rows", len(main_clean), len(dual_clean))

    output = {"timestamp": datetime.now().isoformat(), "boards": {}}

    for board_name, board_df in [("main", main_clean), ("dual", dual_clean)]:
        if len(board_df) == 0:
            logger.warning("[%s] No samples, skipping", board_name)
            continue
        result = eval_board(board_name, board_df, top_n=50)
        output["boards"][board_name] = {
            "n_features": result["n_features"],
            "n_strong": result["n_strong"],
            "n_weak": result["n_weak"],
            "n_noise": result["n_noise"],
            "top_20": result["top_20"],
        }

    # Save full results
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    out_path = os.path.join(REGISTRY_DIR, f"feature_eval_sampled_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Evaluation saved to: %s", out_path)

    # Save detailed top-50 CSVs for review
    for board_name in ["main", "dual"]:
        if board_name not in output["boards"]:
            continue
        csv_path = os.path.join(REGISTRY_DIR, f"feature_eval_{board_name}_top50_{tag}.csv")
        pd.DataFrame(output["boards"][board_name].get("top_50", [])).to_csv(csv_path, index=False)
        logger.info("Top-50 CSV saved: %s", csv_path)


if __name__ == "__main__":
    main()
