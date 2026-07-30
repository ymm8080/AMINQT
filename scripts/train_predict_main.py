#!/usr/bin/env python
"""MAIN board train + predict using 3-layer pipeline.

Reads Layer1 feature parquet (column-filtered) + Layer2 selected features
→ LightGBM → prediction. No cross_sectional_rank for MAIN.

Usage:
  python scripts/train_predict_main.py
  python scripts/train_predict_main.py --tag 2026W31
  python scripts/train_predict_main.py --train-only
  python scripts/train_predict_main.py --predict-only
  python scripts/train_predict_main.py --max-stocks 100  # Test subset
"""

import argparse
import glob
import json
import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.predict_runner import find_bundles, run_prediction

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_predict_main")

REGISTRY_DIR = "data/factor_registry"
MODEL_DIR = "models/pipeline1"
PANEL_PATH = "data/panel_full_enriched_v3.parquet"


def find_latest_selection(board):
    """Find Layer2 selected_features.json for MAIN."""
    sel_files = sorted(
        glob.glob(os.path.join(REGISTRY_DIR, f"selected_{board}_20*.json")),
        reverse=True,
    )
    if not sel_files:
        raise FileNotFoundError(
            f"No selected_{board}_*.json. Run select_features.py first."
        )
    return sel_files[0]


def load_features_for_training(board, features_path, selected_features, max_stocks=0):
    """Load only the selected feature columns + id/label cols from Layer1 parquet.

    Avoids loading all 3,300 cols for MAIN — only reads ~1,000 selected + overhead.
    """
    import pyarrow.parquet as pq

    schema = pq.read_schema(features_path)
    all_names = [f.name for f in schema]

    # Always load: id cols, labels, selected features
    label_cols = [c for c in all_names if c.startswith("label_")]
    id_cols = ["symbol", "date", "board", "industry"]
    read_cols = [
        c for c in id_cols + label_cols + list(selected_features) if c in all_names
    ]
    missing = set(selected_features) - set(read_cols)
    if missing:
        logger.warning(
            f"  {len(missing)} selected features missing from parquet, will be skipped"
        )

    logger.info(
        f"  Loading {len(read_cols)} cols from {os.path.basename(features_path)} "
        f"({os.path.getsize(features_path) / 1024 / 1024:.0f}MB)"
    )
    df = pd.read_parquet(features_path, columns=read_cols)

    if max_stocks and max_stocks > 0 and df["symbol"].nunique() > max_stocks:
        stocks = sorted(
            np.random.choice(df["symbol"].unique(), size=max_stocks, replace=False)
        )
        df = df[df["symbol"].isin(stocks)]

    logger.info(
        f"  Training data: {len(df):,} rows, {df['symbol'].nunique()} stocks, {len(read_cols)} cols"
    )
    return df


def train_main(features_path, sel_path, tag, max_stocks=0):
    """Train LightGBM on MAIN board using pre-built features and Dedup L2 selection."""
    with open(sel_path) as f:
        sel = json.load(f)
    feature_cols = sel["features"]
    logger.info(
        "MAIN: %d features (%s), pool=%d",
        len(feature_cols),
        sel.get("pipeline", "?"),
        sel.get("pool_size", 0),
    )

    df = load_features_for_training("main", features_path, feature_cols, max_stocks)
    # Drop rows with NaN in ALL feature columns to avoid training failures
    df = df.dropna(subset=feature_cols, how="all")

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    t0 = time.time()
    results = trainer.weekly_retrain({"main": df}, {"main": feature_cols}, tag)
    elapsed = time.time() - t0

    for b, res in results.items():
        oos_1d = res["oos"]["ics"].get("1d_reg", 0)
        logger.info(
            "[%s] model=%s OOS_IC(1d)=%.4f switched=%s n_feats=%d time=%.0fs",
            b,
            os.path.basename(res["path"]),
            oos_1d,
            res["switched"],
            len(feature_cols),
            elapsed,
        )
    return results


def predict_main(trade_date, max_stocks=0):
    """Generate MAIN predictions using the latest MAIN model."""
    bundles = find_bundles(MODEL_DIR)
    if "main" not in bundles:
        logger.error("No main model found.")
        return None
    logger.info("Model: main -> %s", os.path.basename(bundles["main"]))

    panel = pd.read_parquet(PANEL_PATH)
    # No 1Y filter for MAIN — use full 3Y data
    main_panel = panel[~panel["board"].isin(["GEM", "STAR"])]

    if max_stocks and max_stocks > 0:
        stocks = sorted(
            np.random.choice(
                main_panel["symbol"].unique(), size=max_stocks, replace=False
            )
        )
        main_panel = main_panel[main_panel["symbol"].isin(stocks)]

    logger.info(
        "Prediction panel: %d stocks, %d rows",
        main_panel["symbol"].nunique(),
        len(main_panel),
    )

    # Pass only the "main" bundle — prediction pipeline handles main board
    result = run_prediction(
        main_panel,
        trade_date,
        bundles,
        list_dir="data/lists",
        market_state="range",
        supply=None,
    )
    return result.get("list")


def main():
    ap = argparse.ArgumentParser(description="Train + Predict MAIN board")
    ap.add_argument("--tag", default=None, help="Model tag (default: ISO week)")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--max-stocks", type=int, default=0, help="Cap stocks (0=all)")
    args = ap.parse_args()

    trade_date = datetime.now().strftime("%Y%m%d")
    tag = args.tag
    if tag is None:
        iso = datetime.now().isocalendar()
        tag = f"{iso[0]}W{iso[1]:02d}"
    logger.info(
        "MAIN pipeline: tag=%s trade_date=%s max_stocks=%s",
        tag,
        trade_date,
        args.max_stocks or "all",
    )

    do_all = not (args.train_only or args.predict_only)

    if args.train_only or do_all:
        print(f"\n{'=' * 60}")
        print(f"  MAIN Train (tag={tag})")
        print(f"{'=' * 60}")

        feat_files = sorted(
            glob.glob(os.path.join(REGISTRY_DIR, "features_main_*.parquet")),
            reverse=True,
        )
        if not feat_files:
            print(
                "ERROR: No feature parquet. Run: python scripts/build_features.py --board main"
            )
            sys.exit(1)

        sel_path = find_latest_selection("main")
        print(f"  Features: {feat_files[0]}")
        print(f"  Selection: {sel_path}")
        train_main(feat_files[0], sel_path, tag, args.max_stocks)

    if args.predict_only or do_all:
        print(f"\n{'=' * 60}")
        print(f"  MAIN Predict ({trade_date})")
        print(f"{'=' * 60}")
        lst = predict_main(trade_date, args.max_stocks)
        if lst is not None and len(lst):
            cols = ["symbol", "board", "pred_ret_1d", "prob_up", "score"]
            available = [c for c in cols if c in lst.columns]
            print(lst[available].head(20).to_string(index=False))
            print(f"\n  Total: {len(lst)} candidates")
        else:
            print("  No MAIN candidates (safety valve or empty)")

    print(f"\nDone: {trade_date} tag={tag}")


if __name__ == "__main__":
    main()
