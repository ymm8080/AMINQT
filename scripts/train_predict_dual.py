#!/usr/bin/env python
"""Focused DUAL board train + predict using 3-layer pipeline.

Reads Layer1 feature parquet + Layer2 selected features → LightGBM → prediction.

Usage:
  python scripts/train_predict_dual.py
  python scripts/train_predict_dual.py --tag 2026W31
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.predict_runner import find_bundles, run_prediction
from config.settings import data_others_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_predict_dual")

REGISTRY_DIR = "data/factor_registry"
REGISTRY_JSON_DIR = str(data_others_path("data/factor_registry"))
MODEL_DIR = "models/pipeline1"
PANEL_PATH = "data/panel_full_enriched_v3.parquet"


def train_dual(features_path: str, selected_path: str, tag: str) -> dict:
    """Train LightGBM on DUAL board using pre-built features and selected features."""
    logger.info("Loading features: %s", features_path)
    df = pd.read_parquet(features_path)

    with open(selected_path) as f:
        sel = json.load(f)
    feature_cols = sel["features"]
    logger.info(
        "DUAL: %d stocks, %d labels, %d features (Gate D, %s)",
        df["symbol"].nunique(),
        len([c for c in df.columns if c.startswith("label_")]),
        len(feature_cols),
        sel.get("pipeline", "?"),
    )

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    panels = {"dual": df}
    cols_by_board = {"dual": feature_cols}

    t0 = time.time()
    results = trainer.weekly_retrain(panels, cols_by_board, tag)
    elapsed = time.time() - t0

    for b, res in results.items():
        oos_1d = res["oos"]["ics"].get("1d_reg", 0)
        logger.info(
            "[%s] model=%s OOS_IC(1d)=%.4f switched=%s n_feats=%d time=%.0fs",
            b,
            os.path.basename(res["path"]),
            oos_1d,
            res["switched"],
            len(cols_by_board[b]),
            elapsed,
        )
    return results


def predict_dual(trade_date: str) -> pd.DataFrame | None:
    """Generate DUAL predictions using latest model."""
    bundles = find_bundles(MODEL_DIR)
    if "dual" not in bundles:
        logger.error("No dual model found.")
        return None
    logger.info("Model: dual -> %s", os.path.basename(bundles["dual"]))

    panel = pd.read_parquet(PANEL_PATH)
    # 1Y window for DUAL
    cutoff = panel["date"].max() - pd.Timedelta(days=365)
    panel = panel[panel["date"] >= cutoff]
    dual_panel = panel[panel["board"].isin(["GEM", "STAR"])]

    # Limit to 300 stocks for speed
    if dual_panel["symbol"].nunique() > 300:
        np.random.seed(42)  # 抽样可复现 (量化铁律)
        stocks = np.random.choice(
            dual_panel["symbol"].unique(), size=300, replace=False
        )
        dual_panel = dual_panel[dual_panel["symbol"].isin(stocks)]

    logger.info(
        "Prediction panel: %d stocks, %d rows (1Y window)",
        dual_panel["symbol"].nunique(),
        len(dual_panel),
    )

    result = run_prediction(
        dual_panel,
        trade_date,
        bundles,
        list_dir="data/lists",
        market_state="range",
        supply=None,
    )
    return result.get("list")


def main():
    ap = argparse.ArgumentParser(description="Train + Predict DUAL board")
    ap.add_argument("--tag", default=None, help="Model tag (default: ISO week)")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    trade_date = datetime.now().strftime("%Y%m%d")
    tag = args.tag
    if tag is None:
        iso = datetime.now().isocalendar()
        tag = f"{iso[0]}W{iso[1]:02d}"
    logger.info("DUAL pipeline: tag=%s trade_date=%s", tag, trade_date)

    do_all = not (args.train_only or args.predict_only)

    if args.train_only or do_all:
        print(f"\n{'=' * 60}")
        print(f"  STAGE 2-3: Train DUAL (tag={tag})")
        print(f"{'=' * 60}")

        # Find latest Layer1 features
        import glob

        feat_files = sorted(
            glob.glob(os.path.join(REGISTRY_DIR, "features_dual_*.parquet")),
            reverse=True,
        )
        if not feat_files:
            print(
                "ERROR: No feature parquet. Run: python scripts/build_features.py --board dual"
            )
            sys.exit(1)

        # Find latest Layer2 selected
        sel_files = sorted(
            glob.glob(os.path.join(REGISTRY_JSON_DIR, "selected_dual_20*.json")),
            reverse=True,
        )
        sel_file = sel_files[0] if sel_files else None
        if not sel_file:
            print(
                "ERROR: No selected features. Run: python scripts/select_features.py --board dual --update"
            )
            sys.exit(1)

        print(f"  Features: {feat_files[0]}")
        print(f"  Selection: {sel_file}")
        train_dual(feat_files[0], sel_file, tag)

    if args.predict_only or do_all:
        print(f"\n{'=' * 60}")
        print(f"  STAGE 4: Predict DUAL ({trade_date})")
        print(f"{'=' * 60}")
        lst = predict_dual(trade_date)
        if lst is not None and len(lst):
            cols = ["symbol", "board", "pred_ret_1d", "prob_up", "score"]
            available = [c for c in cols if c in lst.columns]
            print(lst[available].head(20).to_string(index=False))
            print(f"\n  Total candidates: {len(lst)}")
        else:
            print("  No DUAL candidates (safety valve or empty)")

    print(f"\nDone: {trade_date} tag={tag}")


if __name__ == "__main__":
    main()
