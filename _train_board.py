# -*- coding: utf-8 -*-
"""Single-board training script with parallel feature selection support.

Usage:
  # Full pipeline (DUAL, or MAIN without parallelization):
  python _train_board.py --board main
  python _train_board.py --board dual

  # MAIN parallel workflow (brute-force features split across agents):
  # Step 1: Compute features once
  python _train_board.py --board main --prepare-only --features-out data/tmp/main_features.parquet
  # Step 2: Run N parallel shards (launch as background agents)
  python _select_features_main.py --shard 0 --total 4 --features-in data/tmp/main_features.parquet
  python _select_features_main.py --shard 1 --total 4 --features-in data/tmp/main_features.parquet
  ...
  # Step 3: Train with merged selections
  python _train_board.py --board main --train-only --features-in data/tmp/main_features.parquet --selected-dir data/tmp
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_board")

import pandas as pd
import numpy as np
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import FeatureSelector, BruteForceGenerator, nan_filter, dedup_l2
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.dual_track_trainer import DualTrackTrainer

MASK_RECENT_DAYS = 6
PANEL_PATH = "data/panel_full_enriched_v3.parquet"
MODEL_DIR = "models/pipeline1"
REGISTRY_PATH = "data/factor_registry"


def prepare_features(board: str) -> pd.DataFrame:
    """Load panel, clean, compute features + labels. Returns enriched DataFrame."""
    t0 = time.time()
    panel = pd.read_parquet(PANEL_PATH)
    logger.info("Panel: %d stocks, %d rows (%.1fs)", panel["symbol"].nunique(), len(panel), time.time() - t0)

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel, board=board)
    board_df = main_df if board == "main" else dual_df
    logger.info("Cleaning: %s=%d rows (%.1fs)", board, len(board_df), time.time() - t0)

    if len(board_df) == 0:
        raise RuntimeError(f"No samples after cleaning for board={board}")

    t0 = time.time()
    features = FeatureEngineV35()
    use_xrank = board != "main"
    df = features.build(board_df, cross_sectional_rank=use_xrank, registry=None)
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    logger.info("Features: %d cols, %d rows (%.1fs)", len(df.columns), len(df), time.time() - t0)
    return df


def train_with_features(df: pd.DataFrame, board: str, tag: str, picked: list[str]):
    """Train model with given features."""
    t0 = time.time()
    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    panels = {board: df}
    cols_by_board = {board: picked}
    results = trainer.weekly_retrain(panels, cols_by_board, tag)
    res = results.get(board, {})
    elapsed = time.time() - t0
    oos_1d = res.get("oos", {}).get("ics", {}).get("1d_reg", 0.0)
    logger.info("DONE [%s] path=%s OOS_IC(1d)=%.4f switched=%s n_feats=%d (train %.0fs)",
                board, res.get("path", "?"), oos_1d, res.get("switched", False),
                len(picked), elapsed)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", required=True, choices=["main", "dual"])
    parser.add_argument("--tag", default=None, help="Model tag (default: ISO week)")
    parser.add_argument("--no-ic-screen", action="store_true")

    # Parallel workflow
    parser.add_argument("--prepare-only", action="store_true", help="Compute features, save to parquet, exit")
    parser.add_argument("--features-out", default=None, help="Path to save prepared features (for --prepare-only)")
    parser.add_argument("--train-only", action="store_true", help="Load pre-computed features + merged selections, train")
    parser.add_argument("--features-in", default=None, help="Path to pre-computed features parquet (for --train-only)")
    parser.add_argument("--selected-dir", default="data/tmp", help="Dir with selected_shard_*.json files (for --train-only)")

    args = parser.parse_args()
    board = args.board
    tag = args.tag or time.strftime("%GW%V")
    logger.info("=== BOARD=%s TAG=%s ===", board, tag)

    # ── Prepare-only mode ──
    if args.prepare_only:
        out_path = args.features_out or f"data/tmp/{board}_features.parquet"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = prepare_features(board)
        df.to_parquet(out_path, index=False)
        logger.info("Saved prepared features: %s (%d cols, %d rows)", out_path, len(df.columns), len(df))
        return 0

    # ── Train-only mode (with merged parallel selections) ──
    if args.train_only:
        if not args.features_in:
            logger.error("--train-only requires --features-in")
            return 1
        t0 = time.time()
        df = pd.read_parquet(args.features_in)
        logger.info("Loaded features: %d cols, %d rows (%.1fs)", len(df.columns), len(df), time.time() - t0)

        # Merge shard selections (union — dedup_l2 is per-base-group, so no cross-shard conflicts)
        selected_dir = args.selected_dir
        all_selected = set()
        for fname in sorted(os.listdir(selected_dir)):
            if fname.startswith("selected_shard_") and fname.endswith(".json"):
                with open(os.path.join(selected_dir, fname)) as f:
                    data = json.load(f)
                    sel = data.get("selected", [])
                    all_selected.update(sel)
                    logger.info("  Shard %s: %d selected", fname, len(sel))

        if not all_selected:
            logger.warning("No selected features from shards, falling back to all FeatureEngineV35 features")
            picked = FeatureEngineV35.feature_columns(df)
        else:
            picked = sorted(all_selected)
            logger.info("Merged: %d total selected features from %d shards",
                        len(picked), sum(1 for f in os.listdir(selected_dir)
                                         if f.startswith("selected_shard_") and f.endswith(".json")))

        # Ensure all picked features exist in df (generate missing brute-force features)
        missing = [f for f in picked if f not in df.columns]
        if missing:
            logger.info("Generating %d missing brute-force features...", len(missing))
            gen = BruteForceGenerator()
            raw_cols = gen._eligible(df)
            new_feats = gen.generate(df, raw_cols=raw_cols)
            keep_cols = [c for c in missing if c in new_feats.columns]
            if keep_cols:
                df = df.join(new_feats[keep_cols])
                logger.info("  Injected %d brute-force features", len(keep_cols))
            still_missing = [f for f in picked if f not in df.columns]
            if still_missing:
                logger.warning("  %d features still missing after injection", len(still_missing))
        picked_final = [f for f in picked if f in df.columns]
        logger.info("Final training features: %d", len(picked_final))

        train_with_features(df, board, tag, picked_final)
        return 0

    # ── Full pipeline (default) ──
    df = prepare_features(board)

    # Feature Selection
    t0 = time.time()
    if not args.no_ic_screen:
        selector = FeatureSelector(registry_dir=REGISTRY_PATH)
        logger.info("FeatureSelector: %s=%s", board,
                     selector.config.get(board, {}).get("pipeline", "?"))
        try:
            selected = selector.select(df, board)
            missing = [f for f in selected if f not in df.columns]
            if missing:
                gen = BruteForceGenerator()
                raw_cols = gen._eligible(df)
                new_feats = gen.generate(df, raw_cols=raw_cols)
                keep_cols = [c for c in missing if c in new_feats.columns]
                if keep_cols:
                    df = df.join(new_feats[keep_cols])
            picked = [f for f in selected if f in df.columns]
            logger.info("FeatureSelection: %d/%d selected (%.0fs)", len(picked), len(selected), time.time() - t0)
        except Exception as exc:
            logger.error("FeatureSelector failed (%s), using all features", exc)
            picked = FeatureEngineV35.feature_columns(df)
    else:
        picked = FeatureEngineV35.feature_columns(df)

    train_with_features(df, board, tag, picked)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
