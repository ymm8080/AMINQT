# -*- coding: utf-8 -*-
"""Parallel brute-force feature selection for MAIN board.

Splits eligible raw columns into N shards. Each shard:
  1. Generates brute-force features ONLY for its raw columns
  2. Runs nan_filter + dedup_l2
  3. Saves selected features to JSON

After all shards complete, merge and train.

Usage:
  # Step 1: Compute features once
  python _train_board.py --board main --prepare-only --features-out data/tmp/main_features.parquet

  # Step 2: Run N parallel shard agents
  python _select_features_main.py --shard 0 --total 4 --features-in data/tmp/main_features.parquet
  python _select_features_main.py --shard 1 --total 4 --features-in data/tmp/main_features.parquet
  ...

  # Step 3: Train with merged features
  python _train_board.py --board main --train-only --features-in data/tmp/main_features.parquet --selected-dir data/tmp
"""
from __future__ import annotations

import argparse, json, logging, os, sys, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("select_shard")

import pandas as pd
import numpy as np
from app.pipeline1.feature_selector import BruteForceGenerator, nan_filter, dedup_l2


def main():
    parser = argparse.ArgumentParser(description="Parallel brute-force feature selection shard")
    parser.add_argument("--shard", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--total", type=int, required=True, help="Total number of shards")
    parser.add_argument("--features-in", required=True, help="Path to pre-computed features parquet")
    parser.add_argument("--out-dir", default="data/tmp", help="Output directory for selected features JSON")
    parser.add_argument("--nan-threshold", type=float, default=0.95)
    parser.add_argument("--dedup-threshold", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load pre-computed features
    t0 = time.time()
    df = pd.read_parquet(args.features_in)
    logger.info("Loaded features: %d cols, %d rows (%.1fs)", len(df.columns), len(df), time.time() - t0)

    # Get eligible raw columns and split into shards
    gen = BruteForceGenerator()
    all_raw = sorted(gen._eligible(df))
    logger.info("Eligible raw columns: %d total", len(all_raw))

    shard_size = (len(all_raw) + args.total - 1) // args.total
    start = args.shard * shard_size
    end = min(start + shard_size, len(all_raw))
    shard_cols = all_raw[start:end]

    if not shard_cols:
        logger.warning("Shard %d/%d has no columns (total=%d, range=%d:%d)",
                       args.shard, args.total, len(all_raw), start, end)
        # Write empty result
        out_path = os.path.join(args.out_dir, f"selected_shard_{args.shard:02d}.json")
        with open(out_path, "w") as f:
            json.dump([], f)
        return 0

    logger.info("Shard %d/%d: %d raw columns (%d:%d)",
                args.shard, args.total, len(shard_cols), start, end)
    logger.info("  Columns: %s ... %s", shard_cols[:3], shard_cols[-3:])

    # Generate brute-force features for this shard's raw columns
    t0 = time.time()
    new = gen.generate(df, raw_cols=shard_cols)
    logger.info("Generated %d brute-force features in %.0fs", len(new.columns), time.time() - t0)

    # Join and filter
    df_exp = df.join(new)
    brute_cols = list(new.columns)

    # nan filter
    t0 = time.time()
    valid = nan_filter(brute_cols, df_exp, args.nan_threshold)
    logger.info("NaN filter: %d -> %d (%.0fs)", len(brute_cols), len(valid), time.time() - t0)

    if not valid:
        logger.warning("No features survived NaN filter")
        out_path = os.path.join(args.out_dir, f"selected_shard_{args.shard:02d}.json")
        with open(out_path, "w") as f:
            json.dump([], f)
        return 0

    # dedup_l2
    t0 = time.time()
    selected = dedup_l2(valid, df_exp, args.dedup_threshold)
    logger.info("Dedup: %d -> %d (%.0fs)", len(valid), len(selected), time.time() - t0)

    # Save
    out_path = os.path.join(args.out_dir, f"selected_shard_{args.shard:02d}.json")
    with open(out_path, "w") as f:
        json.dump({"shard": args.shard, "total": args.total, "raw_cols": shard_cols,
                    "selected": selected, "n_brute": len(brute_cols), "n_valid": len(valid)},
                  f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s (%d features)", out_path, len(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
