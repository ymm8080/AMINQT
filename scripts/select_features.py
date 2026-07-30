#!/usr/bin/env python
"""Layer 2: Select model features per board from Layer1 output.

Usage:
  python scripts/select_features.py --board main --status       # View current
  python scripts/select_features.py --board main --update       # Run + diff + confirm
  python scripts/select_features.py --board main --update --dry-run  # Preview only
  python scripts/select_features.py --board main --keep         # Keep current
  python scripts/select_features.py --board main --history      # Version list
  python scripts/select_features.py --board main --rollback <id> # Rollback
"""

import argparse
import glob
import json
import os
import sys
import time
import logging

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.feature_selector import (
    FeatureSelector,
    BruteForceGenerator,
    nan_filter,
    dedup_l2,
    gate_d_ablation,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("select_features")

REGISTRY_DIR = "data/factor_registry"


def find_latest_features(board):
    """Find the latest Layer1 features parquet for a board."""
    pattern = os.path.join(REGISTRY_DIR, f"features_{board}_*.parquet")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No Layer1 features found for {board}. Run build_features.py first."
        )
    return files[0]


def load_panel_for_board(board, features_path):
    """Load the pre-built features parquet and add labels from source panel."""
    df = pd.read_parquet(features_path)
    # Ensure labels exist
    label = "label_pm_1d_net"
    if label not in df.columns:
        label = "label_1d_net"
    if label not in df.columns:
        # Need to rebuild labels from source
        panel = pd.read_parquet("data/panel_full_enriched_v3.parquet")
        # Use full 3Y for main, 1Y for dual
        if board == "main":
            pass  # 3Y
        else:
            cutoff = panel["date"].max() - pd.Timedelta(days=365)
            panel = panel[panel["date"] >= cutoff]
        # Align symbols
        symbols = set(df["symbol"].unique())
        panel = panel[panel["symbol"].isin(symbols)]
        from app.pipeline1.cleaning_pipeline import CleaningPipeline
        from app.pipeline1.label_engine import LabelEngine

        main_d, dual_d = CleaningPipeline().run_train(panel)
        source = (
            main_d
            if board == "main"
            else (dual_d if len(dual_d) > len(main_d) else main_d)
        )
        source = LabelEngine.build_path_labels(source)
        source = LabelEngine.build_labels(source)
        source = LabelEngine.mask_suspension(source)
        source = LabelEngine.mask_recent_days(source, days=3)
        # Merge labels
        label_cols = [c for c in source.columns if c.startswith("label_")]
        id_cols = ["symbol", "date"]
        df = df.merge(source[id_cols + label_cols], on=id_cols, how="left")
    return df, label


def run_selection(df, board, label, dry_run=False):
    """Run feature selection pipeline for the board."""
    FeatureSelector(registry_dir=REGISTRY_DIR)
    t0 = time.time()

    if board == "main":
        # Load current panel for BruteForce generation
        panel = pd.read_parquet("data/panel_full_enriched_v3.parquet")
        symbols = set(df["symbol"].unique())
        panel = panel[panel["symbol"].isin(symbols)]
        from app.pipeline1.cleaning_pipeline import CleaningPipeline

        main_d, _ = CleaningPipeline().run_train(panel)
        from app.pipeline1.label_engine import LabelEngine

        raw = LabelEngine.build_path_labels(main_d)
        raw = LabelEngine.build_labels(raw)
        raw = LabelEngine.mask_suspension(raw)
        raw = LabelEngine.mask_recent_days(raw, days=3)

        gen = BruteForceGenerator()
        new = gen.generate(raw)
        raw_exp = raw.join(new)
        all_feats = [
            c
            for c in raw_exp.columns
            if c not in BruteForceGenerator.EXCLUDE_COLS
            and not c.startswith("label_")
            and raw_exp[c].dtype in ("float64", "int64")
        ]
        valid = nan_filter(all_feats, raw_exp, 0.95)
        selected = dedup_l2(valid, raw_exp, 0.7)
        pipeline = "bruteforce_dedup"
        pool_size = len(valid)
    else:
        # DUAL: use FeatureEngineV35 curated features
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        all_feats = FeatureEngineV35.feature_columns(df)
        valid = nan_filter(all_feats, df, 0.95)
        selected = gate_d_ablation(
            valid, df, label_col=label, min_feats=30, sat_pct=0.95
        )
        pipeline = "gate_d"
        pool_size = len(valid)

    elapsed = time.time() - t0
    result = {
        "board": board,
        "pipeline": pipeline,
        "created": pd.Timestamp.now().isoformat(),
        "pool_size": pool_size,
        "selected_count": len(selected),
        "params": {"nan_threshold": 0.95, "dedup_threshold": 0.7}
        if board == "main"
        else {"min_features": 30, "saturation_pct": 0.95},
        "features": selected,
    }

    if not dry_run:
        return result, selected, elapsed

    logger.info(f"  Pool: {pool_size} -> Selected: {len(selected)} ({elapsed:.0f}s)")
    return result, selected, elapsed


def cmd_status(board):
    selector = FeatureSelector(registry_dir=REGISTRY_DIR)
    status = selector.get_status(board)
    print(json.dumps(status, indent=2, ensure_ascii=False))


def cmd_history(board):
    selector = FeatureSelector(registry_dir=REGISTRY_DIR)
    versions = selector.list_versions(board)
    current = None
    try:
        c = selector.load_current(board)
        current = (
            c.get("active_version", "")
            .replace(f"selected_{board}_", "")
            .replace(".json", "")
        )
    except Exception:
        pass

    print(f"{'Version':<25} {'Status':<12}")
    print("-" * 40)
    for v in versions:
        status = "ACTIVE" if v == current else ""
        print(f"{v:<25} {status:<12}")


def cmd_update(board, dry_run=False, draft_only=False):
    # Find Layer1 features
    feat_path = find_latest_features(board)
    logger.info(f"Layer1 features: {feat_path}")
    df, label = load_panel_for_board(board, feat_path)

    # Run new selection
    result, selected, elapsed = run_selection(df, board, label, dry_run=dry_run)

    if dry_run:
        print(f"Pool: {result['pool_size']}")
        print(f"Selected: {result['selected_count']}")
        print(f"Time: {elapsed:.0f}s")
        if result["selected_count"] < 30:
            print("Sample:", result["features"][:10])
        return

    # Compare with current
    selector = FeatureSelector(registry_dir=REGISTRY_DIR)
    try:
        current = selector.load_current(board)
        current_feats = current.get("features", [])
        diff = selector.diff_versions(current_feats, selected)
        print(f"\n  DIFF vs current ({current.get('created', '?')}):")
        print(
            f"    +{diff['added_count']} added, -{diff['removed_count']} removed (net {diff['net_change']:+d})"
        )
        if diff["sample_added"]:
            print(f"    Added sample: {diff['sample_added'][:3]}")
        if diff["sample_removed"]:
            print(f"    Removed sample: {diff['sample_removed'][:3]}")
    except FileNotFoundError:
        print("  (No current version - this will be the first)")

    print(
        f"\n  Pool: {result['pool_size']} -> Selected: {result['selected_count']} ({elapsed:.0f}s)"
    )

    if draft_only:
        path = selector.save_version(result, board, activate=False)
        print(f"  Saved as draft: {path}")
        return

    resp = input("\n  Save as new current? [y/N]: ").strip().lower()
    if resp == "y":
        path = selector.save_version(result, board, activate=True)
        print(f"  Activated: {path}")
    else:
        path = selector.save_version(result, board, activate=False)
        print(f"  Saved as draft: {path}")


def cmd_keep(board):
    selector = FeatureSelector(registry_dir=REGISTRY_DIR)
    try:
        status = selector.get_status(board)
        print(f"Keeping current version: {status.get('active_version', '?')}")
        print(f"  Selected: {status.get('selected_count', '?')} features")
    except Exception:
        print("No current version set.")


def cmd_rollback(board, version_id):
    selector = FeatureSelector(registry_dir=REGISTRY_DIR)
    selector.rollback(board, version_id)
    print(f"Rolled back {board} to {version_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, choices=["main", "dual"])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--rollback", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--draft-only", action="store_true")
    args = ap.parse_args()

    if args.status:
        cmd_status(args.board)
    elif args.history:
        cmd_history(args.board)
    elif args.rollback:
        cmd_rollback(args.board, args.rollback)
    elif args.keep:
        cmd_keep(args.board)
    elif args.update:
        cmd_update(args.board, dry_run=args.dry_run, draft_only=args.draft_only)
    else:
        print("Specify --status, --history, --update, --keep, or --rollback")
        print("Use --dry-run with --update for preview")


if __name__ == "__main__":
    main()
