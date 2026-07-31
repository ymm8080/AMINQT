#!/usr/bin/env python
"""Layer 2: Select model features per board from Layer1 output.

Usage:
  python scripts/select_features.py --board main --status       # View current
  python scripts/select_features.py --board main --update       # Run + diff + confirm
  python scripts/select_features.py --board main --update --dry-run  # Preview only
  python scripts/select_features.py --board main --keep         # Keep current
  python scripts/select_features.py --board main --history      # Version list
  python scripts/select_features.py --board main --rollback <id> # Rollback

Memory-safe for MAIN: reads only schema + sample rows from Layer1 parquet.
"""

import argparse
import glob
import json
import os
import sys
import time
import logging
import pyarrow.parquet as pq
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
    pattern = os.path.join(REGISTRY_DIR, f"features_{board}_*.parquet")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise FileNotFoundError(
            f"No Layer1 features found for {board}. Run build_features.py first."
        )
    return files[0]


def load_panel_for_board(board, features_path):
    """Load features for selection. MAIN: schema+sample only. DUAL: full load."""
    schema = pq.read_schema(features_path)
    all_names = [f.name for f in schema]
    n_rows = pq.ParquetFile(features_path).metadata.num_rows

    label = "label_pm_1d_net"
    if label not in all_names:
        label = "label_1d_net" if "label_1d_net" in all_names else None
    if label is None:
        raise RuntimeError(f"No label in {features_path}. Re-run build_features.py.")

    if board == "main":
        # Read only brute-force cols + label, sampled rows to avoid 5-8GB load
        brute_cols = [c for c in all_names if "_brute_" in c]
        read_cols = ["symbol", "date", label] + brute_cols
        logger.info(
            f"  MAIN Layer1: {len(brute_cols)} brute-force cols, "
            f"{n_rows:,} rows ({os.path.getsize(features_path) / 1024 / 1024:.0f}MB)"
        )

        # Read first ~25% of row groups for a representative sample
        pf = pq.ParquetFile(features_path)
        n_groups = pf.metadata.num_row_groups
        groups_to_read = min(n_groups, max(1, n_groups // 4))
        batches, total = [], 0
        for i in range(groups_to_read):
            tbl = pf.read_row_group(i, columns=read_cols)
            batches.append(tbl.to_pandas())
            total += len(batches[-1])
            if total >= 10000:
                break
        df = pd.concat(batches, ignore_index=True)
        if len(df) > 10000:
            df = df.sample(n=10000, random_state=42)
        logger.info(f"  Loaded {len(df):,} sample rows ({len(read_cols)} cols)")
        return df, label

    # DUAL: full load is safe (~59MB)
    df = pd.read_parquet(features_path)
    return df, label


def run_selection(df, board, label, dry_run=False):
    """Run feature selection. MAIN: uses sampled df from load_panel_for_board."""
    FeatureSelector(registry_dir=REGISTRY_DIR)
    t0 = time.time()

    if board == "main":
        all_feats = [
            c
            for c in df.columns
            if c not in BruteForceGenerator.EXCLUDE_COLS and not c.startswith("label_")
        ]
        if not all_feats:
            raise RuntimeError(
                "No brute-force columns in sample. Run build_features.py --board main first."
            )
        logger.info(
            f"  MAIN: {len(all_feats)} brute-force cols, {len(df):,} sample rows"
        )
        valid = nan_filter(all_feats, df, 0.95)
        selected = dedup_l2(valid, df, 0.7)
        pipeline = "bruteforce_dedup"
        pool_size = len(valid)
    else:
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
        "params": (
            {"nan_threshold": 0.95, "dedup_threshold": 0.7}
            if board == "main"
            else {"min_features": 30, "saturation_pct": 0.95}
        ),
        "features": selected,
    }

    if dry_run:
        logger.info(
            f"  Pool: {pool_size} -> Selected: {len(selected)} ({elapsed:.0f}s)"
        )
    return result, selected, elapsed


def cmd_status(board):
    sel = FeatureSelector(registry_dir=REGISTRY_DIR)
    print(json.dumps(sel.get_status(board), indent=2, ensure_ascii=False))


def cmd_history(board):
    sel = FeatureSelector(registry_dir=REGISTRY_DIR)
    versions = sel.list_versions(board)
    current = None
    try:
        c = sel.load_current(board)
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
        print(f"{v:<25} {'ACTIVE' if v == current else '':<12}")


def cmd_update(board, dry_run=False, draft_only=False, yes=False):
    feat_path = find_latest_features(board)
    logger.info(f"Layer1 features: {feat_path}")
    df, label = load_panel_for_board(board, feat_path)
    result, selected, elapsed = run_selection(df, board, label, dry_run=dry_run)

    if dry_run:
        print(f"Pool: {result['pool_size']}")
        print(f"Selected: {result['selected_count']}")
        print(f"Time: {elapsed:.0f}s")
        if result["selected_count"] < 30:
            print("Sample:", result["features"][:10])
        return

    sel = FeatureSelector(registry_dir=REGISTRY_DIR)
    try:
        current = sel.load_current(board)
        diff = sel.diff_versions(current.get("features", []), selected)
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
        path = sel.save_version(result, board, activate=False)
        print(f"  Saved as draft: {path}")
        return

    if yes:
        path = sel.save_version(result, board, activate=True)
        print(f"  Activated: {path}")
        return

    resp = input("\n  Save as new current? [y/N]: ").strip().lower()
    if resp == "y":
        path = sel.save_version(result, board, activate=True)
        print(f"  Activated: {path}")
    else:
        path = sel.save_version(result, board, activate=False)
        print(f"  Saved as draft: {path}")


def cmd_keep(board):
    sel = FeatureSelector(registry_dir=REGISTRY_DIR)
    try:
        s = sel.get_status(board)
        print(
            f"Keeping current: {s.get('active_version', '?')} ({s.get('selected_count', '?')} features)"
        )
    except Exception:
        print("No current version.")


def cmd_rollback(board, version_id):
    FeatureSelector(registry_dir=REGISTRY_DIR).rollback(board, version_id)
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
    ap.add_argument("--yes", "-y", action="store_true", help="Auto-confirm activation")
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
        cmd_update(
            args.board, dry_run=args.dry_run, draft_only=args.draft_only, yes=args.yes
        )
    else:
        print("Specify --status, --history, --update, --keep, or --rollback")
        print("Use --dry-run with --update for preview")


if __name__ == "__main__":
    main()
