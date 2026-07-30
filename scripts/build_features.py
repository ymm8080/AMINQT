#!/usr/bin/env python
"""Layer 1: Build features per board from panel.

Usage:
  python scripts/build_features.py                           # Full build (registry + both boards)
  python scripts/build_features.py --board main               # MAIN only
  python scripts/build_features.py --board dual               # DUAL only
  python scripts/build_features.py --adoption-only             # Only sync registry
  python scripts/build_features.py --data-window 3Y            # Data window (1Y/3Y/ALL)
  python scripts/build_features.py --board main --max-stocks 100  # Test with 100 stocks
"""
import argparse, os, sys, time, logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import BruteForceGenerator
from app.pipeline1.feature_registry import FeatureRegistry

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_features")

REGISTRY_DIR = "data/factor_registry"
PANEL_PATH = "data/panel_full_enriched_v3.parquet"
os.makedirs(REGISTRY_DIR, exist_ok=True)


def load_panel(window=None):
    panel = pd.read_parquet(PANEL_PATH)
    if window == "1Y":
        cutoff = panel["date"].max() - pd.Timedelta(days=365)
        panel = panel[panel["date"] >= cutoff]
    elif window in ("3Y", None):
        pass  # use all available data (3 years)
    elif window == "ALL":
        pass
    return panel


def get_default_window(board):
    """MAIN uses 3Y rolling data, DUAL uses 1Y rolling data."""
    return "3Y" if board == "main" else "1Y"


def step1_update_registry(panel):
    """Sync registry with panel: add new columns, mark removed columns."""
    reg_path = os.path.join(REGISTRY_DIR, "feature_registry.json")
    registry = FeatureRegistry(path=reg_path)

    # Auto-seed if empty
    if not registry.features:
        logger.info("Registry empty, seeding from panel sample...")
        sample = (
            panel.groupby("symbol", group_keys=False)
            .apply(lambda g: g.head(min(30, len(g))))
            .reset_index(drop=True)
        )
        registry._seed(sample)

    # Adoption: add new
    logger.info("Checking for new panel columns...")
    added = 0
    removed = 0

    # Get registered source cols
    registered_cols = set()
    for name, meta in registry.get_all().items():
        for sc in meta.get("source_cols", []):
            registered_cols.add(sc)

    # Discover new numeric columns
    skip = {
        "symbol", "date", "board", "industry", "announce_date",
        "is_suspended", "is_st", "tradestatus", "open", "high", "low",
        "close", "volume", "amount", "pre_close", "turnover_rate",
    }
    panel_cols = [
        c
        for c in panel.columns
        if c not in skip
        and not c.startswith("label_")
        and panel[c].dtype in ("float64", "int64")
        and panel[c].isna().mean() < 0.70
    ]

    for col in panel_cols:
        if col not in registered_cols and col not in registry.get_all():
            registry.register_new(
                col,
                {
                    "dim_group": "_auto_adopted",
                    "source_cols": [col],
                    "status": "active",
                    "grade": "trial",
                    "registered_at": datetime.now().isoformat(),
                },
            )
            added += 1
            logger.info(f"  + {col}")

    # Remove stale columns (in registry but not in panel)
    for name, meta in list(registry.get_all().items()):
        srcs = meta.get("source_cols", [])
        if srcs and all(sc not in panel.columns for sc in srcs):
            if meta.get("status") != "removed":
                meta["status"] = "removed"
                meta["removed_at"] = datetime.now().isoformat()
                registry._data["features"][name] = meta
                removed += 1
                logger.info(f"  - {name} (source cols gone)")

    registry.save()
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    reg_out = os.path.join(REGISTRY_DIR, f"registry_{ts}.json")
    registry._save_as(reg_out)

    logger.info(f"Registry updated: +{added} added, -{removed} removed -> {reg_out}")
    return reg_out


def step2_build_board(panel, board="main", window="3Y", max_stocks=0):
    """Build features for a specific board.

    MAIN: family-based batching — generates one transform family at a time
    (pct_change→rolling_mean→...), joins incrementally. Peak ~3GB for 2k stocks.
    max_stocks=0 → all MAIN stocks.
    """
    import shutil as _shutil

    logger.info(f"Building {board} board features (window={window})...")
    t0 = time.time()

    # Apply data window before filtering by board
    if window == "1Y":
        cutoff = panel["date"].max() - pd.Timedelta(days=365)
        panel = panel[panel["date"] >= cutoff]
    # else: 3Y / ALL — use full panel

    if board == "main":
        # Full MAIN board (60/00/002/601/603/605), not just CSI 300
        board_panel = panel[~panel["board"].isin(["GEM", "STAR"])].copy()
        if max_stocks and max_stocks > 0:
            stocks = sorted(np.random.choice(
                board_panel["symbol"].unique(), size=max_stocks, replace=False))
            board_panel = board_panel[board_panel["symbol"].isin(stocks)]
        logger.info(f"  MAIN: {board_panel['symbol'].nunique():,} stocks")
    else:
        dual = panel[panel["board"].isin(["GEM", "STAR"])]
        board_stocks = sorted(dual["symbol"].unique())
        if len(board_stocks) > 300:
            board_stocks = sorted(np.random.choice(board_stocks, size=300, replace=False))
        board_panel = panel[panel["symbol"].isin(board_stocks)].copy()
        logger.info(f"  DUAL: {board_panel['symbol'].nunique():,} stocks")

    # Clean + labels (shared for both boards)
    cleaner = CleaningPipeline()
    main_d, dual_d = cleaner.run_train(board_panel)
    df = main_d if board == "main" else (dual_d if len(dual_d) > len(main_d) else main_d)
    logger.info(f"  Cleaned: {len(df):,} rows, {df['symbol'].nunique()} stocks")

    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    # Must mask >= 5 days for label_5d horizon to avoid look-ahead bias
    # (train_runner.py uses MASK_RECENT_DAYS=6, all eval scripts use 6)
    df = LabelEngine.mask_recent_days(df, days=6)

    if board == "main":
        # Family-based batching: generate one transform family at a time,
        # join to df incrementally. Peak: base + largest family ≈ 3GB.
        FAMILIES = ["pct_change", "rolling_mean", "rolling_std",
                     "rolling_max", "diff", "momentum", "EMA"]
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        logger.info(f"  BruteForce: {len(raw_cols)} eligible raw cols x {len(FAMILIES)} families")

        total_new_cols = 0
        for fam in FAMILIES:
            new = gen.generate_family(df, fam, raw_cols=raw_cols, dtype="float32")
            for c in new.columns:
                new[c] = new[c].replace([np.inf, -np.inf], np.nan).astype("float32")
            n_cols = len(new.columns)
            df = df.join(new)
            total_new_cols += n_cols
            del new
            logger.info(
                f"  Family [{fam}]: {n_cols} cols, "
                f"accumulated {total_new_cols} brute-force features"
            )
    else:
        fe = FeatureEngineV35()
        import tempfile
        reg_dir = tempfile.mkdtemp()
        registry = FeatureRegistry(path=os.path.join(reg_dir, "feature_registry.json"))
        sample = (
            df.groupby("symbol", group_keys=False)
            .apply(lambda g: g.head(min(30, len(g))))
            .reset_index(drop=True)
        )
        registry._seed(sample)
        df = prepare_board_frame(df, fe, cross_sectional_rank=True, registry=registry)
        _shutil.rmtree(reg_dir)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(REGISTRY_DIR, f"features_{board}_{ts}.parquet")
    df.to_parquet(out_path)

    n_feat = len(FeatureEngineV35.feature_columns(df)) if board == "dual" else len(
        [c for c in df.columns if c not in BruteForceGenerator.EXCLUDE_COLS
         and not c.startswith("label_")
         and df[c].dtype in ("float64", "int64", "float32")])

    logger.info(
        f"  Saved: {out_path} ({n_feat} features, "
        f"{os.path.getsize(out_path) / 1024 / 1024:.0f}MB, "
        f"{time.time() - t0:.0f}s)"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=["main", "dual"], help="Board to build")
    ap.add_argument("--data-window", default=None, choices=["1Y", "3Y", "ALL"])
    ap.add_argument("--adoption-only", action="store_true")
    ap.add_argument("--max-stocks", type=int, default=0, help="Cap stocks (0=all, use for testing)")
    args = ap.parse_args()

    panel = load_panel(args.data_window)
    logger.info(
        f"Panel loaded: {len(panel):,} rows, {panel['symbol'].nunique()} stocks"
    )

    # Step 1: Registry update
    step1_update_registry(panel)

    if args.adoption_only:
        logger.info("Adoption-only mode. Done.")
        return

    # Step 2: Board build
    boards = [args.board] if args.board else ["main", "dual"]
    for b in boards:
        # If user explicitly set --data-window, use it; otherwise use board default
        window = args.data_window if args.data_window else get_default_window(b)
        step2_build_board(panel, b, window, args.max_stocks)

    logger.info("All done.")


if __name__ == "__main__":
    main()
