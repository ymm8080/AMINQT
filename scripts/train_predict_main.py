#!/usr/bin/env python
"""MAIN board train + predict using 3-layer pipeline (robust, resumable).

Reads Layer1 feature parquet (column-filtered) + Layer2 selected features
→ LightGBM → prediction. No cross_sectional_rank for MAIN.

Usage:
  python scripts/train_predict_main.py                           # Full run (state-tracked)
  python scripts/train_predict_main.py --resume                   # Resume from last state
  python scripts/train_predict_main.py --tag 2026W31
  python scripts/train_predict_main.py --train-only
  python scripts/train_predict_main.py --predict-only
  python scripts/train_predict_main.py --max-stocks 100           # Test subset
  python scripts/train_predict_main.py --force train              # Re-run training even if done
  python scripts/train_predict_main.py --status                   # Show pipeline state

Resilience:
  - PipelineState tracks step completion across Claude sessions
  - TrainingCheckpoint saves after each model kind (crash → resume at last kind)
  - Atomic file writes (temp + rename) prevent corruption
  - Memory: gc.collect() between stages, column-filtered parquet loading
"""

import argparse
import gc
import glob
import json
import os
import signal
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.predict_runner import find_bundles, run_prediction
from app.pipeline1.checkpoint import PipelineState

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("train_predict_main")

REGISTRY_DIR = Path("data/factor_registry")
MODEL_DIR = Path("models/pipeline1")
PANEL_PATH = Path("data/panel_full_enriched_v3.parquet")

# ── Interrupt flag for graceful shutdown ──
_interrupted = False


def _on_interrupt(signum, frame):
    global _interrupted
    _interrupted = True
    logger.warning("收到中断信号 (Ctrl+C), 正在安全退出... (再次 Ctrl+C 强制退出)")


signal.signal(signal.SIGINT, _on_interrupt)
signal.signal(signal.SIGTERM, _on_interrupt)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _log_memory(tag: str = "") -> None:
    """Log current RSS memory usage."""
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info("  MEM [%s]: %.0f MB", tag, rss_mb)
    except ImportError:
        pass  # psutil not installed — skip memory logging


def find_latest_selection(board: str) -> str:
    """Find the latest Layer2 selected_features.json for a board."""
    sel_files = sorted(
        glob.glob(str(REGISTRY_DIR / f"selected_{board}_20*.json")),
        reverse=True,
    )
    if not sel_files:
        raise FileNotFoundError(
            f"No selected_{board}_*.json found. Run: python scripts/select_features.py --board {board} --update"
        )
    return sel_files[0]


def find_latest_features(board: str) -> str:
    """Find the latest Layer1 feature parquet for a board."""
    feat_files = sorted(
        glob.glob(str(REGISTRY_DIR / f"features_{board}_*.parquet")),
        reverse=True,
    )
    if not feat_files:
        raise FileNotFoundError(
            f"No features_{board}_*.parquet found. Run: python scripts/build_features.py --board {board}"
        )
    return feat_files[0]


def load_features_for_training(
    board: str, features_path: str, selected_features: set, max_stocks: int = 0
) -> pd.DataFrame:
    """Load only selected feature columns + id/label cols from Layer1 parquet.

    Avoids loading all 3,300 cols for MAIN — only reads selected + overhead.
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
    missing = selected_features - set(read_cols)
    if missing:
        logger.warning(
            "  %d selected features missing from parquet, will be skipped", len(missing)
        )

    file_mb = os.path.getsize(features_path) / 1024 / 1024
    logger.info(
        "  Loading %d cols from %s (%.0fMB)",
        len(read_cols),
        os.path.basename(features_path),
        file_mb,
    )
    df = pd.read_parquet(features_path, columns=read_cols)

    # ── 内存: 下转特征列为 float32 (混合 dtype 会 upcast 到 float64) ──
    feat_cols_present = [c for c in selected_features if c in df.columns]
    if feat_cols_present:
        df[feat_cols_present] = df[feat_cols_present].astype("float32", copy=False)
    gc.collect()

    if max_stocks and max_stocks > 0 and df["symbol"].nunique() > max_stocks:
        stocks = sorted(
            np.random.choice(df["symbol"].unique(), size=max_stocks, replace=False)
        )
        df = df[df["symbol"].isin(stocks)]

    logger.info(
        "  Training data: %s rows, %s stocks, %d cols",
        f"{len(df):,}",
        df["symbol"].nunique(),
        len(read_cols),
    )
    return df


# ──────────────────────────────────────────────
# Step: Build Features (Layer 1)
# ──────────────────────────────────────────────


def step_build_features(board: str, state: PipelineState) -> str | None:
    """Run Layer1 feature build. Returns path to features parquet."""
    step = "build_features"
    if not state.step_should_run(step):
        output = state._state["steps"][step].get("output", "")
        if output and os.path.exists(output):
            logger.info(
                "[%s] 已完成 (state), 跳过. Output: %s", step, os.path.basename(output)
            )
            return output
        else:
            logger.warning("[%s] state 标记 done 但 output 缺失, 重新运行", step)

    state.mark_running(step)
    logger.info("=" * 60)
    logger.info("  Layer 1: Build Features (%s)", board)
    logger.info("=" * 60)

    # Run build_features as subprocess to isolate memory
    import subprocess

    cmd = [sys.executable, "scripts/build_features.py", "--board", board]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        state.mark_failed(
            step, f"build_features.py exited with code {result.returncode}"
        )
        raise RuntimeError(f"build_features.py failed (exit {result.returncode})")

    feat_path = find_latest_features(board)
    state.mark_done(step, output=feat_path)
    _log_memory("after build_features")
    gc.collect()
    return feat_path


# ──────────────────────────────────────────────
# Step: Select Features (Layer 2)
# ──────────────────────────────────────────────


def step_select_features(board: str, state: PipelineState) -> str | None:
    """Ensure Layer2 feature selection exists. Returns path to selected JSON."""
    step = "select_features"
    if not state.step_should_run(step):
        output = state._state["steps"][step].get("output", "")
        if output and os.path.exists(output):
            logger.info(
                "[%s] 已完成 (state), 跳过. Output: %s", step, os.path.basename(output)
            )
            return output

    # Check if selection already exists (may have been run manually)
    try:
        existing = find_latest_selection(board)
        # If selection is from today, consider it done
        sel_mtime = os.path.getmtime(existing)
        if time.time() - sel_mtime < 86400:  # < 24h old
            logger.info(
                "[%s] 发现最近的 selection (%s), 跳过", step, os.path.basename(existing)
            )
            state.mark_done(step, output=existing)
            return existing
    except FileNotFoundError:
        pass  # Need to run

    state.mark_running(step)
    logger.info("=" * 60)
    logger.info("  Layer 2: Select Features (%s)", board)
    logger.info("=" * 60)

    import subprocess

    cmd = [
        sys.executable,
        "scripts/select_features.py",
        "--board",
        board,
        "--update",
        "--yes",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        state.mark_failed(
            step, f"select_features.py exited with code {result.returncode}"
        )
        raise RuntimeError(f"select_features.py failed (exit {result.returncode})")

    sel_path = find_latest_selection(board)
    state.mark_done(step, output=sel_path)
    gc.collect()
    return sel_path


# ──────────────────────────────────────────────
# Step: Train
# ──────────────────────────────────────────────


def step_train(
    board: str,
    features_path: str,
    sel_path: str,
    tag: str,
    max_stocks: int,
    state: PipelineState,
    resume: bool,
) -> dict:
    """Train LightGBM on MAIN board using pre-built features. Supports checkpoint resume."""
    step = "train"

    with open(sel_path) as f:
        sel = json.load(f)
    feature_cols = sel["features"]
    logger.info(
        "MAIN: %d features (%s), pool=%d",
        len(feature_cols),
        sel.get("pipeline", "?"),
        sel.get("pool_size", 0),
    )

    # Check if model already exists for this tag
    model_path = MODEL_DIR / f"main_{tag}.pkl"
    if model_path.exists() and not resume:
        logger.info(
            "[%s] 模型已存在: %s. 跳过训练 (用 --resume 或 --force train 覆盖)",
            step,
            model_path,
        )
        state.mark_done(
            step, output=str(model_path), meta={"features": len(feature_cols)}
        )
        return {"main": {"path": str(model_path), "oos": {"ics": {}}, "switched": True}}

    state.mark_running(step)
    logger.info("=" * 60)
    logger.info("  Train MAIN (tag=%s)", tag)
    logger.info("=" * 60)

    df = load_features_for_training(board, features_path, set(feature_cols), max_stocks)
    # Drop rows with NaN in ALL feature columns to avoid training failures
    df = df.dropna(subset=feature_cols, how="all")

    trainer = DualTrackTrainer(model_dir=str(MODEL_DIR))
    t0 = time.time()
    results = trainer.weekly_retrain(
        {"main": df}, {"main": feature_cols}, tag, resume=resume
    )
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

    state.mark_done(
        step,
        output=results.get("main", {}).get("path", ""),
        meta={
            "features": len(feature_cols),
            "oos_ic_1d": results.get("main", {})
            .get("oos", {})
            .get("ics", {})
            .get("1d_reg", 0),
        },
    )
    _log_memory("after train")
    gc.collect()
    return results


# ──────────────────────────────────────────────
# Step: Predict
# ──────────────────────────────────────────────


def step_predict(
    board: str,
    trade_date: str,
    max_stocks: int,
    state: PipelineState,
    low_memory: bool = False,
) -> pd.DataFrame | None:
    """Generate MAIN predictions using the latest MAIN model.

    Memory optimizations:
      - PyArrow predicate pushdown: only reads MAIN board rows (not full 2.7GB)
      - low_memory=True: only loads last 180 trading days (sufficient for feature lookback)
    """
    step = "predict"
    list_path = Path(f"data/lists/list_{trade_date}.parquet")

    if not state.step_should_run(step):
        if list_path.exists():
            df = pd.read_parquet(list_path)
            logger.info("[%s] 清单已存在: %s (%d 候选)", step, list_path, len(df))
            return df
        else:
            logger.warning("[%s] state 标记 done 但清单文件缺失, 重新运行", step)

    state.mark_running(step)
    logger.info("=" * 60)
    logger.info("  Predict MAIN (%s)", trade_date)
    logger.info("=" * 60)

    bundles = find_bundles(str(MODEL_DIR))
    if "main" not in bundles:
        logger.error("No main model found in %s", MODEL_DIR)
        state.mark_failed(step, "No main model found")
        return None
    logger.info("Model: main -> %s", os.path.basename(bundles["main"]))

    # ── 内存优化: PyArrow predicate pushdown (只读主板, 非 GEM/STAR) ──
    import pyarrow.parquet as pq

    if low_memory:
        import pyarrow.compute as pc

        # Only load last ~180 trading days for feature lookback
        date_col = pd.to_datetime(
            pq.read_table(PANEL_PATH, columns=["date"]).column("date").to_pandas()
        )
        all_dates = sorted(date_col.unique())
        if len(all_dates) > 180:
            cutoff = all_dates[-180]
            logger.info(
                "  low_memory: date filter >= %s (%d/%d dates)",
                str(cutoff)[:10],
                180,
                len(all_dates),
            )
            date_mask = pc.field("date") >= pd.Timestamp(cutoff)
            board_mask = pc.invert(
                pc.or_(
                    pc.equal(pc.field("board"), "GEM"),
                    pc.equal(pc.field("board"), "STAR"),
                )
            )
            table = pq.read_table(
                PANEL_PATH,
                filters=pq.filters.Filter.mask(pc.and_(date_mask, board_mask)),
            )
        else:
            table = pq.read_table(
                PANEL_PATH, filters=[("board", "not in", ["GEM", "STAR"])]
            )
    else:
        table = pq.read_table(
            PANEL_PATH, filters=[("board", "not in", ["GEM", "STAR"])]
        )

    main_panel = table.to_pandas()
    del table
    gc.collect()

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

    result = run_prediction(
        main_panel,
        trade_date,
        bundles,
        list_dir="data/lists",
        market_state="range",
        supply=None,
    )
    lst = result.get("list")

    if lst is not None and len(lst):
        state.mark_done(step, output=str(list_path), meta={"candidates": len(lst)})
    else:
        state.mark_done(step, output="(empty — safety valve)", meta={"candidates": 0})

    _log_memory("after predict")
    del main_panel
    gc.collect()
    return lst


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Train + Predict MAIN board (robust, resumable)"
    )
    ap.add_argument("--tag", default=None, help="Model tag (default: ISO week)")
    ap.add_argument("--train-only", action="store_true", help="Skip prediction")
    ap.add_argument("--predict-only", action="store_true", help="Skip training")
    ap.add_argument("--max-stocks", type=int, default=0, help="Cap stocks (0=all)")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last pipeline state + training checkpoint",
    )
    ap.add_argument(
        "--force",
        nargs="*",
        default=None,
        help="Force re-run specific steps even if marked done (build_features, select_features, train, predict)",
    )
    ap.add_argument(
        "--status", action="store_true", help="Show pipeline state and exit"
    )
    ap.add_argument(
        "--reset", action="store_true", help="Clear pipeline state and start fresh"
    )
    ap.add_argument(
        "--low-memory",
        action="store_true",
        help="Aggressive memory saving: date-filtered panel, extra gc",
    )
    args = ap.parse_args()

    board = "main"
    trade_date = datetime.now().strftime("%Y%m%d")
    tag = args.tag
    if tag is None:
        iso = datetime.now().isocalendar()
        tag = f"{iso[0]}W{iso[1]:02d}"

    # ── State management ──
    state = PipelineState("train_predict", tag=tag, board=board)

    if args.reset:
        state.clear_all()
        logger.info("Pipeline state cleared. Starting fresh.")
        if not args.status:
            pass  # continue to run

    if args.status:
        print(f"\n{state.summary()}")
        # Also check training checkpoint
        from app.pipeline1.checkpoint import TrainingCheckpoint

        ck = TrainingCheckpoint(str(MODEL_DIR), board, tag)
        if ck.exists():
            print("\nTraining checkpoint exists:")
            print(f"  Completed kinds: {ck.completed_kinds}")
            print(f"  Completed extras: {ck.completed_extras}")
            print(f"  Remaining kinds: {ck.remaining_kinds()}")
            print(f"  Remaining extras: {ck.remaining_extras()}")
        else:
            print("\nNo training checkpoint (clean state)")
        return

    # ── Force re-run ──
    force_steps = set(args.force or [])
    for s in force_steps:
        state.reset_step(s)
        logger.info("Force reset step: %s", s)

    do_all = not (args.train_only or args.predict_only)

    logger.info(
        "MAIN pipeline: tag=%s trade_date=%s max_stocks=%s resume=%s",
        tag,
        trade_date,
        args.max_stocks or "all",
        args.resume,
    )

    try:
        # ── Layer 1: Build ──
        if args.train_only or do_all:
            feat_path = step_build_features(board, state)
            if _interrupted:
                logger.warning("中断: build_features 后退出 (state 已保存)")
                return
        else:
            feat_path = find_latest_features(board)
            logger.info("Using existing features: %s", os.path.basename(feat_path))

        # ── Layer 2: Select ──
        if args.train_only or do_all:
            sel_path = step_select_features(board, state)
            if _interrupted:
                logger.warning("中断: select_features 后退出 (state 已保存)")
                return
        else:
            sel_path = find_latest_selection(board)
            logger.info("Using existing selection: %s", os.path.basename(sel_path))

        # ── Train ──
        if args.train_only or do_all:
            step_train(
                board,
                feat_path,
                sel_path,
                tag,
                args.max_stocks,
                state,
                resume=args.resume,
            )
            if _interrupted:
                logger.warning("中断: train 后退出 (state + checkpoint 已保存)")
                logger.warning(
                    "下次运行: python scripts/train_predict_main.py --resume"
                )
                return

        # ── Predict ──
        if args.predict_only or do_all:
            lst = step_predict(
                board, trade_date, args.max_stocks, state, low_memory=args.low_memory
            )
            if lst is not None and len(lst):
                cols = ["symbol", "board", "pred_ret_1d", "prob_up", "score"]
                available = [c for c in cols if c in lst.columns]
                print(lst[available].head(20).to_string(index=False))
                print(f"\n  Total: {len(lst)} candidates")
            else:
                print("  No MAIN candidates (safety valve or empty)")

    except KeyboardInterrupt:
        logger.warning(
            "用户中断 (KeyboardInterrupt). State 已保存, 下次用 --resume 继续."
        )
        print(f"\n{state.summary()}")
        sys.exit(130)
    except Exception as e:
        logger.error("Pipeline 失败: %s", e, exc_info=True)
        print(f"\n{state.summary()}")
        sys.exit(1)

    print(f"\n{state.summary()}")
    print(f"\nDone: {trade_date} tag={tag}")


if __name__ == "__main__":
    main()
