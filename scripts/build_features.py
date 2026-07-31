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

import argparse
import os
import shutil
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import BruteForceGenerator, dedup_l2
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.train_runner import prepare_board_frame

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
        "symbol",
        "date",
        "board",
        "industry",
        "announce_date",
        "is_suspended",
        "is_st",
        "tradestatus",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pre_close",
        "turnover_rate",
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


def pre_screen_features(feat_cols, full_table, random_state=42):
    """Pre-screen on the FULL table (half-year rows).

    All columns go through NaN/Dedup/Mode/VarThresh, but only brute-force
    columns (containing '_brute_') are dropped. Base columns always survive.

    Uses ALL rows for NaN, Mode, and VarThresh (column-level stats).
    Only DedupL2 uses a 5K sample (for Spearman correlation matrix).

    Pipeline: NaN filter → Dedup L2 → Mode > 95% → MinMax VarThresh.
    """
    from sklearn.preprocessing import MinMaxScaler

    n_rows = full_table.num_rows
    survived = list(feat_cols)
    n_total = len(feat_cols)
    logger.info(f"  Pre-screen: {n_total} columns on {n_rows:,} rows (half-year)")

    # ── 1. NaN filter on ALL rows (> 95% NaN → drop) ──
    drop_nan = {c for c in survived if full_table.column(c).null_count / n_rows > 0.95}
    survived = [c for c in survived if c not in drop_nan]
    logger.info(
        f"  Pre-screen NaN: {n_total} → {len(survived)} "
        f"({len(drop_nan)} dropped >95% NaN)"
    )

    # ── 2. Dedup L2 — 5K sample for Spearman correlation ──
    n_sample = min(5000, n_rows)
    indices = sorted(
        np.random.RandomState(random_state).choice(n_rows, size=n_sample, replace=False)
    )
    sample_df = full_table.select(survived).take(indices).to_pandas()
    survived = dedup_l2(survived, sample_df)
    del sample_df

    # ── 3. Mode > 95% on ALL rows ──
    drop_mode = set()
    for c in survived:
        freqs = full_table.column(c).value_counts()
        if len(freqs) > 0 and freqs.column(1)[0].as_py() / n_rows > 0.95:
            drop_mode.add(c)
    survived = [c for c in survived if c not in drop_mode]
    logger.info(
        f"  Pre-screen Mode>95%: {len(drop_mode)} dropped → {len(survived)} remain"
    )

    # ── 4. MinMax normalize on ALL rows + VarThresh (threshold 0.0001) ──
    X = full_table.select(survived).to_pandas().values
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(X)
    variances = np.nanvar(normed, axis=0, ddof=0)
    del X, normed

    len(survived)
    drop_var = {
        survived[i]
        for i in range(len(survived))
        if np.isnan(variances[i]) or variances[i] < 0.0001
    }
    survived = [c for c in survived if c not in drop_var]
    logger.info(
        f"  Pre-screen VarThresh(<0.0001): {len(drop_var)} dropped → {len(survived)} remain"
    )

    logger.info(
        f"  Pre-screen TOTAL: {n_total} → {len(survived)} ({n_total - len(survived)} dropped)"
    )
    return survived


def step2_build_board(panel, board="main", window="3Y", max_stocks=0):
    """Build features for a specific board.

    MAIN: family-based batching — generates one transform family at a time
    (pct_change→rolling_mean→...), joins incrementally. Peak ~3GB for 2k stocks.
    max_stocks=0 → all MAIN stocks.
    """
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
            stocks = sorted(
                np.random.choice(
                    board_panel["symbol"].unique(), size=max_stocks, replace=False
                )
            )
            board_panel = board_panel[board_panel["symbol"].isin(stocks)]
        logger.info(f"  MAIN: {board_panel['symbol'].nunique():,} stocks")
    else:
        dual = panel[panel["board"].isin(["GEM", "STAR"])]
        board_stocks = sorted(dual["symbol"].unique())
        if len(board_stocks) > 300:
            board_stocks = sorted(
                np.random.choice(board_stocks, size=300, replace=False)
            )
        board_panel = panel[panel["symbol"].isin(board_stocks)].copy()
        logger.info(f"  DUAL: {board_panel['symbol'].nunique():,} stocks")

    # Clean + labels (shared for both boards)
    cleaner = CleaningPipeline()
    main_d, dual_d = cleaner.run_train(board_panel)
    df = (
        main_d if board == "main" else (dual_d if len(dual_d) > len(main_d) else main_d)
    )
    logger.info(f"  Cleaned: {len(df):,} rows, {df['symbol'].nunique()} stocks")

    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    # Must mask >= 5 days for label_5d horizon to avoid look-ahead bias
    # (train_runner.py uses MASK_RECENT_DAYS=6, all eval scripts use 6)
    df = LabelEngine.mask_recent_days(df, days=6)

    if board == "main":
        # ── Two-phase build ──
        # Phase 1: 1Y data → generate 3,680 features → 4-step IC screen → ~635 selected
        # Phase 2: Full 3Y data → generate only selected features → final parquet
        import tempfile
        import pyarrow.parquet as pq
        import pyarrow as pa
        import re
        import gc as _gc

        FAMILIES = [
            "pct_change",
            "rolling_mean",
            "rolling_std",
            "rolling_max",
            "diff",
            "momentum",
            "EMA",
        ]
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        logger.info(
            f"  BruteForce: {len(raw_cols)} eligible raw cols x {len(FAMILIES)} families"
        )

        # ══════════════════════════════════════════════════
        # Phase 1: 1Y expansion + IC screen (no in-memory merge)
        # ══════════════════════════════════════════════════
        dates_18m = sorted(df["date"].unique())[-378:]  # ~1.5 years trading days
        df_p1 = df[df["date"].isin(dates_18m)]
        logger.info(f"  Phase 1 (1.5Y): {len(df_p1):,} rows, {len(dates_18m)} dates")

        tmp1 = tempfile.mkdtemp(prefix="brute_p1_", dir=REGISTRY_DIR)
        try:
            # Save base + generate each family → separate parquet files
            bp = os.path.join(tmp1, "base.parquet")
            df_p1.to_parquet(bp, index=False)
            base_names = set(pq.read_schema(bp).names)

            family_files = {}  # fam → path
            all_bf_cols = []  # track all brute-force column names
            for fam in FAMILIES:
                new = gen.generate_family(
                    df_p1, fam, raw_cols=raw_cols, dtype="float32"
                )
                fp = os.path.join(tmp1, f"{fam}.parquet")
                bf_only = [
                    c for c in new.columns if "_brute_" in c and c not in base_names
                ]
                if bf_only:
                    new[bf_only].to_parquet(fp, index=False)
                    family_files[fam] = fp
                    all_bf_cols.extend(bf_only)
                del new
                _gc.collect()

            total_cols = len(base_names) + len(all_bf_cols)
            logger.info(
                f"  Phase 1: {len(base_names)} base + {len(all_bf_cols)} BF = {total_cols} cols"
            )

            # ── Structured sample for DedupL2 ──
            # Pick ~12 trading days (one per month), Tue/Wed/Thu only,
            # 2nd/3rd week, market return [-1%, +1%], away from holidays.
            from pandas.tseries.holiday import (
                AbstractHolidayCalendar,
                nearest_workday,
                Holiday,
            )

            # Simple Chinese holiday calendar (Spring Festival, National Day, etc.)
            class _CNHoliday(AbstractHolidayCalendar):
                rules = [
                    Holiday("New Year", month=1, day=1),
                    Holiday(
                        "Spring Festival Eve",
                        month=1,
                        day=28,
                        offset=pd.DateOffset(days=-2),
                    ),
                    Holiday("Spring Festival", month=1, day=28),
                    Holiday("Qingming", month=4, day=5),
                    Holiday("Labor Day", month=5, day=1),
                    Holiday("Dragon Boat", month=6, day=10),
                    Holiday("Mid-Autumn", month=9, day=25),
                    Holiday(
                        "National Day", month=10, day=1, observance=nearest_workday
                    ),
                    Holiday("National Day 2", month=10, day=2),
                    Holiday("National Day 3", month=10, day=3),
                ]

            hol_cal = _CNHoliday()
            holidays = set(
                d.date()
                for d in hol_cal.holidays(start=dates_18m[0], end=dates_18m[-1])
            )

            # Compute market index return approximation
            # Use close_hfq if available in base columns
            has_price = "close_hfq" in base_names and "open_hfq" in base_names
            sampled_dates = []
            for y in sorted(set(d.year for d in dates_18m)):
                for m in range(1, 13):
                    month_dates = [d for d in dates_18m if d.year == y and d.month == m]
                    if len(month_dates) < 8:
                        continue
                    # 2nd or 3rd week (day 8-21)
                    month_start = month_dates[0]
                    week2_start = month_start + pd.Timedelta(days=7)
                    week3_end = month_start + pd.Timedelta(days=20)
                    candidates = [
                        d for d in month_dates if d >= week2_start and d <= week3_end
                    ]
                    # Tue/Wed/Thu only (weekday 1/2/3)
                    candidates = [d for d in candidates if d.weekday() in (1, 2, 3)]
                    # Away from holidays (>= 5 trading days)
                    candidates = [
                        d
                        for d in candidates
                        if all(
                            abs((d.date() - h).days) > 5
                            for h in holidays
                            if h is not None
                        )
                    ]
                    # Market return check (use panel data)
                    for d in candidates:
                        day_mask = df_p1["date"] == d
                        if has_price and day_mask.any():
                            day_data = df_p1[day_mask]
                            opens = day_data["open_hfq"].values
                            closes = day_data["close_hfq"].values
                            # Equal-weight market return
                            mkt_ret = np.nanmean(
                                (closes - opens) / (np.abs(opens) + 1e-8)
                            )
                            if -0.01 <= mkt_ret <= 0.01:
                                sampled_dates.append(d)
                                break  # one day per month
                        else:
                            sampled_dates.append(d)
                            break
                    if len(sampled_dates) >= 3:
                        break
                if len(sampled_dates) >= 3:
                    break
            # Fallback: if not enough dates found, fill with simple Tue/Wed/Thu
            if len(sampled_dates) < 2:
                sampled_dates = [d for d in dates_18m if d.weekday() in (1, 2, 3)][:3]

            logger.info(f"  Dedup sample: {len(sampled_dates)} dates selected")
            sample_mask = df_p1["date"].isin(sampled_dates)
            # Use boolean mask directly on parquet read
            sample_row_mask = sample_mask.values  # numpy bool array

            # Read sample from each file using boolean mask
            sample_parts = [pd.read_parquet(bp)[sample_row_mask]]
            for fam, fp in family_files.items():
                sample_parts.append(pd.read_parquet(fp)[sample_row_mask])
            sample_df = pd.concat(sample_parts, axis=1)
            del sample_parts
            _gc.collect()
            logger.info(
                f"  Dedup sample: {len(sample_df):,} rows from {len(sampled_dates)} dates"
            )

            # Run pre-screen on the sample (NaN/Mode on ALL rows, Dedup/Var on sample)
            # For NaN: check ALL rows from each file
            # For Mode: check ALL rows from each file
            # For Dedup: use sample
            # For VarThresh: use ALL rows per column (can compute min/max/var without full load)
            all_cols = list(base_names) + all_bf_cols

            # 1. Dedup L2 on sample
            survived = list(all_cols)
            sample_survived = [c for c in survived if c in sample_df.columns]
            survived = dedup_l2(sample_survived, sample_df[sample_survived])

            # 2. Mode > 95% + VarThresh < 0.001 — single pass per column
            drop_mode = set()
            drop_var = set()
            for c in survived:
                if c in base_names:
                    col = pq.read_table(bp, columns=[c]).column(c)
                else:
                    col = None
                    for fam, fp in family_files.items():
                        if c in pq.read_schema(fp).names:
                            col = pq.read_table(fp, columns=[c]).column(c)
                            break
                if col is None:
                    continue
                # Skip non-numeric
                if (
                    pa.types.is_string(col.type)
                    or pa.types.is_timestamp(col.type)
                    or pa.types.is_boolean(col.type)
                ):
                    continue
                # (A) Mode check
                freqs = col.value_counts()
                if len(freqs) > 0:
                    top = (
                        freqs.column(1)[0].as_py()
                        if hasattr(freqs, "column")
                        else freqs.field(1)[0].as_py()
                    )
                    if top / col.length() > 0.95:
                        drop_mode.add(c)
                        continue  # already dropped, skip VarThresh
                # (B) VarThresh
                try:
                    vals = col.to_pandas().values.astype(np.float64)
                except (ValueError, TypeError):
                    continue
                valid = vals[~np.isnan(vals)]
                if len(valid) < 10:
                    drop_var.add(c)
                    continue
                mn, mx = valid.min(), valid.max()
                if mx == mn:
                    drop_var.add(c)
                    continue
                normed = (valid - mn) / (mx - mn)
                v = np.var(normed, ddof=0)
                if np.isnan(v) or v < 0.0022:
                    drop_var.add(c)
            survived = [c for c in survived if c not in drop_mode and c not in drop_var]
            logger.info(
                f"  Phase 1 Step2: Mode={len(drop_mode)} + VarThresh<0.0022={len(drop_var)} dropped → {len(survived)} remain"
            )

            bf_survived = [c for c in survived if "_brute_" in c]
            base_survived = [c for c in survived if "_brute_" not in c]
            logger.info(
                f"  Phase 1 screen: {total_cols} → {len(survived)} "
                f"({len(base_survived)} base + {len(bf_survived)} BF)"
            )
        finally:
            shutil.rmtree(tmp1, ignore_errors=True)

        # Parse brute-force specs: 'bias_10_brute_pct40' → raw='bias_10', fam='pct_change', w=40
        FAM_ABBR_MAP = {
            "pct": "pct_change",
            "d": "diff",
            "max": "rolling_max",
            "min": "rolling_min",
            "std": "rolling_std",
            "ma": "rolling_mean",
            "ema": "EMA",
            "mom": "momentum",
        }
        selected_specs = []
        for c in bf_survived:
            parts = c.split("_brute_")
            m = re.match(r"(pct|d|max|min|std|ma|ema|mom)(\d+)", parts[1])
            if m:
                selected_specs.append(
                    {
                        "raw": parts[0],
                        "family": FAM_ABBR_MAP[m.group(1)],
                        "window": int(m.group(2)),
                    }
                )
        logger.info(
            f"  Phase 1 DONE: {len(selected_specs)} brute-force + {len(base_survived)} base selected"
        )

        # ══════════════════════════════════════════════════
        # Phase 2: 1.5Y — generate ONLY selected features
        # (Use 1.5Y to avoid OOM from generate_family on full 3Y)
        # ══════════════════════════════════════════════════
        specs_by_fam = {}
        for s in selected_specs:
            specs_by_fam.setdefault(s["family"], []).append(s)

        tmp2 = tempfile.mkdtemp(prefix="brute_p2_", dir=REGISTRY_DIR)
        try:
            bp2 = os.path.join(tmp2, "base.parquet")
            df_p1.to_parquet(bp2, index=False)  # use 1.5Y data, not full 3Y
            base_table = pq.read_table(bp2)

            n_gen = 0
            fam_abbr_rev = {v: k for k, v in FAM_ABBR_MAP.items()}
            for fam in FAMILIES:
                if fam not in specs_by_fam:
                    continue
                specs = specs_by_fam[fam]
                new = gen.generate_family(
                    df_p1, fam, raw_cols=raw_cols, dtype="float32"
                )
                abbr = fam_abbr_rev.get(fam, fam[:3])
                needed = {f"{s['raw']}_brute_{abbr}{s['window']}" for s in specs}
                keep = [c for c in new.columns if c in needed]
                if keep:
                    # Save to parquet first (ensures type consistency), then read back
                    tmp_fp = os.path.join(tmp2, f"p2_{fam}.parquet")
                    new[keep].to_parquet(tmp_fp, index=False)
                    ft = pq.read_table(tmp_fp)
                    base_table = pa.concat_tables(
                        [base_table, ft], promote_options="permissive"
                    )
                    n_gen += len(keep)
                    del ft
                del new
                _gc.collect()
                # Flush to disk
                acc = os.path.join(tmp2, "accumulated.parquet")
                pq.write_table(base_table, acc)
                del base_table
                _gc.collect()
                base_table = pq.read_table(acc)

            logger.info(f"  Phase 2: {n_gen} features generated on 1.5Y data")

            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            out_path = os.path.join(REGISTRY_DIR, f"features_{board}_{ts}.parquet")
            pq.write_table(base_table, out_path)
            del base_table
            _gc.collect()
            logger.info(
                f"  Saved: {out_path} ({n_gen} features, "
                f"{os.path.getsize(out_path) / 1024 / 1024:.0f}MB, "
                f"{time.time() - t0:.0f}s)"
            )
            return out_path
        finally:
            try:
                del base_table
            except Exception:
                pass
            _gc.collect()
            shutil.rmtree(tmp2, ignore_errors=True)
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
        shutil.rmtree(reg_dir)

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = os.path.join(REGISTRY_DIR, f"features_{board}_{ts}.parquet")
    df.to_parquet(out_path)

    n_feat = (
        len(FeatureEngineV35.feature_columns(df))
        if board == "dual"
        else len(
            [
                c
                for c in df.columns
                if c not in BruteForceGenerator.EXCLUDE_COLS
                and not c.startswith("label_")
                and df[c].dtype in ("float64", "int64", "float32")
            ]
        )
    )

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
    ap.add_argument(
        "--max-stocks", type=int, default=0, help="Cap stocks (0=all, use for testing)"
    )
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
