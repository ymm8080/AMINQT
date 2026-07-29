# -*- coding: utf-8 -*-
"""Save new bias/derived feature IC results to factor registry.

Uses the full ICScreener.screen() for proper grading with all thresholds.
"""

import sys
import json
import gc

sys.path.insert(0, ".")

import pandas as pd
from datetime import datetime
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.ic_screener import ICScreener

tag = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load panel (last 200 dates)
panel = pd.read_parquet("data/panel_full_enriched_v3.parquet")
all_dates = sorted(panel["date"].unique())
panel = panel[panel["date"].isin(all_dates[-200:])]
print(f"Panel: {len(panel)} rows, {panel.date.nunique()} dates")

cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel)
del panel
gc.collect()

for board, board_df in [("main", main_df), ("dual", dual_df)]:
    if len(board_df) == 0:
        continue
    print(f"\n=== {board}: {len(board_df)} rows ===")

    eng = FeatureEngineV35()
    df = eng.build(board_df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)

    candidates = FeatureEngineV35.feature_columns(df)
    valid = [
        c
        for c in candidates
        if c in df.columns and df[c].isna().mean() < 0.95 and df[c].dtype != object
    ]
    print(f"  {len(valid)} feature candidates")

    # Screen all features (full ICScreener pipeline with proper grading)
    screener = ICScreener(registry_path="data/factor_registry")
    window_id = f"dual_{board}_{tag}_new_features"
    result = screener.screen(df, valid, window_id=window_id)

    # Extract only new feature results
    new_features = [
        "bias_5",
        "bias_10",
        "bias_20",
        "bias_60",
        "bias_120",
        "bias_250",
        "bias_5_20_cross",
        "bias_20_60_cross",
        "ma_vol_ratio_5_20",
        "amplitude_5d",
    ]
    print(f"\n  {board.upper()} — New Feature IC Results:")
    print(
        f"  {'Feature':25s} {'IC_1d':>8s} {'IC_3d':>8s} {'IC_5d':>8s} {'AUC':>6s} {'Grade':>8s} {'RollPos':>7s}"
    )
    print(f"  {'-' * 70}")

    passed, failed = [], []
    for f in new_features:
        d = result["detail"].get(f)
        if d is None:
            print(f"  {f:25s}  NOT EVALUATED")
            continue
        ic_1d = d.get("ic_1d", 0)
        ic_3d = d.get("ic_3d", 0)
        ic_5d = d.get("ic_5d", 0)
        auc = d.get("auc", 0)
        grade = d.get("grade", "?")
        roll_pos = d.get("rolling_pos_ratio", 0)
        print(
            f"  {f:25s} {ic_1d:>+8.4f} {ic_3d:>+8.4f} {ic_5d:>+8.4f} {auc:>6.4f} {grade:>8s} {roll_pos:>7.1%}"
        )
        if grade in ("strong", "weak"):
            passed.append(f)
        else:
            failed.append(f)

    print(f"\n  PASS ({len(passed)}): {passed}")
    print(f"  FAIL ({len(failed)}): {failed}")

    # Save focused registry entry
    new_only = {
        "window_id": f"new_features_{board}_{tag}",
        "factors": passed,
        "detail": {
            f: result["detail"][f] for f in new_features if f in result["detail"]
        },
    }
    path = f"data/factor_registry/factors_new_{board}_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(new_only, f, ensure_ascii=False, indent=1)
    print(f"  Saved: {path}")

    del df
    gc.collect()

del main_df, dual_df
gc.collect()
print("\nDONE")
