"""Quick IC eval for new bias/derived features — works directly on panel data."""

import json
import os
import sys

sys.path.insert(0, ".")

from datetime import datetime

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

tag = datetime.now().strftime("%Y%m%d_%H%M%S")
NEW_FEATURES = [
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

# Load panel, restrict to recent dates
panel = pd.read_parquet("data/panel_full_enriched_v3.parquet")
all_dates = sorted(panel["date"].unique())
panel = panel[panel["date"].isin(all_dates[-200:])]
print(f"Panel: {len(panel)} rows, {panel.date.nunique()} dates, {panel.shape[1]} cols")

# Clean + Feature Engine
cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel)

for board, board_df in [("main", main_df), ("dual", dual_df)]:
    print(f"\n{'=' * 60}")
    print(f"  {board.upper()}")
    print(f"{'=' * 60}")
    if len(board_df) == 0:
        continue

    eng = FeatureEngineV35()
    df = eng.build(board_df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)

    label_col = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    print(f"\nNew Feature IC vs {label_col}:")
    print(
        f"{'Feature':25s} {'IC_1d':>8s} {'IC_3d':>8s} {'IC_5d':>8s} {'ICIR':>6s} {'Pos%':>6s} {'NaN%':>6s}"
    )
    print("-" * 75)

    results = []
    for col in NEW_FEATURES:
        if col not in df.columns:
            print(f"  {col:25s}  NOT IN DF")
            continue
        nan_rate = df[col].isna().mean()

        ic_vals = {}
        for _k, lbl_sfx in [(1, "1d"), (3, "3d"), (5, "5d")]:
            lbl = (
                f"label_{lbl_sfx}_net"
                if f"label_{lbl_sfx}_net" in df.columns
                else f"label_{lbl_sfx}"
            )
            sub = df[[col, lbl, "date"]].dropna()
            if len(sub) > 50:
                ic = mean_rank_ic(sub, col, lbl, abs_mean=False)
                ic_vals[f"ic_{lbl_sfx}"] = round(float(ic), 4)
            else:
                ic_vals[f"ic_{lbl_sfx}"] = 0.0

        # ICIR and pos_ratio from 1d
        sub1 = df[[col, label_col, "date"]].dropna()
        if len(sub1) > 50:
            ic_series = daily_rank_ic_series(sub1, col, label_col)
            pos_ratio = (
                round(float((ic_series > 0).mean()), 4) if len(ic_series) > 5 else 0.0
            )
            ic_std = float(ic_series.std())
            icir = round(abs(ic_vals["ic_1d"]) / ic_std if ic_std > 0 else 0, 4)
        else:
            pos_ratio, icir = 0.0, 0.0

        results.append(
            {
                **ic_vals,
                "pos_ratio": pos_ratio,
                "icir": icir,
                "nan_rate": round(float(nan_rate), 4),
            }
        )
        print(
            f"{col:25s} {ic_vals['ic_1d']:>+8.4f} {ic_vals['ic_3d']:>+8.4f} {ic_vals['ic_5d']:>+8.4f} {icir:>6.4f} {pos_ratio:>6.1%} {nan_rate:>6.1%}"
        )

    # Save
    out = {
        "window_id": f"new_features_{board}_{tag}",
        "factors": [c for c in NEW_FEATURES if c in df.columns],
        "detail": dict(zip([c for c in NEW_FEATURES if c in df.columns], results)),
    }
    os.makedirs("data/factor_registry", exist_ok=True)
    path = f"data/factor_registry/factors_new_{board}_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Saved: {path}")

print("\nDONE")
