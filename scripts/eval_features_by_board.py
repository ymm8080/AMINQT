"""特征评估 — 主板/双创分池 Rank IC.

用法:
  python scripts/eval_features_by_board.py                     # 全量评估
  python scripts/eval_features_by_board.py --top 50            # 只输出 top/bottom 50
  python scripts/eval_features_by_board.py --board main        # 只评估主板
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PANEL_PATHS = [
    "data/panel_full_enriched_v3.parquet",
    "data/panel_full_enriched_v3.parquet",
]
REGISTRY_DIR = "data/factor_registry"


def load_panel() -> pd.DataFrame:
    for p in PANEL_PATHS:
        if os.path.exists(p):
            logger.info("加载面板: %s", p)
            panel = pd.read_parquet(p)
            logger.info(
                "  %d stocks, %d rows, %d cols",
                panel["symbol"].nunique(),
                len(panel),
                len(panel.columns),
            )
            return panel
    raise FileNotFoundError(f"无可用面板: {PANEL_PATHS}")


def eval_board(board_name: str, board_df: pd.DataFrame, top_n: int = 50) -> dict:
    """单板特征评估: 构建特征+标签, 计算 Rank IC vs label_1d_net."""
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine
    from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

    features = FeatureEngineV35()
    cross_sectional_rank = board_name != "main"

    logger.info("[%s] 构建特征...", board_name)
    df = features.build(board_df, cross_sectional_rank=cross_sectional_rank)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)

    feature_cols = FeatureEngineV35.feature_columns(df)
    # 只保留数值列, 排除 label/mask/标识列
    feature_cols = [
        c for c in feature_cols if c in df.columns and df[c].dtype != object
    ]
    # NaN 率 > 95% 的列跳过 (无意义)
    feature_cols = [c for c in feature_cols if df[c].isna().mean() < 0.95]

    label_col = "label_1d_net"
    if label_col not in df.columns:
        label_col = "label_1d"
    logger.info("[%s] %d features vs %s", board_name, len(feature_cols), label_col)

    results = []
    for i, col in enumerate(feature_cols):
        valid = df[[col, label_col, "date"]].dropna()
        if len(valid) < 100:
            continue
        try:
            ic_series = daily_rank_ic_series(valid, col, label_col)
            ic_mean = mean_rank_ic(ic_series)
            ic_abs = mean_rank_ic(ic_series, abs_mean=True)
            ic_std = np.nanstd(ic_series.values) if len(ic_series) else np.nan
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            pos_ratio = (ic_series > 0).mean()
            nan_rate = board_df[col].isna().mean() if col in board_df.columns else 1.0
            results.append(
                {
                    "factor": col,
                    "ic_mean": round(float(ic_mean), 6),
                    "ic_abs": round(float(ic_abs), 6),
                    "ic_std": round(float(ic_std), 6),
                    "icir": round(float(icir), 4),
                    "pos_ratio": round(float(pos_ratio), 4),
                    "nan_rate": round(float(nan_rate), 4),
                    "n_dates": len(ic_series),
                }
            )
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            logger.info(
                "[%s]  %d/%d features evaluated", board_name, i + 1, len(feature_cols)
            )

    results_df = pd.DataFrame(results).sort_values("ic_abs", ascending=False)
    logger.info("[%s] 完成: %d features with valid IC", board_name, len(results_df))

    # Top / bottom
    top = results_df.head(top_n)
    bottom = results_df.tail(top_n).iloc[::-1]

    print(f"\n{'=' * 80}")
    print(f"  {board_name.upper()} — Top {top_n} features by |IC|")
    print(f"{'=' * 80}")
    print(
        f"{'Factor':<40s} {'IC':>8s} {'|IC|':>8s} {'ICIR':>8s} {'Pos%':>7s} {'NaN%':>7s}"
    )
    print("-" * 80)
    for _, r in top.iterrows():
        print(
            f"{r['factor']:<40s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>8.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}"
        )

    print(f"\n{'=' * 80}")
    print(f"  {board_name.upper()} — Bottom {top_n} features by |IC|")
    print(f"{'=' * 80}")
    print(
        f"{'Factor':<40s} {'IC':>8s} {'|IC|':>8s} {'ICIR':>8s} {'Pos%':>7s} {'NaN%':>7s}"
    )
    print("-" * 80)
    for _, r in bottom.iterrows():
        print(
            f"{r['factor']:<40s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>8.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}"
        )

    # Summary stats
    strong = (results_df["ic_abs"] >= 0.03).sum()
    weak = ((results_df["ic_abs"] >= 0.01) & (results_df["ic_abs"] < 0.03)).sum()
    noise = (results_df["ic_abs"] < 0.01).sum()
    print(
        f"\n  Summary: {strong} strong (|IC|>=0.03) | {weak} weak (0.01-0.03) | {noise} noise (<0.01)"
    )

    return {
        "board": board_name,
        "n_features": len(results_df),
        "n_strong": int(strong),
        "n_weak": int(weak),
        "n_noise": int(noise),
        "top_20": results_df.head(20)["factor"].tolist(),
        "results": results_df.to_dict(orient="records"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=50, help="Top/bottom N to display")
    parser.add_argument("--board", choices=["main", "dual"], help="Single board only")
    args = parser.parse_args()

    panel = load_panel()

    from app.pipeline1.cleaning_pipeline import CleaningPipeline

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    logger.info("清洗后: main=%d rows, dual=%d rows", len(main_df), len(dual_df))

    output = {"timestamp": datetime.now().isoformat(), "boards": {}}

    for board_name, board_df in [("main", main_df), ("dual", dual_df)]:
        if args.board and board_name != args.board:
            continue
        if len(board_df) == 0:
            logger.warning("[%s] 无样本, 跳过", board_name)
            continue
        result = eval_board(board_name, board_df, top_n=args.top)
        output["boards"][board_name] = {
            "n_features": result["n_features"],
            "n_strong": result["n_strong"],
            "n_weak": result["n_weak"],
            "n_noise": result["n_noise"],
            "top_20": result["top_20"],
        }

    # Save
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REGISTRY_DIR, f"feature_eval_{tag}.json")
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("评估结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
