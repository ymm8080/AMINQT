# -*- coding: utf-8 -*-
"""IC 评估与特征筛选 — 单板块 (main 或 dual).

用法: python scripts/ic_eval_board.py main
      python scripts/ic_eval_board.py dual

全量历史数据 IC 评估 — 不做流动性/成交额过滤.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.cleaning_pipeline import board_of

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ic_eval")

PANEL_PATH = "data/panel_full_enriched_v3.parquet"
REGISTRY_DIR = "data/factor_registry"
MASK_RECENT_DAYS = 6  # 近端掩码天数


def main():
    board = sys.argv[1] if len(sys.argv) > 1 else "main"
    if board not in ("main", "dual"):
        logger.error("用法: python scripts/ic_eval_board.py [main|dual]")
        sys.exit(1)

    t0 = time.time()
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    window_id = f"{board}_{tag}"

    logger.info("=== IC 评估 [%s] 开始 ===", board.upper())

    # ---- Load panel ----
    panel = pd.read_parquet(PANEL_PATH)
    logger.info("面板: %d rows, %d cols", len(panel), len(panel.columns))

    # ---- 基础清洗 (成交额+停牌过滤) ----
    MIN_AMOUNT = 200_000  # 日成交额 >= 20万 (元), 剔除僵尸股/仙股

    if "board" not in panel.columns:
        panel = panel.copy()
        panel["board"] = panel["symbol"].map(board_of)

    clean = panel[~panel["is_suspended"].astype(bool)]
    key_cols = ["open", "high", "low", "close", "close_hfq", "volume", "amount"]
    clean = clean.dropna(subset=key_cols)

    # 成交额底线过滤
    n_before = len(clean)
    clean = clean[clean["amount"] >= MIN_AMOUNT]
    logger.info(
        "[%s] 成交额过滤 (>=%.0f万): %d → %d rows (保留%.1f%%)",
        board,
        MIN_AMOUNT / 1e4,
        n_before,
        len(clean),
        len(clean) / n_before * 100,
    )

    if board == "main":
        board_df = clean[clean["board"] == "main"]
    else:
        board_df = clean[clean["board"].isin(["GEM", "STAR"])]

    logger.info(
        "[%s] 清洗后: %d rows, %d stocks",
        board,
        len(board_df),
        board_df["symbol"].nunique(),
    )

    if len(board_df) < 5000:
        logger.error("[%s] 样本不足, 退出", board)
        sys.exit(1)

    # ---- Feature engineering ----
    use_xrank = board != "main"
    logger.info("[%s] 特征工程开始 (cross_sectional_rank=%s)...", board, use_xrank)
    t_fe = time.time()

    fe = FeatureEngineV35()
    df = fe.build(board_df, cross_sectional_rank=use_xrank)
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)

    logger.info(
        "[%s] 特征工程完成: %d cols, %.1fs",
        board,
        len(df.columns),
        time.time() - t_fe,
    )

    # ---- IC screening ----
    candidates = FeatureEngineV35.feature_columns(df)
    valid = []
    for col in candidates:
        if col not in df.columns:
            continue
        if df[col].isna().mean() >= 0.95:
            continue
        if df[col].dtype == object:
            continue
        valid.append(col)

    logger.info(
        "[%s] 候选因子: %d → 预筛后: %d (剔除 %d)",
        board,
        len(candidates),
        len(valid),
        len(candidates) - len(valid),
    )

    # 标签覆盖
    for lc in ["label_1d_net", "label_3d_net", "label_5d_net"]:
        if lc in df.columns:
            logger.info(
                "[%s] %s non-null: %.1f%%", board, lc, df[lc].notna().mean() * 100
            )

    t_ic = time.time()
    screener = ICScreener(registry_path=REGISTRY_DIR)
    result = screener.screen(df, valid, window_id=window_id)

    strong = sum(1 for v in result["detail"].values() if v["grade"] == "strong")
    weak = sum(1 for v in result["detail"].values() if v["grade"] == "weak")
    dead = sum(1 for v in result["detail"].values() if v["grade"] == "dead")

    logger.info(
        "[%s] IC 筛选完成 (%.1fs): strong=%d weak=%d dead=%d → selected=%d",
        board,
        time.time() - t_ic,
        strong,
        weak,
        dead,
        len(result["factors"]),
    )

    # ---- Top-30 输出 ----
    top_ic = sorted(
        result["detail"].items(),
        key=lambda kv: max(
            kv[1].get("ic_1d", 0), kv[1].get("ic_3d", 0), kv[1].get("ic_5d", 0)
        ),
        reverse=True,
    )[:30]

    print(f"\n{'=' * 90}")
    print(f"  [{board.upper()}] Top-30 因子 (best IC 1d/3d/5d)")
    print(f"{'=' * 90}")
    print(
        f"{'Factor':<38s} {'IC_1d':>7s} {'IC_3d':>7s} {'IC_5d':>7s} {'AUC':>7s} {'RollM':>7s} {'Roll+%':>7s} {'ICIR':>6s} {'Grade'}"
    )
    print("-" * 90)
    for fname, detail in top_ic:
        print(
            f"{fname:<38s} {detail.get('ic_1d', 0):>7.4f} {detail.get('ic_3d', 0):>7.4f} "
            f"{detail.get('ic_5d', 0):>7.4f} {detail.get('auc', 0):>7.4f} "
            f"{detail.get('rolling_mean', 0):>7.4f} {detail.get('rolling_pos_ratio', 0):>7.4f} "
            f"{detail.get('icir', 0):>6.4f} {detail['grade']:>7s}"
        )

    # ---- 摘要 ----
    elapsed = time.time() - t0
    summary = {
        "board": board,
        "window_id": window_id,
        "timestamp": datetime.now().isoformat(),
        "elapsed_s": round(elapsed, 1),
        "n_rows": len(board_df),
        "n_stocks": int(board_df["symbol"].nunique()),
        "n_candidates": len(valid),
        "n_strong": strong,
        "n_weak": weak,
        "n_dead": dead,
        "n_selected": len(result["factors"]),
        "top_30": [
            {
                "factor": f,
                "ic_1d": d.get("ic_1d", 0),
                "ic_3d": d.get("ic_3d", 0),
                "ic_5d": d.get("ic_5d", 0),
                "auc": d.get("auc", 0),
                "grade": d.get("grade", ""),
            }
            for f, d in top_ic
        ],
        "n_strong_factors": len(
            [
                f
                for f in result["factors"]
                if result["detail"].get(f, {}).get("grade") == "strong"
            ]
        ),
        "n_weak_factors": len(
            [
                f
                for f in result["factors"]
                if result["detail"].get(f, {}).get("grade") == "weak"
            ]
        ),
    }

    summary_path = os.path.join(REGISTRY_DIR, f"ic_summary_{window_id}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n[{board.upper()}] DONE ({elapsed:.0f}s)")
    print(
        f"  strong={strong} weak={weak} dead={dead} selected={len(result['factors'])}"
    )
    print(f"  注册表: data/factor_registry/factors_{window_id}.json")
    print(f"  摘要:   {summary_path}")


if __name__ == "__main__":
    main()
