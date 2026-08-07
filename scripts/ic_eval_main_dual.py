"""IC 评估与特征筛选 — MAIN vs DUAL 独立运行.

使用缓存的 panel_full_enriched_v3.parquet + 追加当日数据
→ FeatureEngineV35.build() (全量 dim01–dim30 特征计算)
→ CleaningPipeline 主板/双创拆分
→ ICScreener 逐板块 IC 筛选
→ 输出 factor_registry 注册表 per board
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.label_engine import LabelEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ic_eval_main_dual")

# ---- Config ----
PANEL_PATH = "data/panel_full_enriched_v3.parquet"
REGISTRY_DIR = "data/factor_registry"
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
MASK_RECENT_DAYS = 6  # 推理端近端掩码天数


def prepare_board_frame(
    board_df: pd.DataFrame,
    features: FeatureEngineV35,
    cross_sectional_rank: bool = False,
) -> pd.DataFrame:
    """单板块特征工程 + 标签 (复制 train_runner.prepare_board_frame)."""
    df = features.build(board_df, cross_sectional_rank=cross_sectional_rank)
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def main():
    t0 = time.time()

    # ---- Step 1: 加载缓存面板 + 追加当日数据 ----
    logger.info("加载缓存面板: %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    logger.info(
        "缓存面板: %d rows, %d cols, %d stocks, date=%s ~ %s",
        len(panel),
        len(panel.columns),
        panel["symbol"].nunique(),
        str(panel["date"].min())[:10],
        str(panel["date"].max())[:10],
    )

    # 追加当日 (2026-07-28) 数据
    today_str = datetime.now().strftime("%Y%m%d")
    max_date = str(panel["date"].max())[:10].replace("-", "")
    logger.info("缓存最新日期: %s, 今日: %s", max_date, today_str)

    if max_date < today_str:
        logger.info("追加今日 (%s) OHLCV + margin + lhb 数据...", today_str)
        try:
            supply = DataSupplyChain()
            panel = panel[panel["date"] < pd.to_datetime(today_str)]
            panel = supply.append_today_to_panel(
                panel,
                trade_date=today_str,
                sources=["ohlcv", "margin", "lhb"],  # northbound 已移除
            )
            logger.info(
                "追加完成: %d rows, %d stocks", len(panel), panel["symbol"].nunique()
            )
        except Exception as e:
            logger.warning("追加今日数据失败 (%s), 继续使用缓存数据", e)
    else:
        logger.info("缓存已含今日数据, 跳过追加")

    # ---- Step 2: 轻量清洗 (IC 评估用全量数据, 不做流动性/成交额过滤) ----
    # 仅: 主板/双创拆分 + 剔除ST + 剔除list_days<60(滚动IC需要)+ 剔除停牌
    from app.pipeline1.cleaning_pipeline import board_of

    # 确保 board 列存在
    if "board" not in panel.columns:
        panel = panel.copy()
        panel["board"] = panel["symbol"].map(board_of)

    # 基础过滤 (最小化, 保留全量数据做 IC 评估)
    clean = panel[~panel["is_suspended"].astype(bool)]

    # 关键字段非空
    key_cols = ["open", "high", "low", "close", "close_hfq", "volume", "amount"]
    clean = clean.dropna(subset=key_cols)

    # 板块拆分
    main_df = clean[clean["board"] == "main"]
    dual_df = clean[clean["board"].isin(["GEM", "STAR"])]

    logger.info(
        "全量清洗 (无流动性过滤): 主板=%d rows/%d stocks | 双创=%d rows/%d stocks",
        len(main_df),
        main_df["symbol"].nunique(),
        len(dual_df),
        dual_df["symbol"].nunique(),
    )

    if len(main_df) < 5000:
        logger.error("主板样本不足 (%d rows), 无法做 IC 评估", len(main_df))
        return
    if len(dual_df) < 5000:
        logger.error("双创样本不足 (%d rows), 无法做 IC 评估", len(dual_df))
        return

    # ---- Step 3: 逐板块特征工程 + IC 筛选 ----
    features = FeatureEngineV35()
    screener = ICScreener(registry_path=REGISTRY_DIR)

    results = {}
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        logger.info("=" * 60)
        logger.info(
            "[%s] 开始特征工程 (%d rows, %d stocks)...",
            board,
            len(board_df),
            board_df["symbol"].nunique(),
        )

        use_xrank = board != "main"
        t_board = time.time()
        df = prepare_board_frame(board_df, features, cross_sectional_rank=use_xrank)
        logger.info(
            "[%s] 特征工程完成: %d cols, %.1fs",
            board,
            len(df.columns),
            time.time() - t_board,
        )

        # 候选因子列表
        candidates = FeatureEngineV35.feature_columns(df)
        logger.info("[%s] 候选因子: %d", board, len(candidates))

        # NaN + 类型预筛
        valid = []
        for col in candidates:
            if col not in df.columns:
                continue
            nan_rate = df[col].isna().mean()
            if nan_rate >= 0.95:
                continue
            if df[col].dtype == object:
                continue
            valid.append(col)
        dropped = len(candidates) - len(valid)
        if dropped:
            logger.info(
                "[%s] NaN/类型预筛剔除 %d/%d, 保留 %d",
                board,
                dropped,
                len(candidates),
                len(valid),
            )

        # 验证标签可用
        [c for c in df.columns if c.startswith("label_")]
        for lc in ["label_1d_net", "label_1d", "label_3d_net", "label_5d_net"]:
            if lc in df.columns:
                nn = df[lc].notna().mean()
                logger.info("[%s] %s non-null: %.1f%%", board, lc, nn * 100)

        # IC 筛选
        t_ic = time.time()
        window_id = f"{board}_{TAG}"
        result = screener.screen(df, valid, window_id=window_id)
        strong = sum(1 for v in result["detail"].values() if v["grade"] == "strong")
        weak = sum(1 for v in result["detail"].values() if v["grade"] == "weak")
        dead = sum(1 for v in result["detail"].values() if v["grade"] == "dead")
        logger.info(
            "[%s] IC 筛选完成 (%.1fs): strong=%d, weak=%d, dead=%d, selected=%d",
            board,
            time.time() - t_ic,
            strong,
            weak,
            dead,
            len(result["factors"]),
        )

        # Top-30 by best IC
        top_ic = sorted(
            result["detail"].items(),
            key=lambda kv: max(
                kv[1].get("ic_1d", 0),
                kv[1].get("ic_3d", 0),
                kv[1].get("ic_5d", 0),
            ),
            reverse=True,
        )[:30]

        print(f"\n{'=' * 90}")
        print(f"  [{board.upper()}] Top-30 因子 by best IC (1d/3d/5d)")
        print(f"{'=' * 90}")
        print(
            f"{'Factor':<38s} {'IC_1d':>7s} {'IC_3d':>7s} {'IC_5d':>7s} "
            f"{'AUC':>7s} {'RollM':>7s} {'Roll+%':>7s} {'ICIR':>6s} {'NW_t3d':>7s} {'Grade'}"
        )
        print("-" * 90)
        for fname, detail in top_ic:
            print(
                f"{fname:<38s} {detail.get('ic_1d', 0):>7.4f} {detail.get('ic_3d', 0):>7.4f} "
                f"{detail.get('ic_5d', 0):>7.4f} {detail.get('auc', 0):>7.4f} "
                f"{detail.get('rolling_mean', 0):>7.4f} {detail.get('rolling_pos_ratio', 0):>7.4f} "
                f"{detail.get('icir', 0):>6.4f} {detail.get('nw_t_3d', 0):>7.4f} "
                f"{detail['grade']:>7s}"
            )

        results[board] = {
            "window_id": window_id,
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
        }

    # ---- Step 4: 输出摘要 ----
    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"  SUMMARY — IC Evaluation Complete ({elapsed:.0f}s)")
    print(f"{'=' * 90}")
    for board in ("main", "dual"):
        r = results[board]
        print(
            f"  [{board.upper()}] candidates={r['n_candidates']} "
            f"strong={r['n_strong']} weak={r['n_weak']} dead={r['n_dead']} "
            f"→ selected={r['n_selected']}"
        )

    # 保存摘要
    summary_path = os.path.join(REGISTRY_DIR, f"ic_eval_summary_{TAG}.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "panel": PANEL_PATH,
                "elapsed_s": round(elapsed, 1),
                "config": {
                    "min_list_days": 60,
                    "mask_recent_days": MASK_RECENT_DAYS,
                    "note": "全量数据IC评估, 无流动性/成交额过滤",
                },
                "results": results,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    logger.info("摘要: %s", summary_path)

    # 列出本次生成的文件
    print("\n本次生成的文件:")
    for f in sorted(os.listdir(REGISTRY_DIR)):
        if TAG in f:
            print(f"  data/factor_registry/{f}")


if __name__ == "__main__":
    main()
