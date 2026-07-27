# -*- coding: utf-8 -*-
"""Pipeline-1 训练编排 (周频重训主链路)
=====================================================
面板 → 训练端清洗 (步骤0→3+5) → 特征 (FeatureEngineV35) →
标签 (路径标签 → 主标签 → 停牌/近端掩码) → IC 筛选 (ICScreener) →
双轨训练 (DualTrackTrainer.weekly_retrain) → 模型包落盘.

重训频率: **每周一次** (用户 2026-07-22 裁决: 周频, 非月频;
2026-07-26 重申). OOS IC >= 0.03 才切换新模型, 否则保留旧模型 + 告警.
"""

from __future__ import annotations

import logging

import pandas as pd

from .cleaning_pipeline import CleaningPipeline
from .dual_track_trainer import DualTrackTrainer
from .feature_engine_v35 import FeatureEngineV35
from .ic_screener import ICScreener
from .label_engine import LabelEngine

logger = logging.getLogger(__name__)

MASK_RECENT_DAYS = 6  # 近端标签未成熟掩码 (与 test_pipeline1_v38 口径一致)


def prepare_board_frame(
    board_df: pd.DataFrame,
    features: FeatureEngineV35,
    float_shares_map: dict | None = None,
) -> pd.DataFrame:
    """单板块: 特征 → 路径标签 → 主标签 → 停牌/近端掩码 (训练准备标准序列)."""
    df = features.build(board_df, float_shares_map)
    df = LabelEngine.build_path_labels(df)  # [E2] label_mdd_* + label_pain
    df = LabelEngine.build_labels(df)  # 主标签 label_*d + label_pm_*d + *_net
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def select_features(
    df: pd.DataFrame,
    board: str,
    tag: str,
    screener: ICScreener | None = None,
) -> list[str]:
    """IC 筛选当期因子; 全部被淘汰时回退全量特征列 (告警, 不中断训练)."""
    candidates = FeatureEngineV35.feature_columns(df)

    # ---- NaN-rate 预筛 (工程强制, 铁律 #1 防止噪音特征污染模型) ----
    NaN_DROP_THRESHOLD = 0.95
    valid_candidates = []
    for col in candidates:
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate < NaN_DROP_THRESHOLD:
                valid_candidates.append(col)
    dropped = len(candidates) - len(valid_candidates)
    if dropped > 0:
        logger.warning(
            "[%s] NaN预筛剔除 %d/%d 因子 (>%.0f%% NaN), 保留 %d",
            board,
            dropped,
            len(candidates),
            NaN_DROP_THRESHOLD * 100,
            len(valid_candidates),
        )
    candidates = valid_candidates
    # ----------------------------------------------------------------

    if screener is None:
        return candidates
    try:
        result = screener.screen(df, candidates, window_id=f"{board}_{tag}")
        picked = [f for f in result["factors"] if f in candidates]
        if picked:
            logger.info(
                "[%s] IC 筛选保留 %d/%d 因子", board, len(picked), len(candidates)
            )
            return picked
        logger.warning("[%s] IC 筛选全部淘汰, 回退全量 %d 因子", board, len(candidates))
    except Exception as exc:  # 筛选失败不阻断训练 (降级全量)
        logger.error("[%s] IC 筛选失败 (%s), 回退全量因子", board, exc)
    return candidates


def run_training(
    panel: pd.DataFrame,
    tag: str,
    model_dir: str = "models/pipeline1",
    registry_path: str = "data/factor_registry",
    float_shares_map: dict | None = None,
    use_ic_screen: bool = True,
) -> dict:
    """周频重训主入口: 面板 → 双板块模型包.

    Args:
        panel: enrich 后的全市场面板 (panel_builder.assemble_panel 输出)
        tag: 模型包标签 (如 '2026W30'), 落盘 {board}_{tag}.pkl
        use_ic_screen: False 跳过 IC 筛选 (全量特征)

    Returns:
        {board: {'path', 'oos': {...}, 'switched': bool, 'n_features': int}}
    """
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    screener = ICScreener(registry_path=registry_path) if use_ic_screen else None

    main_df, dual_df = cleaner.run_train(panel)
    panels, cols_by_board = {}, {}
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        if len(board_df) == 0:
            logger.warning("[%s] 清洗后无样本, 跳过该板块训练", board)
            continue
        df = prepare_board_frame(board_df, features, float_shares_map)
        panels[board] = df
        cols_by_board[board] = select_features(df, board, tag, screener)

    if not panels:
        raise RuntimeError("训练面板为空: 主板/双创清洗后均无样本")

    trainer = DualTrackTrainer(model_dir=model_dir)
    results = trainer.weekly_retrain(panels, cols_by_board, tag)
    for board, res in results.items():
        res["n_features"] = len(cols_by_board[board])
        logger.info(
            "[%s] 模型包 %s | OOS IC(1d)=%.4f | switched=%s",
            board,
            res["path"],
            res["oos"]["ics"].get("1d_reg", 0.0),
            res["switched"],
        )
    return results
