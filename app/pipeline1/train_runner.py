# -*- coding: utf-8 -*-
"""Pipeline-1 训练编排 (周频重训主链路)
=====================================================
面板 → 训练端清洗 (步骤0→3+5) → 特征 (FeatureEngineV35) →
标签 (路径标签 → 主标签 → 停牌/近端掩码) → FeatureSelector (Layer2 精选) →
双轨训练 (DualTrackTrainer.weekly_retrain) → 模型包落盘.

重训频率: **每周一次** (用户 2026-07-22 裁决: 周频, 非月频;
2026-07-26 重申). OOS IC >= 0.03 才切换新模型, 否则保留旧模型 + 告警.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from config.settings import data_others_path

from .cleaning_pipeline import CleaningPipeline
from .dual_track_trainer import DualTrackTrainer
from .feature_engine_v35 import FeatureEngineV35
from .feature_selector import (
    BRUTE_FAMILIES,
    BruteForceGenerator,
    FeatureSelector,
    apply_event_scope_screens,
)
from .label_engine import LabelEngine

logger = logging.getLogger(__name__)

MASK_RECENT_DAYS = 6  # 近端标签未成熟掩码 (与 test_pipeline1_v38 口径一致)


def prepare_board_frame(
    board_df: pd.DataFrame,
    features: FeatureEngineV35,
    float_shares_map: dict | None = None,
    cross_sectional_rank: bool = False,
    registry=None,  # FeatureRegistry | None
) -> pd.DataFrame:
    """单板块: 特征 → 路径标签 → 主标签 → 停牌/近端掩码 (训练准备标准序列).

    cross_sectional_rank: 仅双创开启截面排名 (主板大票定价效率高, 截面因子负贡献).
    """
    df = features.build(
        board_df,
        float_shares_map,
        cross_sectional_rank=cross_sectional_rank,
        registry=registry,
    )
    df = LabelEngine.build_path_labels(df)  # [E2] label_mdd_* + label_pain
    df = LabelEngine.build_labels(df)  # 主标签 label_*d + label_pm_*d + *_net
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


# 事件类稀疏特征子数据集 IC 筛入配置见 feature_selector.EVENT_SCOPE_SCREENS
# (增减持/大宗/LHB), 由 apply_event_scope_screens 统一执行 (训练主链路 + 独立工具共用).


def select_features(
    df: pd.DataFrame,
    board: str,
    tag: str,
    selector: FeatureSelector | None = None,
    registry=None,  # FeatureRegistry | None
) -> tuple[list[str], pd.DataFrame]:
    """Layer2 特征精选 (FeatureSelector) → 训练特征列表 + 增强面板.

    selector=None → 全量 FeatureEngine 特征 (测试/回退兼容).
    selector 存在 → bruteforce_dedup (MAIN) 或 gate_d (DUAL) 精选.

    Returns:
        (selected_feature_names, augmented_df)
        augmented_df 可能包含 bruteforce_dedup 注入的 BruteForce 特征列.
    """
    candidates = FeatureEngineV35.feature_columns(df)

    # ---- NaN-rate 预筛 + 类型过滤 (工程强制, 铁律 #1) ----
    # 事件类稀疏特征 (holder/LHB/大宗, 覆盖率 ~0.4%) 豁免: 只要非空行数 >=
    # MIN_SUPPORT 就保留, 由 scope_ic_union 在事件子数据集上评 IC 决定去留.
    NaN_DROP_THRESHOLD = 0.95
    MIN_SUPPORT = 1000
    valid_candidates = []
    for col in candidates:
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            if nan_rate >= NaN_DROP_THRESHOLD and df[col].notna().sum() < MIN_SUPPORT:
                continue
            # 剔除非数值列 (name/tradestatus 等字符串列会破坏 LightGBM)
            if df[col].dtype == object:
                continue
            valid_candidates.append(col)
    dropped = len(candidates) - len(valid_candidates)
    if dropped > 0:
        logger.warning(
            "[%s] NaN+类型预筛剔除 %d/%d 因子, 保留 %d",
            board,
            dropped,
            len(candidates),
            len(valid_candidates),
        )
    candidates = valid_candidates
    # ----------------------------------------------------------------

    if selector is None:
        logger.info("[%s] 无 FeatureSelector, 使用全量 %d 特征", board, len(candidates))
        return candidates, df

    board_cfg = selector.config.get(board, selector.config.get("fallback", {}))
    pipeline = board_cfg.get("pipeline", "ic_screener")

    try:
        selected = selector.select(df, board)

        # ── 子数据集 scope-IC 筛入 (事件类稀疏特征; 配置见
        #    feature_selector.EVENT_SCOPE_SCREENS) ──
        selected = apply_event_scope_screens(selected, df)

        # ── Post-selection BruteForce injection ──
        # _run_bruteforce_dedup generates BruteForce features internally for
        # correlation/dedup, but they land in a local df_exp copy.  Any
        # selected feature missing from the training df must be generated
        # here in a single pass (no pre-generation — avoids H1 double-gen
        # where second _eligible() picks up first-pass brute-force cols).
        missing = [f for f in selected if f not in df.columns]
        if missing:
            gen = BruteForceGenerator()
            raw_cols = gen._eligible(df)
            # 内存安全: 族批生成, 只注入缺失的选中列, 不物化全量 brute 列.
            keep_cols = []
            for fam in BRUTE_FAMILIES:
                new = gen.generate_family(df, fam, raw_cols=raw_cols, dtype="float32")
                pick = [c for c in missing if c in new.columns]
                if pick:
                    keep_cols.extend(pick)
                    df = df.join(new[pick])
                del new
            if keep_cols:
                logger.info(
                    "[%s] BruteForce 后注入 %d 个选中特征 (缺失 %d)",
                    board,
                    len(keep_cols),
                    len(missing),
                )

        df_col_set = set(df.columns)
        picked = [f for f in selected if f in df_col_set]
        if picked:
            logger.info(
                "[%s] FeatureSelector(%s) 精选 %d/%d 因子 (pool=%d)",
                board,
                pipeline,
                len(picked),
                len(candidates),
                len(selected),
            )
            return picked, df
        logger.warning(
            "[%s] FeatureSelector 全部淘汰, 回退全量 %d 因子",
            board,
            len(candidates),
        )
    except Exception as exc:  # 精选失败不阻断训练 (降级全量)
        logger.error("[%s] FeatureSelector 失败 (%s), 回退全量因子", board, exc)
    return candidates, df


def run_training(
    panel: pd.DataFrame,
    tag: str,
    model_dir: str = "models/pipeline1",
    registry_path: str = str(data_others_path("data/factor_registry")),
    float_shares_map: dict | None = None,
    use_ic_screen: bool = True,
    use_registry: bool = True,
    enable_adoption: bool = False,
    feature_list_path: str | None = None,
    selector_config: dict | None = None,
) -> dict:
    """周频重训主入口: 面板 → 双板块模型包.

    Args:
        panel: enrich 后的全市场面板 (panel_builder.assemble_panel 输出)
        tag: 模型包标签 (如 '2026W30'), 落盘 {board}_{tag}.pkl
        use_ic_screen: True → FeatureSelector (Layer2 精选);
                       False → 全量特征 (测试/回退)
        use_registry: True → 启用 FeatureRegistry 驱动的 dim 门控 + 特征裁剪
        enable_adoption: True → 启用自动采纳新面板列 (需 use_registry=True)
        feature_list_path: [已废弃] 忽略; FeatureSelector 直接运行, 不再从文件加载
        selector_config: FeatureSelector 配置覆盖 (None → 用默认 config)

    Returns:
        {board: {'path', 'oos': {...}, 'switched': bool, 'n_features': int}}
    """
    from .feature_registry import FeatureRegistry

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()

    # ── FeatureSelector (Layer2) — 替代 ICScreener ──
    selector = None
    if use_ic_screen:
        selector = FeatureSelector(
            config=selector_config,
            registry_dir=registry_path,
        )
        logger.info(
            "FeatureSelector 已初始化: MAIN=%s DUAL=%s",
            selector.config.get("main", {}).get("pipeline", "?"),
            selector.config.get("dual", {}).get("pipeline", "?"),
        )

    # ── P19 Registry setup ──
    registry = None
    if use_registry:
        reg_file = os.path.join(registry_path, "feature_registry.json")
        registry = FeatureRegistry(path=reg_file)
        if enable_adoption:
            registry.enable_adoption()
        # Auto-seed if empty
        if not registry.features:
            logger.info("Registry empty, seeding from panel sample...")
            try:
                sample = (
                    panel.groupby("symbol")
                    .apply(lambda g: g.head(min(50, len(g))))
                    .reset_index(drop=True)
                )
                registry._seed(sample)
            except Exception as exc:
                logger.warning(
                    "Registry seed failed (%s), continuing without registry", exc
                )
                registry = None

    main_df, dual_df = cleaner.run_train(panel)
    panels, cols_by_board = {}, {}
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        if len(board_df) == 0:
            logger.warning("[%s] 清洗后无样本, 跳过该板块训练", board)
            continue
        use_xrank = board != "main"  # 仅双创加截面排名 (主板大票定价有效, 截面负贡献)
        df = prepare_board_frame(
            board_df,
            features,
            float_shares_map,
            cross_sectional_rank=use_xrank,
            registry=registry,
        )
        cols, augmented_df = select_features(
            df, board, tag, selector, registry=registry
        )
        cols_by_board[board] = cols
        panels[board] = augmented_df

    if not panels:
        raise RuntimeError("训练面板为空: 主板/双创清洗后均无样本")

    trainer = DualTrackTrainer(model_dir=model_dir)
    results = trainer.weekly_retrain(panels, cols_by_board, tag)
    for board, res in results.items():
        res["n_features"] = len(cols_by_board[board])
        logger.info(
            "[%s] 模型包 %s | OOS weighted_IC=%.4f | switched=%s",
            board,
            res["path"],
            res["oos"].get("weighted_ic", 0.0),
            res["switched"],
        )
    return results
