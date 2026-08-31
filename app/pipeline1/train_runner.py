"""Pipeline-1 训练编排 (周频重训主链路)
=====================================================
面板 → 训练端清洗 (步骤0→3+5) → 特征 (FeatureEngineV35) →
标签 (路径标签 → 主标签 → 停牌/近端掩码) → FeatureSelector (Layer2 精选) →
双轨训练 (DualTrackTrainer.weekly_retrain) → 模型包落盘.

重训频率: **每周一次** (用户 2026-07-22 裁决: 周频, 非月频;
2026-07-26 重申). OOS IC >= 0.03 才切换新模型, 否则保留旧模型 + 告警.
"""

from __future__ import annotations

import gc
import logging
import os
import time

import pandas as pd

from config.settings import (
    LEGACY_EXCESS_LABEL_BOARDS,
    LEGACY_MKT_EXPECT_WINDOW,
    data_others_path,
)

from .cleaning_pipeline import CleaningPipeline
from .dual_track_trainer import DualTrackTrainer
from .feature_engine_v35 import FeatureEngineV35
from .feature_selector import (
    BRUTE_FAMILIES,
    BruteForceGenerator,
    FeatureSelector,
    apply_event_scope_screens,
)
from .label_engine import MASK_RECENT_DAYS, LabelEngine

logger = logging.getLogger(__name__)


EXCESS_HORIZONS = (3, 5, 10)


def demean_excess_labels(
    df: pd.DataFrame, horizons: tuple = EXCESS_HORIZONS
) -> pd.DataFrame:
    """label_pm_{k}d_net 板内按日去均值 (超额标签, 2026-08-29 main 采纳).

    同日减同一常数 → 当日截面排名/形状不变, 跨日水平中性化; 按日 rank IC 不变,
    validate_oos 的 OOS 判据对新旧标签直接可比.
    """
    for k in horizons:
        c = f"label_pm_{k}d_net"
        if c in df.columns:
            df[c] = df[c] - df.groupby("date")[c].transform("mean")
    return df


def compute_mkt_expected(df: pd.DataFrame, window: int) -> dict[str, float]:
    """bundle 市场均值常数: 近 window 个已实现决策日的板内等权日均值 (按视界).

    与展示层 _mkt_expected 同窗同滞后 (label 未实现即 NaN 剔除), 无 look-ahead;
    推理端加回它复原绝对口径 (闸/概率阈值语义).
    """
    out: dict[str, float] = {}
    for k in EXCESS_HORIZONS:
        c = f"label_pm_{k}d_net"
        if c not in df.columns:
            continue
        real = df.dropna(subset=[c])
        days = sorted(real["date"].unique())[-window:]
        daily = real[real["date"].isin(days)].groupby("date")[c].mean()
        out[f"mkt_expected_{k}d"] = float(daily.mean()) if len(daily) else 0.0
    return out


def prepare_board_frame(
    board_df: pd.DataFrame,
    features: FeatureEngineV35,
    float_shares_map: dict | None = None,
    cross_sectional_rank: bool = False,
    registry=None,  # FeatureRegistry | None
    label_excess: bool = False,
) -> pd.DataFrame:
    """单板块: 特征 → 路径标签 → 主标签 → 停牌/近端掩码 (训练准备标准序列).

    cross_sectional_rank: 仅双创开启截面排名 (主板大票定价效率高, 截面因子负贡献).
    label_excess: True → 回归头净标签板内按日去均值 (超额标签; cls/prob 标签不动).
    """
    df = features.build(
        board_df,
        float_shares_map,
        cross_sectional_rank=cross_sectional_rank,
        registry=registry,
    )
    df = LabelEngine.build_path_labels(df)  # [E2] label_mdd_* + label_pain
    df = LabelEngine.build_labels(df)  # 主标签 label_*d + label_pm_*d + *_net
    # mask 先于去均值: 停牌假标签不得进入日均基准
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    if label_excess:
        # 常数须在去均值前计算 (去均值后日均按构造≈0); 最后挂 attrs 免中间拷贝丢失
        mkt = compute_mkt_expected(df, LEGACY_MKT_EXPECT_WINDOW)
        df = demean_excess_labels(df)
        df.attrs["mkt_expected"] = mkt
    return df


# 事件类稀疏特征子数据集 IC 筛入配置见 feature_selector.EVENT_SCOPE_SCREENS
# (增减持/大宗/LHB), 由 apply_event_scope_screens 统一执行 (训练主链路 + 独立工具共用).


def select_features(
    df: pd.DataFrame,
    board: str,
    tag: str,
    selector: FeatureSelector | None = None,
    registry=None,  # FeatureRegistry | None
    fallback_boards: set[str] | None = None,
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

    if selector is None or (fallback_boards and board in fallback_boards):
        # fallback_boards: 跳过该板块的 FeatureSelector (如 main 的 bruteforce_dedup
        # 选择过大必 OOM 时), 直用全量过滤后特征; 只作用于指定板块, 其余板块照常精选.
        logger.info(
            "[%s] %s, 使用全量 %d 特征",
            board,
            "fallback_boards 命中" if fallback_boards else "无 FeatureSelector",
            len(candidates),
        )
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
            # 内存安全 (2026-08-11 OOM 修复): generate_family 物化全族
            # (1223918, 2544)=11.6GB 单块 → _ArrayMemoryError → FeatureSelector
            # 回退全量 → main 3d_cls 塌缩. 改用 generate_columns 逐 symbol 流式,
            # 只驻留缺失选中列; 单次 join 避免多次 block consolidation.
            need = set(missing)
            keep_cols = []
            picks = []
            for fam in BRUTE_FAMILIES:
                new = gen.generate_columns(
                    df, fam, need, raw_cols=raw_cols, dtype="float32"
                )
                if new is None or not len(new.columns):
                    continue
                keep_cols.extend(new.columns)
                picks.append(new)
            if picks:
                # 内存安全 (2026-08-12): df.join 全宽帧 → block consolidation OOM.
                # generate_columns 逐 symbol 保留 df 原索引 (0..n-1 已 sort reset),
                # 位置赋值与 join 逐元素一致.
                _brute = pd.concat(picks, axis=1)
                for _c in _brute.columns:
                    df[_c] = _brute[_c].to_numpy()
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
    panel: pd.DataFrame | None = None,
    tag: str | None = None,
    model_dir: str = "models/pipeline1",
    registry_path: str = str(data_others_path("data/factor_registry")),
    float_shares_map: dict | None = None,
    use_ic_screen: bool = True,
    use_registry: bool = True,
    enable_adoption: bool = False,
    feature_list_path: str | None = None,
    selector_config: dict | None = None,
    fallback_boards: set[str] | None = None,
    boards: tuple[str, ...] = ("main", "dual"),
    *,
    panel_path: str | None = None,
) -> dict:
    """周频重训主入口: 面板 → 板块模型包 (默认双板).

    Args:
        panel: enrich 后的全市场面板 (panel_builder.assemble_panel 输出)
        tag: 模型包标签 (如 '2026W30'), 落盘 {board}_{tag}.pkl
        use_ic_screen: True → FeatureSelector (Layer2 精选);
                       False → 全量特征 (测试/回退)
        use_registry: True → 启用 FeatureRegistry 驱动的 dim 门控 + 特征裁剪
        enable_adoption: True → 启用自动采纳新面板列 (需 use_registry=True)
        feature_list_path: [已废弃] 忽略; FeatureSelector 直接运行, 不再从文件加载
        selector_config: FeatureSelector 配置覆盖 (None → 用默认 config)
        fallback_boards: 跳过 FeatureSelector、直用全量特征的板块子集 (None → 全部精选)
        boards: 只重训的板块子集 (默认双板; ('dual',) 只训双创, 省内存省时)

    Returns:
        {board: {'path', 'oos': {...}, 'switched': bool, 'n_features': int}}
    """
    from .feature_registry import FeatureRegistry

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    t_start = time.time()

    # ── 内存 (2026-08-13 修复): 全量重训 build 阶段 OOM (dim17 峰值贴 commit 上限) ──
    # 根因: 调用方 (scripts/_retrain_legacy_full.py) 在 run_training 全程持有 panel 引用,
    # 函数内 `del panel` 只删局部名, 面板 (~2GB) 在特征 build 阶段仍驻留 → 峰值超标.
    # panel_path 模式: run_training 自己持有 panel, cleaning 后 del panel 真正归还内存.
    if panel is None and panel_path is not None:
        from .cleaning_pipeline import load_panel_v3

        panel = load_panel_v3(path=panel_path)
        cut = panel["date"].max() - pd.DateOffset(years=3)
        panel = panel[panel["date"] >= cut]
        logger.info(
            "[panel_path] V3 直读 + 3y 窗口: %d rows max=%s",
            len(panel),
            panel["date"].max().date(),
        )
    elif panel is None:
        raise ValueError("run_training: panel 与 panel_path 必须提供其一")
    logger.info("[timing] panel load: %.1fs", time.time() - t_start)

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

    # 内存分期: 逐板块 prepare→select→train→释放, 任意时刻只持一份增强帧.
    # 两板块是独立模型 (独立特征/独立 OOS), 分期只降内存峰值, 不改模型内容.
    # boards 单板时 board=run_train 只清洗该板 (另一板返回空帧, 省内存).
    run_board = boards[0] if len(boards) == 1 else None
    t_clean = time.time()
    board_dfs = dict(zip(("main", "dual"), cleaner.run_train(panel, board=run_board)))
    board_dfs = {b: board_dfs[b] for b in boards}
    del panel  # run_train 后 panel 不再需要 → 尽早归还内存
    gc.collect()
    logger.info("[timing] cleaning (run_train): %.1fs", time.time() - t_clean)
    trainer = DualTrackTrainer(model_dir=model_dir)
    results: dict = {}
    for board in boards:
        board_df = board_dfs.pop(board)
        if len(board_df) == 0:
            logger.warning("[%s] 清洗后无样本, 跳过该板块训练", board)
            continue
        use_xrank = board != "main"  # 仅双创加截面排名 (主板大票定价有效, 截面负贡献)
        label_excess = board in LEGACY_EXCESS_LABEL_BOARDS
        extras = None
        t_feat = time.time()
        df = prepare_board_frame(
            board_df,
            features,
            float_shares_map,
            cross_sectional_rank=use_xrank,
            registry=registry,
            label_excess=label_excess,
        )
        if label_excess:
            # 常数已在 prepare_board_frame (去均值前) 算好挂 attrs, 须在 select_features 释放 df 前取出
            extras = {
                "label_excess": True,
                **df.attrs.get("mkt_expected", {}),
            }
            logger.info(
                "[%s] 超额标签: 市场均值常数 %s",
                board,
                {k: f"{v:+.4f}" for k, v in extras.items() if k != "label_excess"},
            )
        # 释放清洗切片: FeatureSelector 峰值期不白占 ~2GB 中间帧
        del board_df
        gc.collect()
        logger.info("[timing][%s] feature build: %.1fs", board, time.time() - t_feat)
        t_sel = time.time()
        cols, augmented_df = select_features(
            df,
            board,
            tag,
            selector,
            registry=registry,
            fallback_boards=fallback_boards,
        )
        # 释放中间特征帧 (仅保留增强帧用于训练)
        del df
        gc.collect()
        logger.info("[timing][%s] feature selection: %.1fs", board, time.time() - t_sel)
        t_train = time.time()
        board_results = trainer.weekly_retrain(
            {board: augmented_df}, {board: cols}, tag, resume=True, extras=extras
        )
        logger.info("[timing][%s] model training: %.1fs", board, time.time() - t_train)
        n_features = len(cols)
        del augmented_df, cols
        gc.collect()
        res = board_results[board]
        res["n_features"] = n_features
        results[board] = res
        logger.info(
            "[%s] 模型包 %s | OOS weighted_IC=%.4f | switched=%s",
            board,
            res["path"],
            res["oos"].get("weighted_ic", 0.0),
            res["switched"],
        )

    if not results:
        raise RuntimeError("训练面板为空: 主板/双创清洗后均无样本")
    return results
