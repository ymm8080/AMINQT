# -*- coding: utf-8 -*-
"""3年数据训练+预测完整流程.

面板优先级: panel_full_enriched_v2 → panel_full_enriched → panel_3y (fallback).
使用 enriched 面板 (3,227 stocks + alt data) 以保证训练特征完整性.
缺失的 alt 数据源按需补充拉取, 不重复已存在的列.
"""
import sys
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAG = "2026W31_3y"

# 各数据源在面板中对应的特征列 (任一存在即视为该源已 enrich)
_SOURCE_SIGNATURE_COLS: dict[str, list[str]] = {
    "northbound":     ["north_net_buy_sh", "north_net_buy_sz"],
    "lhb":            ["lhb_net_buy", "lhb_buy_amt"],
    "sector_index":   ["sw_ret_1d", "sw_index_close"],
    "margin":         ["margin_balance"],
    "fina_indicator": ["roe", "roa", "gross_margin"],
    "holdernumber":   ["holder_count"],
    "holdertrade":    ["sh_net_change_sign"],
    "daily_basic":    ["pe_ttm", "pb", "total_mv"],
    "stk_limit":      ["up_limit_raw"],
    "cyq_tushare":    ["his_low", "winner_rate", "weight_avg"],
}
ALL_ALT_SOURCES = list(_SOURCE_SIGNATURE_COLS.keys())

# 数据陈旧阈值: 列的 NaN 率超过此值视为该源需要 re-enrich (refresh=True)
# 不同类型数据有不同的预期稀疏度:
_STALENESS_THRESHOLDS: dict[str, float] = {
    "northbound":     0.50,  # 日频市场级数据, >50% NaN 说明 fetch 不全
    "sector_index":   0.50,  # 同上
    "margin":         0.80,  # 日频个股数据, 仅两融标的 (~55% stocks)
    "daily_basic":    0.50,  # 日频全市场
    "stk_limit":      0.50,  # 日频全市场
    "fina_indicator": 0.95,  # 季频 PIT, 天然稀疏 (~1/63 交易日有新公告)
    "holdernumber":   0.95,  # 季频, 同上
    "cyq_tushare":   0.50,  # Tushare cyq_perf 日频, 全A覆盖
    # lhb / holdertrade: 事件驱动, 天然极度稀疏, 永不判定为 stale
}


def _detect_missing_sources(panel: pd.DataFrame) -> list[str]:
    """检测面板中缺少哪些 alt 数据源的列, 只补充拉取缺失的源."""
    missing = []
    for src, sig_cols in _SOURCE_SIGNATURE_COLS.items():
        if not any(c in panel.columns for c in sig_cols):
            missing.append(src)
    return missing


def _detect_stale_sources(panel: pd.DataFrame) -> list[str]:
    """检测面板中哪些 alt 数据源列存在但 NaN 率过高 (fetch 不全 → 需 refresh).

    两个条件任一触发即判定为 stale:
    1. 行级 NaN 率: 最不空的签名列 NaN 率 > 该源阈值
    2. 股级覆盖率: 所有签名列全 NaN 的股票占比 > 50% (fetch 只覆盖了少数股票)
    """
    stale = []
    for src, sig_cols in _SOURCE_SIGNATURE_COLS.items():
        threshold = _STALENESS_THRESHOLDS.get(src)
        if threshold is None:
            continue  # lhb/holdertrade: never stale
        present = [c for c in sig_cols if c in panel.columns]
        if not present:
            continue  # 完全缺失 → _detect_missing_sources 处理

        # 条件1: 行级 NaN 率
        min_nan = min(panel[c].isna().mean() for c in present)
        row_stale = min_nan > threshold

        # 条件2: 股级覆盖率 (所有签名列全 NaN 的股票占比)
        all_nan_mask = panel[present[0]].isna()
        for c in present[1:]:
            all_nan_mask = all_nan_mask & panel[c].isna()
        stock_no_data_frac = panel.loc[all_nan_mask, "symbol"].nunique() / max(panel["symbol"].nunique(), 1)
        stock_stale = stock_no_data_frac > 0.50

        if row_stale or stock_stale:
            stale.append(src)

    return stale


def main():
    from app.pipeline1.train_runner import run_training
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor
    from app.pipeline1.panel_builder import enrich_cyq, enrich_alt_data
    from app.pipeline1.data_supply import DataSupplyChain

    # --- 加载面板: 优先 enriched 版本 (3,227 stocks + alt data) ---
    _PANEL_CANDIDATES = [
        "data/panel_full_enriched_v3.parquet",
        "data/panel_full_enriched_v2.parquet",
    ]
    panel_path = None
    for path in _PANEL_CANDIDATES:
        if os.path.exists(path):
            panel_path = path
            break
    if panel_path is None:
        raise FileNotFoundError(
            f"无可用面板文件, 已尝试: {_PANEL_CANDIDATES}"
        )
    panel = pd.read_parquet(panel_path)
    logger.info("加载面板: %s (%d stocks, %d rows, %d cols)",
                panel_path, panel["symbol"].nunique(), len(panel), len(panel.columns))

    # --- 检测数据缺口 (先检测, 再决定 CYQ 策略) ---
    missing_sources = _detect_missing_sources(panel)
    stale_sources = _detect_stale_sources(panel)
    # 允许通过环境变量跳过特定源 (逗号分隔, 用于 API 不可用时)
    _skip = set(os.environ.get("SKIP_SOURCES", "").split(",")) - {""}
    missing_sources = [s for s in missing_sources if s not in _skip]
    stale_sources = [s for s in stale_sources if s not in _skip]
    if _skip:
        logger.info("跳过数据源 (SKIP_SOURCES): %s", sorted(_skip))
    will_fetch_cyq = "cyq_tushare" in missing_sources or "cyq_tushare" in stale_sources

    # --- CYQ 筹码分布: Tushare cyq_perf 优先, 跳过慢速 OHLCV 计算器 ---
    if will_fetch_cyq:
        logger.info("cyq_tushare 将在 alt enrich 中拉取, 跳过 enrich_cyq (OHLCV 计算器)")
    else:
        panel = enrich_cyq(panel, cyq_cache="data/cyq_panel.parquet")
        cyq_cols = [c for c in panel.columns if c.startswith(('pct_', 'cost_', 'benefit', 'weight_avg'))]
        logger.info("CYQ (OHLCV calculator): %d cols (%s...)", len(cyq_cols),
                    ', '.join(cyq_cols[:8]) if cyq_cols else 'none')

    # --- 替代数据: 按需补充缺失源 + 陈旧源强制 refresh ---
    if missing_sources or stale_sources:
        if missing_sources:
            logger.info("检测到缺失 alt 数据源: %s", missing_sources)
        if stale_sources:
            logger.info("检测到陈旧 alt 数据源 (NaN率过高): %s", stale_sources)

        try:
            _supply = DataSupplyChain()
            if missing_sources:
                panel = enrich_alt_data(panel, _supply, sources=missing_sources)
            if stale_sources:
                panel = enrich_alt_data(panel, _supply, sources=stale_sources, refresh=True)
            logger.info("替代数据 enrich 完成, 面板: %s", panel.shape)
        except Exception as e:
            logger.warning("替代数据 enrich 失败 (不阻断): %s", e)
    else:
        logger.info("所有 alt 数据源已就绪, 跳过 enrich")

    enrich_only = os.environ.get("ENRICH_ONLY", "0") == "1"
    if enrich_only:
        # 保存 enriched panel 为新版本, 后续训练直接加载
        panel.to_parquet("data/panel_full_enriched_v3.parquet", index=False)
        logger.info("ENRICH_ONLY=1: 面板已保存到 panel_full_enriched_v3.parquet, 跳过训练")
        return

    results = run_training(panel, tag=TAG, use_ic_screen=True)

    for board, res in results.items():
        logger.info(f"[训练] {board}: OOS_IC(1d)={res['oos']['ics'].get('1d_reg', 0):.4f} "
                     f"feats={res['n_features']} switched={res['switched']}")

    # --- 预测 ---
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"\n清洗后: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)

    output = {"tag": TAG, "panel_shape": list(panel.shape), "boards": {}}

    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            logger.info(f"{board} 跳过")
            continue

        feats = features.build(df, cross_sectional_rank=(board != "main"))
        logger.info(f"\n{'='*60}")
        logger.info(f"{board} 预测 (features: {feats.shape})")
        logger.info(f"{'='*60}")

        predictor = V35Predictor(bundles)
        pred = predictor.predict(feats, board)

        board_out = {"pred_stats": {}, "model_info": {}, "oos": {}}

        for col in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up']:
            if col in pred.columns:
                vals = pred[col].dropna()
                if len(vals) > 0:
                    stats = {
                        "mean": round(float(vals.mean()), 6),
                        "std": round(float(vals.std()), 6),
                        "min": round(float(vals.min()), 6),
                        "max": round(float(vals.max()), 6),
                        "n": int(len(vals)),
                    }
                    board_out["pred_stats"][col] = stats
                    logger.info(f"  {col}: mean={vals.mean():.6f} std={vals.std():.6f} "
                                f"min={vals.min():.6f} max={vals.max():.6f} n={len(vals)}")

        # 模型信息
        with open(bundles[board], 'rb') as f:
            bundle = pickle.load(f)

        for kind in ('1d_reg', '3d_reg', '5d_reg', '1d_cls'):
            if kind in bundle.get("models", {}):
                model = bundle["models"][kind][0]
                bi = getattr(model, 'best_iteration_', None)
                ne = model.n_estimators
                fi = model.feature_importances_
                nonzero = int((fi > 0).sum())
                top5_share = float(np.sort(fi)[::-1][:5].sum() / fi.sum()) if fi.sum() > 0 else 0
                board_out["model_info"][kind] = {
                    "best_iter": int(bi) if bi is not None else None,
                    "n_estimators": int(ne),
                    "n_features": int(model.n_features_in_) if hasattr(model, 'n_features_in_') else None,
                    "nonzero_features": nonzero,
                    "top5_share": round(top5_share, 4),
                }
                logger.info(f"  {kind}: best_iter={bi}/{ne} features={model.n_features_in_} "
                            f"nonzero={nonzero} top5={top5_share:.2%}")

        cal = bundle.get("calibrator")
        if cal:
            method = getattr(cal, 'method', type(cal).__name__)
            board_out["calibrator"] = method
            logger.info(f"  calibrator: {method}")

        oos = bundle.get("oos", {})
        if oos:
            ics = oos.get("ics", {})
            for k, v in ics.items():
                board_out["oos"][k] = round(float(v), 4)
                logger.info(f"  OOS IC({k}): {v:.4f}")

        output["boards"][board] = board_out

    out_path = f"result_{TAG}.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\n完整结果已保存到 {out_path}")

if __name__ == "__main__":
    main()
