#!/usr/bin/env python3
"""用已训练的 3y 模型跑预测, 不重训.

输出:
  1. Pipeline IC 摘要 (signed Rank IC, main/dual)
  2. 股票推荐清单 (1D/3D/5D 预测 + pain_prob + 分位数 + 关键参数)
  3. 每只股票的入选原因分析
  4. Markdown 报告文件 (WORM: 带日期后缀, 不覆盖)
"""
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TAG = "2026W31_3y"
REPORT_DIR = "data/prediction_reports"


# ============================================================
# 1. Signed OOS Rank IC (test segment)
# ============================================================
def _compute_signed_ic(feat_df, bundle, board):
    """计算 test 段 (最近 60 交易日, 排除当日) 的 signed Rank IC.

    abs_mean=False → 保留正负号, 反映模型方向性预测力.
    """
    from app.utils.daily_rank_ic import mean_rank_ic

    cols = bundle["feature_cols"]
    models = bundle["models"]

    # 确保 label 列存在: 优先用 bundle 存储的 label, 缺失则从 close 计算
    feat_df = feat_df.copy()
    close_col = "close_hfq" if "close_hfq" in feat_df.columns else "close"
    for horizon in (1, 3, 5):
        label_name = models[f"{horizon}d_reg"][1]
        if label_name in feat_df.columns:
            continue
        for fb in (f"label_{horizon}d_net", f"label_{horizon}d"):
            if fb in feat_df.columns:
                label_name = fb
                break
        else:
            if close_col in feat_df.columns:
                feat_df[f"_label_{horizon}d"] = (
                    feat_df.groupby("symbol")[close_col].shift(-horizon)
                    / feat_df[close_col] - 1
                )

    dates = sorted(feat_df["date"].unique())
    if len(dates) < 10:
        return {"1d_reg": 0.0, "3d_reg": 0.0, "5d_reg": 0.0}

    # test 段: 最后 60 个交易日, 排除当日 (当日无 future label)
    n_test = min(60, len(dates) - 1)
    test_dates = dates[-(n_test + 1):-1]
    test_feat = feat_df[feat_df["date"].isin(test_dates)].copy()

    if len(test_feat) == 0:
        return {"1d_reg": 0.0, "3d_reg": 0.0, "5d_reg": 0.0}

    # 对 test 段跑模型推理
    X = np.nan_to_num(
        test_feat[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    test_feat["_pred_1d"] = models["1d_reg"][0].predict(X)
    test_feat["_pred_3d"] = models["3d_reg"][0].predict(X)
    test_feat["_pred_5d"] = models["5d_reg"][0].predict(X)

    ic_results = {}
    for kind, pred_col, horizon in [
        ("1d_reg", "_pred_1d", 1),
        ("3d_reg", "_pred_3d", 3),
        ("5d_reg", "_pred_5d", 5),
    ]:
        label_name = models[kind][1]
        # resolve actual label column
        resolved = None
        for cand in (label_name, f"label_{horizon}d_net", f"label_{horizon}d",
                     f"_label_{horizon}d"):
            if cand in test_feat.columns:
                resolved = cand
                break
        if resolved is None:
            ic_results[kind] = 0.0
            continue
        # SIGNED IC: abs_mean=False
        ic = mean_rank_ic(test_feat, pred_col, resolved, abs_mean=False)
        ic_results[kind] = float(ic)

    return ic_results


# ============================================================
# 2. Per-stock reason analysis
# ============================================================
def _build_reasons(row):
    """生成每只股票的入选原因分析 (基于筛选条件 + 预测值)."""
    reasons = []

    # --- 概率 ---
    prob = row.get("prob_up", 0)
    if prob >= 0.55:
        reasons.append(f"prob_up={prob:.3f} >= 0.55 (高胜率)")
    else:
        reasons.append(f"prob_up={prob:.3f}")

    # --- 收益预测 ---
    r1 = row.get("pred_ret_1d", 0)
    r3 = row.get("pred_ret_3d", 0)
    r5 = row.get("pred_ret_5d", 0)
    if r3 > 0.01:
        reasons.append(f"pred_ret_3d={r3:.4f} > 1% (3日强预期收益)")
    if r5 > 0.01:
        reasons.append(f"pred_ret_5d={r5:.4f} > 1% (5日趋势正向)")
    if r1 > 0:
        reasons.append(f"pred_ret_1d={r1:.4f} > 0 (次日正向信号)")

    # --- 风险: pain_prob ---
    pain = row.get("pain_prob", None)
    if pain is not None and not pd.isna(pain):
        if pain < 0.35:
            reasons.append(f"pain_prob={pain:.3f} < 0.35 (低回撤风险)")
        else:
            reasons.append(f"pain_prob={pain:.3f}")

    # --- 分位数: 下行/上行区间 ---
    q10 = row.get("pred_q10", None)
    q50 = row.get("pred_q50", None)
    q90 = row.get("pred_q90", None)
    if q10 is not None and not pd.isna(q10):
        reasons.append(f"pred_q10={q10:.4f} (10%分位下行预测)")
    if q50 is not None and not pd.isna(q50):
        reasons.append(f"pred_q50={q50:.4f} (中位预测)")
    if q90 is not None and not pd.isna(q90):
        reasons.append(f"pred_q90={q90:.4f} (90%分位上行预测)")

    # --- 不确定性 ---
    uw = row.get("uncertainty_width", None)
    if uw is not None and not pd.isna(uw):
        reasons.append(f"uncertainty_width={uw:.4f} (预测离散度)")

    # --- 波动率 ---
    atr = row.get("ATR_pct", None)
    if atr is not None and not pd.isna(atr):
        if atr < 0.06:
            reasons.append(f"ATR_pct={atr:.4f} < 6% (低波动)")
        else:
            reasons.append(f"ATR_pct={atr:.4f}")

    # --- 流动性 ---
    adv = row.get("adv20", None)
    if adv is not None and not pd.isna(adv):
        reasons.append(f"adv20={adv:.0f} (20日均量)")

    # --- 综合评分 ---
    score = row.get("score", 0)
    reasons.append(f"score={score:.6f} (综合排序分)")

    # --- 板块/行业 ---
    board = row.get("board", "")
    industry = row.get("industry", "")
    if board:
        reasons.append(f"board={board}")
    if industry and str(industry) != "nan":
        reasons.append(f"industry={industry}")

    return "; ".join(reasons)


# ============================================================
# 3. Markdown report writer (WORM)
# ============================================================
def _write_report(trade_date, ic_by_board, filtered, all_preds, bundles):
    """写入 Markdown 预测报告 (WORM: 每次运行独立文件, 不覆盖)."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(REPORT_DIR, f"prediction_report_{run_ts}.md")

    L = []
    L.append(f"# Pipeline-1 Prediction Report — {trade_date}")
    L.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Model tag: {TAG}")
    bundle_info = ", ".join(
        f"{b}={os.path.basename(p)}" for b, p in bundles.items()
    )
    L.append(f"Bundles: {bundle_info}")

    # ---- 1. IC Summary (signed) ----
    L.append("\n---\n## 1. Pipeline IC Summary (Signed Rank IC, OOS Test Segment)\n")
    L.append("IC > 0 = 模型有正向预测力; IC < 0 = 反向. "
             "Test segment = 最近 60 交易日 (排除当日).\n")
    L.append("| Board | IC_1d | IC_3d | IC_5d | Best IC |")
    L.append("|-------|-------|-------|-------|---------|")
    for board, ics in ic_by_board.items():
        best_key = max(ics, key=lambda k: ics[k])
        L.append(
            f"| {board} | {ics['1d_reg']:+.4f} | {ics['3d_reg']:+.4f} | "
            f"{ics['5d_reg']:+.4f} | {best_key}={ics[best_key]:+.4f} |"
        )

    # ---- 2. Stock Recommendation List ----
    L.append(f"\n---\n## 2. Stock Recommendation List ({len(filtered)} stocks)\n")
    L.append("Filters: `0.55 <= prob_up < 0.99`, `pred_ret_3d > 1%`, "
             "`pred_ret_5d > 1%`, `pain_prob < 0.35`, `ATR_pct < 6%`\n")
    display_cols = [
        "symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
        "pred_ret_5d", "prob_up", "pain_prob", "pred_q10", "pred_q50",
        "pred_q90", "uncertainty_width", "score", "ATR_pct",
    ]
    available = [c for c in display_cols if c in filtered.columns]
    if len(filtered):
        # markdown table
        header = "| " + " | ".join(available) + " |"
        sep = "| " + " | ".join("---" for _ in available) + " |"
        L.append(header)
        L.append(sep)
        for _, row in filtered.iterrows():
            vals = []
            for c in available:
                v = row.get(c, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            L.append("| " + " | ".join(vals) + " |")
    else:
        L.append("(无候选股通过筛选)")

    # ---- 3. Per-Stock Parameter Values & Reason Analysis ----
    L.append("\n---\n## 3. Per-Stock Parameter Values & Selection Reasons\n")
    for _, row in filtered.iterrows():
        sym = row.get("symbol", "?")
        board = row.get("board", "")
        industry = row.get("industry", "")
        L.append(f"### {sym} ({board} / {industry})\n")
        param_cols = [
            "pred_ret_1d", "pred_ret_3d", "pred_ret_5d", "prob_up",
            "pain_prob", "pred_q10", "pred_q50", "pred_q90",
            "uncertainty_width", "score", "ATR_pct", "adv20",
            "turnover_rate", "close", "amount",
            "composite_score", "rank_score",
        ]
        for c in param_cols:
            if c in row.index and not pd.isna(row[c]):
                v = row[c]
                if isinstance(v, float):
                    L.append(f"- **{c}**: {v:.6f}")
                else:
                    L.append(f"- **{c}**: {v}")
        L.append(f"\n**Reason**: {_build_reasons(row)}\n")

    # ---- 4. All Predictions Summary ----
    L.append(f"\n---\n## 4. All Predictions Summary ({len(all_preds)} stocks)\n")
    for board in all_preds["board"].unique() if "board" in all_preds.columns else []:
        sub = all_preds[all_preds["board"] == board]
        L.append(f"\n### Board: {board} ({len(sub)} stocks)\n")
        for c in ['pred_ret_1d', 'pred_ret_3d', 'pred_ret_5d', 'prob_up',
                   'pain_prob']:
            if c in sub.columns:
                v = sub[c].dropna()
                if len(v):
                    L.append(
                        f"- **{c}**: mean={v.mean():.6f}, "
                        f"std={v.std():.6f}, "
                        f"min={v.min():.6f}, max={v.max():.6f}"
                    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    logger.info("Prediction report saved: %s", path)
    return path


# ============================================================
# Main (existing logic + new IC/reason/report)
# ============================================================
def main():
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    trade_date = datetime.now().strftime("%Y%m%d")

    # 优先 enriched 面板 (3,227 stocks + alt data)
    for _p in ["data/panel_full_enriched_v3.parquet",
               "data/panel_full_enriched_v3.parquet"]:
        if os.path.exists(_p):
            panel = pd.read_parquet(_p)
            logger.info("加载面板: %s (%d stocks)", _p, panel["symbol"].nunique())
            break
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info(f"清洗: main={len(main_df)} dual={len(dual_df)} valve={valve}")

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)

    all_preds = []
    ic_by_board = {}
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue
        feats = features.build(df, cross_sectional_rank=(board != "main"))
        predictor = V35Predictor(bundles)

        # --- Signed IC on test segment ---
        bundle = predictor.bundles[board]
        ic = _compute_signed_ic(feats, bundle, board)
        ic_by_board[board] = ic
        logger.info(
            f"[{board}] Signed IC: 1d={ic['1d_reg']:+.4f} "
            f"3d={ic['3d_reg']:+.4f} 5d={ic['5d_reg']:+.4f}"
        )

        pred = predictor.predict(feats, board)
        pred["board"] = board

        # 带额外信息
        latest = feats.sort_values("date").groupby("symbol").tail(1)
        for col in ["ATR_pct", "adv20", "turnover_rate", "amount", "close"]:
            if col in latest.columns:
                pred[col] = latest.set_index("symbol").reindex(pred["symbol"])[col].values

        logger.info(f"{board}: {len(pred)} predictions")
        for c in ['pred_ret_1d','pred_ret_3d','pred_ret_5d','prob_up']:
            if c in pred.columns:
                v = pred[c].dropna()
                logger.info(f"  {c}: mean={v.mean():.6f} std={v.std():.6f} min={v.min():.6f} max={v.max():.6f}")

        all_preds.append(pred)

    preds = pd.concat(all_preds, ignore_index=True)

    # === 多维度筛选 ===
    # 高收益: pred_ret_3d > 1%, pred_ret_5d > 1%
    # 高概率: prob_up 0.55~0.95 (Platt 不会到1, 但仍排除极端)
    # 低风险: pain_prob < 0.35, ATR_pct < 0.06
    mask = (
        (preds["prob_up"] >= 0.55) &
        (preds["prob_up"] < 0.99) &
        (preds["pred_ret_3d"] > 0.01) &
        (preds["pred_ret_5d"] > 0.01) &
        (preds["pain_prob"].fillna(1) < 0.35)
    )
    if "ATR_pct" in preds.columns:
        mask &= (preds["ATR_pct"].fillna(0.1) < 0.06)

    filtered = preds[mask].copy()
    filtered["score"] = (
        filtered["prob_up"] * filtered["pred_ret_3d"] / (1 + filtered["pain_prob"].fillna(0.3))
    )
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    cols = ["symbol", "board", "industry", "pred_ret_1d", "pred_ret_3d",
            "pred_ret_5d", "prob_up", "pain_prob", "score", "ATR_pct"]

    print("\n" + "=" * 100)
    print(f"候选清单 (高收益+高概率+低风险): {len(filtered)} 只 / 总 {len(preds)} 只")
    print("条件: 0.55<=prob_up<0.99, pred_ret_3d>1%, pred_ret_5d>1%, pain_prob<0.35, ATR<6%")
    print("=" * 100)
    print(filtered[cols].to_string(index=False))

    filtered.to_csv("filtered_candidates.csv", index=False)
    preds.to_csv("predictions_3y.csv", index=False)
    print("\n候选清单已保存 filtered_candidates.csv, 全量 predictions_3y.csv")

    # === 写预测报告 ===
    report_path = _write_report(
        trade_date, ic_by_board, filtered, preds, bundles
    )
    print(f"\n预测报告已保存: {report_path}")


if __name__ == "__main__":
    main()
