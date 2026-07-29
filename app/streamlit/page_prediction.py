"""
页面: 预测中心 (Pipeline-1)
=====================================
支持全量预测和指定股票预测, 展示 IC 摘要 + 推荐清单 + 入选原因.
复用 predict_only.py 的核心逻辑 (清洗→特征→推理→IC→筛选→报告).
"""

from __future__ import annotations

import os
import logging
import pandas as pd
import streamlit as st
from datetime import datetime

logger = logging.getLogger(__name__)

TAG = "2026W31_3y"


def _load_panel(symbols: list[str] | None = None) -> pd.DataFrame | None:
    """加载面板, 可选按股票代码过滤."""
    for path in (
        "data/panel_full_enriched_v3.parquet",
        "data/panel_full_enriched_v3.parquet",
    ):
        if os.path.exists(path):
            panel = pd.read_parquet(path)
            if symbols:
                panel = panel[panel["symbol"].isin(symbols)]
            return panel
    return None


def _run_prediction(symbols: list[str] | None = None) -> dict | None:
    """运行预测, 返回结果 dict (ic, filtered, all_preds, report_path)."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor
    import predict_only  # 复用 predict_only.py 的辅助函数

    panel = _load_panel(symbols)
    if panel is None or len(panel) == 0:
        st.error("面板未找到或指定股票不在面板中")
        return None

    st.info(f"面板: {panel['symbol'].nunique()} 只股票, {len(panel)} 行")

    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    st.info(f"清洗: main={len(main_df)} dual={len(dual_df)} valve={valve}")
    if valve == "empty":
        st.warning("流动性安全阀触发, 无候选")
        return None

    features = FeatureEngineV35()
    bundles = find_bundles(model_dir="models/pipeline1", tag=TAG)
    if not bundles:
        st.error("无可用模型包, 请先训练")
        return None

    all_preds = []
    ic_by_board = {}
    for board, df in [("main", main_df), ("dual", dual_df)]:
        if len(df) == 0 or board not in bundles:
            continue
        feats = features.build(df, cross_sectional_rank=(board != "main"))
        predictor = V35Predictor(bundles)

        # Signed IC
        bundle = predictor.bundles[board]
        ic = predict_only._compute_signed_ic(feats, bundle, board)
        ic_by_board[board] = ic

        pred = predictor.predict(feats, board)
        pred["board"] = board

        # 带额外信息
        latest = feats.sort_values("date").groupby("symbol").tail(1)
        for col in ["ATR_pct", "adv20", "turnover_rate", "amount", "close"]:
            if col in latest.columns:
                pred[col] = (
                    latest.set_index("symbol").reindex(pred["symbol"])[col].values
                )

        all_preds.append(pred)

    if not all_preds:
        st.warning("无预测结果")
        return None

    preds = pd.concat(all_preds, ignore_index=True)

    # 筛选
    mask = (
        (preds["prob_up"] >= 0.55)
        & (preds["prob_up"] < 0.99)
        & (preds["pred_ret_3d"] > 0.01)
        & (preds["pred_ret_5d"] > 0.01)
        & (preds["pain_prob"].fillna(1) < 0.35)
    )
    if "ATR_pct" in preds.columns:
        mask &= preds["ATR_pct"].fillna(0.1) < 0.06

    filtered = preds[mask].copy()
    filtered["score"] = (
        filtered["prob_up"]
        * filtered["pred_ret_3d"]
        / (1 + filtered["pain_prob"].fillna(0.3))
    )
    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    # 写报告
    trade_date = datetime.now().strftime("%Y%m%d")
    report_path = predict_only._write_report(
        trade_date, ic_by_board, filtered, preds, bundles
    )

    return {
        "ic": ic_by_board,
        "filtered": filtered,
        "all_preds": preds,
        "report_path": report_path,
    }


def _display_results(results: dict) -> None:
    """展示预测结果: IC + 推荐清单 + 入选原因."""
    ic_by_board = results["ic"]
    filtered = results["filtered"]
    all_preds = results["all_preds"]
    report_path = results["report_path"]

    # ---- IC Summary ----
    st.subheader("Pipeline IC 摘要 (Signed Rank IC)")
    st.caption("IC > 0 = 模型方向正确 (可信); IC < 0 = 反向 (不可信)")
    ic_rows = []
    for board, ics in ic_by_board.items():
        best_key = max(ics, key=lambda k: ics[k])
        ic_rows.append(
            {
                "Board": board,
                "IC_1d": f"{ics['1d_reg']:+.4f}",
                "IC_3d": f"{ics['3d_reg']:+.4f}",
                "IC_5d": f"{ics['5d_reg']:+.4f}",
                "Best": f"{best_key}={ics[best_key]:+.4f}",
            }
        )
    st.dataframe(pd.DataFrame(ic_rows), use_container_width=True, hide_index=True)

    # ---- Filtered candidates ----
    st.subheader(f"推荐清单 ({len(filtered)} 只 / 总 {len(all_preds)} 只)")
    st.caption(
        "条件: 0.55 <= prob_up < 0.99, pred_ret_3d > 1%, pred_ret_5d > 1%, "
        "pain_prob < 0.35, ATR < 6%"
    )
    if len(filtered):
        display_cols = [
            c
            for c in [
                "symbol",
                "board",
                "industry",
                "pred_ret_1d",
                "pred_ret_3d",
                "pred_ret_5d",
                "prob_up",
                "pain_prob",
                "pred_q10",
                "pred_q50",
                "pred_q90",
                "uncertainty_width",
                "score",
                "ATR_pct",
            ]
            if c in filtered.columns
        ]
        st.dataframe(
            filtered[display_cols].style.format(
                {
                    c: "{:.4f}"
                    for c in display_cols
                    if c not in ("symbol", "board", "industry")
                }
            ),
            use_container_width=True,
            height=min(400, 35 * len(filtered) + 40),
        )

        # ---- Per-stock reasons ----
        st.subheader("入选原因分析")
        import predict_only

        for _, row in filtered.iterrows():
            with st.expander(
                f"{row['symbol']} ({row.get('board', '')} / "
                f"{row.get('industry', '')}) — score={row.get('score', 0):.4f}"
            ):
                param_cols = [
                    "pred_ret_1d",
                    "pred_ret_3d",
                    "pred_ret_5d",
                    "prob_up",
                    "pain_prob",
                    "pred_q10",
                    "pred_q50",
                    "pred_q90",
                    "uncertainty_width",
                    "score",
                    "ATR_pct",
                    "adv20",
                    "turnover_rate",
                    "close",
                    "amount",
                    "composite_score",
                    "rank_score",
                ]
                param_data = []
                for c in param_cols:
                    if c in row.index and not pd.isna(row[c]):
                        v = row[c]
                        param_data.append(
                            {
                                "参数": c,
                                "值": f"{v:.6f}" if isinstance(v, float) else v,
                            }
                        )
                if param_data:
                    st.dataframe(
                        pd.DataFrame(param_data),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.markdown(f"**Reason**: {predict_only._build_reasons(row)}")
    else:
        st.warning("无股票通过筛选条件")

    st.success(f"报告已保存: {report_path}")


def render() -> None:
    """页面入口."""
    st.header("预测中心 · Pipeline-1 (V3.5)")
    st.caption("LightGBM 双轨模型 · Signed IC · 实时推理")

    mode = st.radio(
        "预测模式",
        ["全量预测", "指定股票"],
        horizontal=True,
        key="pred_mode",
    )

    symbols = None
    if mode == "指定股票":
        user_input = st.text_input(
            "股票代码 (逗号分隔)",
            placeholder="如 600519,000001,300750",
            key="pred_symbols",
        )
        if user_input:
            symbols = [s.strip() for s in user_input.split(",") if s.strip()]
            if not symbols:
                st.warning("请输入至少一个股票代码")

    col_run, col_info = st.columns([1, 2])
    with col_run:
        if st.button("运行预测", type="primary", key="btn_run_pred"):
            with st.spinner("预测运行中 (加载面板 → 清洗 → 特征 → 推理)..."):
                results = _run_prediction(symbols)
            if results:
                _display_results(results)
    with col_info:
        st.caption("预测完成后自动生成 Markdown 报告 (含 IC + 清单 + 入选原因)")


if __name__ == "__main__":
    render()
