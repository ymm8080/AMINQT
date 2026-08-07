"""
页面: 预测评估中心 (预测中心 + 回测中心 合并)
=====================================================
一次预测 → 多角度 REVIEW:
  Tab 1 预测质量: IC / 候选清单 / 入选原因 (原 page_prediction)
  Tab 2 回测绩效: 净值 / 回撤 / 15 指标 (原 page_backtest)
  Tab 3 多模式对比: Squad vs Sniper / Jensen Alpha
  Tab 4 参数调优: 16 个 TUNABLE_BOUNDS 网格搜索
  Tab 5 报告管理: JSON/TXT/HTML 持久化

核心设计: 预测结果存入 session_state['pred_result'],
回测 Tab 直接复用同一份预测数据, 保证数据一致性.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol
from app.pipeline1.param_tuner import (
    TUNABLE_ENGINE,
    ParamTuner,
)
from app.rules.config import TUNABLE_BOUNDS

from . import data_service as ds
from .components import (
    comparison_nav_chart,
    drawdown_chart,
    equity_curve,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# 预测管道 (原 page_prediction._run_prediction)
# ══════════════════════════════════════════════════════════


def _load_panel(symbols: list[str] | None = None) -> pd.DataFrame | None:
    """加载 v3 面板, 可选按股票代码过滤."""
    path = "data/panel_full_enriched_v3.parquet"
    if not os.path.exists(path):
        return None
    panel = pd.read_parquet(path)
    if symbols:
        panel = panel[panel["symbol"].isin(symbols)]
    return panel


def _run_prediction(symbols: list[str] | None = None) -> dict | None:
    """运行预测管道, 返回结果 dict."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.predict_runner import resolve_current_bundles
    from app.pipeline1.predictor import V35Predictor

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
    bundles = resolve_current_bundles(model_dir="models/pipeline1")
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
        import predict_only

        bundle = predictor.bundles[board]
        ic = predict_only._compute_signed_ic(feats, bundle, board)
        ic_by_board[board] = ic

        pred = predictor.predict(feats, board)
        pred["board"] = board

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
    from datetime import datetime

    import predict_only

    trade_date = datetime.now().strftime("%Y%m%d")
    report_path = predict_only._write_report(
        trade_date, ic_by_board, filtered, preds, bundles
    )

    return {
        "ic": ic_by_board,
        "filtered": filtered,
        "all_preds": preds,
        "report_path": report_path,
        "panel": panel,
    }


# ══════════════════════════════════════════════════════════
# 演示面板 (回测 fallback)
# ══════════════════════════════════════════════════════════


def _demo_panel(days: int = 180, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = _dt.date.today()
    dates = pd.bdate_range(end=end, periods=days)
    frames = []
    for sym, ind in (("600519", "白酒"), ("601318", "保险"), ("600000", "银行")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, days))
        open_ = close * (1 + rng.normal(0, 0.003, days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                    "board": "main",
                    "industry": ind,
                    "amount": 1e9,
                    "volume": 1e7,
                    "score": rng.uniform(0, 1, days),
                    "prob_up": rng.uniform(0.4, 0.7, days),
                    "pred_ret_1d": rng.uniform(-0.02, 0.05, days),
                    "pred_ret_3d": rng.uniform(-0.03, 0.09, days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _demo_lists(panel: pd.DataFrame) -> dict:
    rng = np.random.default_rng(3)
    return {
        d: pd.DataFrame(
            {
                "symbol": g["symbol"].values,
                "score": rng.uniform(0, 1, len(g)),
                "prob_up": 0.60,
                "industry": g["industry"].values,
            }
        )
        for d, g in panel.groupby("date")
    }


# ══════════════════════════════════════════════════════════
# 数据加载 (回测用, 优先复用预测结果)
# ══════════════════════════════════════════════════════════


def _default_backtest_range() -> tuple[date, date]:
    """计算默认回测日期段: 最近 6 个完整月.

    例: 今天 7/13 → 1/1 ~ 6/30; 今天 7/31 → 1/1 ~ 6/30.
    """
    today = _dt.date.today()
    # 当月 1 号
    first_of_month = today.replace(day=1)
    # 上月最后一天 = 回测终止
    end = first_of_month - timedelta(days=1)
    # 往前推 6 个月: 从 end 所在月的 1 号起算
    start_month = end.month - 5
    start_year = end.year
    if start_month <= 0:
        start_month += 12
        start_year -= 1
    start = date(start_year, start_month, 1)
    return start, end


def _filter_panel_by_date(panel: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """按日期段过滤面板."""
    if panel is None or panel.empty or "date" not in panel.columns:
        return panel
    mask = (panel["date"].dt.date >= start) & (panel["date"].dt.date <= end)
    filtered = panel[mask].copy()
    return filtered if len(filtered) else panel


def _load_bt_panel_from_pred(
    pred_result: dict | None, start: date, end: date
) -> tuple[pd.DataFrame, dict, bool]:
    """从预测结果中提取回测面板和清单 (保证数据一致性), 按日期段过滤."""
    if pred_result is not None and "panel" in pred_result:
        panel = _filter_panel_by_date(pred_result["panel"], start, end)
        result = ds.panel_to_v35_lists(panel)
        if result is not None:
            return result[0], result[1], True
    # fallback: 演示数据
    days = max((end - start).days, 60)
    panel = _demo_panel(days=min(days, 750), seed=9)
    return panel, _demo_lists(panel), False


def _load_v52_from_pred(
    pred_result: dict | None, start: date, end: date
) -> dict | None:
    """从预测结果中提取 V5.2 数据包, 按日期段过滤."""
    if pred_result is not None and "panel" in pred_result:
        panel = _filter_panel_by_date(pred_result["panel"], start, end)
        result = ds.panel_to_v52_format(panel)
        if result is not None:
            pred_df, price_df, trade_dates, dv_hash = result
            from app.backtest.config_manager import ConfigManager

            config = ConfigManager.load("config/backtest_config.yaml")
            return {
                "pred_df": pred_df,
                "price_df": price_df,
                "trade_dates": trade_dates,
                "data_version_hash": dv_hash,
                "config": config,
            }
    # fallback: 演示数据
    days = max((end - start).days, 60)
    panel = _demo_panel(days=min(days, 750), seed=9)
    result = ds.panel_to_v52_format(panel)
    if result is None:
        return None
    pred_df, price_df, trade_dates, dv_hash = result
    from app.backtest.config_manager import BacktestConfig

    return {
        "pred_df": pred_df,
        "price_df": price_df,
        "trade_dates": trade_dates,
        "data_version_hash": dv_hash,
        "config": BacktestConfig(),
    }


def _get_benchmark_df(
    benchmark_sel: str, nav_curve: pd.DataFrame
) -> pd.DataFrame | None:
    if benchmark_sel == "无":
        return None
    try:
        dates = nav_curve["date"]
        rng = np.random.default_rng(42)
        n = len(dates)
        bench_nav = 1.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        return pd.DataFrame({"date": dates, "nav": bench_nav})
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# 指标展示
# ══════════════════════════════════════════════════════════


def _render_metrics(m: dict, engine_name: str) -> None:
    st.subheader(f"绩效指标 ({engine_name})")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总收益", f"{m.get('total_return', 0):+.1%}")
    c2.metric("年化", f"{m.get('annual_return', 0):+.1%}")
    c3.metric("净超额(年化)", f"{m.get('net_excess_annual', 0):+.1%}")
    c4.metric("最大回撤", f"{m.get('max_drawdown', 0):.1%}")
    c5.metric("夏普", f"{m.get('sharpe', m.get('sharpe_ratio', 0)):.2f}")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Sortino", f"{m.get('sortino', 0):.2f}")
    c7.metric("胜率", f"{m.get('win_rate', 0):.1%}")
    c8.metric("盈亏比", f"{m.get('pl_ratio', m.get('profit_loss_ratio', 0)):.2f}")
    c9.metric("期望收益", f"{m.get('expectancy', 0):+.2%}")
    c10.metric("最大连亏", f"{m.get('max_consecutive_loss', 0)} 笔")

    c11, c12, c13, c14, c15 = st.columns(5)
    c11.metric("交易笔数", f"{m.get('total_trades', m.get('num_trades', 0))}")
    c12.metric("平均持仓天数", f"{m.get('avg_holding_days', 0):.1f}")
    c13.metric("OOS Rank IC", f"{m.get('oos_rank_ic', 0):+.4f}")
    c14.metric("Calmar", f"{m.get('calmar_ratio', 0):.2f}")
    c15.metric("回撤持续期", f"{m.get('max_drawdown_duration', 0)} 天")


# ══════════════════════════════════════════════════════════
# 页面入口
# ══════════════════════════════════════════════════════════


def render() -> None:
    st.header("预测评估中心")
    st.caption("一次预测 → 多角度审查: 预测质量 (IC) + 回测绩效 (P&L) + 参数调优")

    # ---------- 侧边栏: 公共参数 ----------
    with st.sidebar:
        st.subheader("预测设置")
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

        st.divider()
        st.subheader("回测设置")
        engine_choice = st.selectbox(
            "回测引擎",
            ["V3.5 (轻量)", "V5.2 (完整风控)"],
            help="V3.5: 基础退出规则; V5.2: ATR止损/仓位模式/日保险丝",
        )
        # 回测日期段: 默认最近6个完整月
        default_start, default_end = _default_backtest_range()
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            bt_start = st.date_input("起始日期", value=default_start, key="bt_start")
        with col_d2:
            bt_end = st.date_input("终止日期", value=default_end, key="bt_end")
        if bt_start >= bt_end:
            st.warning("起始日期必须早于终止日期")
        use_v52 = "V5.2" in engine_choice

        top_n = st.number_input("Top N", 5, 20, 15)
        max_hold = st.slider("最大持仓天数", 2, 5, 3)
        hard_stop = st.slider("硬止损 %", -8.0, -2.0, -4.0, 0.5) / 100
        trailing = st.slider("移动止盈回撤 %", 2.0, 8.0, 4.0, 0.5) / 100
        prob_exit = st.slider("概率衰减退出", 0.40, 0.60, 0.50, 0.05)
        capital = st.number_input("初始资金", 100000, 10000000, 1000000, 100000)

        if use_v52:
            st.divider()
            st.subheader("V5.2 专属")
            position_mode = st.selectbox(
                "仓位模式",
                ["squad", "sniper", "sniper_max"],
                help="squad: 分散5只; sniper: 集中2只; sniper_max: 全仓1只",
            )
            horizon = st.selectbox("持有期", [1, 2, 4], index=1)

        st.divider()
        benchmark_sel = st.selectbox("基准对比", ["无", "沪深300", "中证1000"])

    # ---------- 运行预测 (公共入口) ----------
    col_run, col_info = st.columns([1, 3])
    with col_run:
        if st.button("▶ 运行预测", type="primary", key="btn_run_pred"):
            with st.spinner("预测运行中 (加载面板 → 清洗 → 特征 → 推理)..."):
                results = _run_prediction(symbols)
            if results:
                st.session_state["pred_result"] = results
                st.success(
                    f"预测完成: {len(results['all_preds'])} 只 → "
                    f"通过筛选 {len(results['filtered'])} 只"
                )
    with col_info:
        pred_result = st.session_state.get("pred_result")
        if pred_result:
            st.caption(
                f"✅ 已有预测结果: {len(pred_result['all_preds'])} 只预测, "
                f"{len(pred_result['filtered'])} 只候选 | "
                f"报告: {pred_result['report_path']}"
            )
        else:
            st.caption("⚠ 尚未运行预测 — 回测 Tab 将使用演示数据")

    # ---------- Tabs: 不同角度的 review ----------
    tab_pred, tab_bt, tab_compare, tab_tune, tab_report = st.tabs(
        [
            "预测质量 (IC)",
            "回测绩效 (P&L)",
            "多模式对比",
            "参数调优",
            "报告管理",
        ]
    )

    with tab_pred:
        _render_prediction_tab(pred_result)

    with tab_bt:
        _render_backtest_tab(
            engine_choice,
            bt_start,
            bt_end,
            use_v52,
            pred_result,
            top_n,
            max_hold,
            hard_stop,
            trailing,
            prob_exit,
            capital,
            position_mode if use_v52 else "squad",
            horizon if use_v52 else 2,
            benchmark_sel,
        )

    with tab_compare:
        _render_comparison_tab(pred_result, bt_start, bt_end, capital, benchmark_sel)

    with tab_tune:
        _render_tuning_tab(pred_result, bt_start, bt_end)

    with tab_report:
        _render_report_tab()


# ══════════════════════════════════════════════════════════
# Tab 1: 预测质量 (IC)
# ══════════════════════════════════════════════════════════


def _render_prediction_tab(pred_result: dict | None) -> None:
    st.subheader("Pipeline IC 摘要 + 候选清单")

    if pred_result is None:
        st.info("请先点击「运行预测」")
        return

    ic_by_board = pred_result["ic"]
    filtered = pred_result["filtered"]
    all_preds = pred_result["all_preds"]

    # IC Summary
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

    # 候选清单
    st.subheader(f"推荐清单 ({len(filtered)} 只 / 总 {len(all_preds)} 只)")
    st.caption(
        "条件: 0.55 ≤ prob_up < 0.99, pred_ret_3d > 1%, pred_ret_5d > 1%, "
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

        # 入选原因
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

    st.success(f"报告已保存: {pred_result['report_path']}")


# ══════════════════════════════════════════════════════════
# Tab 2: 回测绩效 (P&L)
# ══════════════════════════════════════════════════════════


def _render_backtest_tab(
    engine_choice,
    bt_start,
    bt_end,
    use_v52,
    pred_result,
    top_n,
    max_hold,
    hard_stop,
    trailing,
    prob_exit,
    capital,
    position_mode,
    horizon,
    benchmark_sel,
) -> None:
    engine_tag = "V5.2" if use_v52 else "V3.5"
    data_tag = "预测数据" if pred_result else "演示数据"
    st.info(f"引擎: {engine_tag} | 数据: {data_tag} | 日期: {bt_start} → {bt_end}")

    if st.button("▶ 执行回测", type="primary", key="btn_run_bt"):
        with st.spinner(f"回测运行中 ({engine_tag})..."):
            if use_v52:
                result = _run_v52_backtest(
                    pred_result,
                    bt_start,
                    bt_end,
                    position_mode,
                    horizon,
                    top_n,
                    max_hold,
                    hard_stop,
                    trailing,
                    prob_exit,
                    capital,
                )
            else:
                panel, lists, _ = _load_bt_panel_from_pred(
                    pred_result, bt_start, bt_end
                )
                result = _run_v35_backtest(
                    panel,
                    lists,
                    top_n,
                    max_hold,
                    hard_stop,
                    trailing,
                    prob_exit,
                    capital,
                )

        if result is None:
            st.error("回测失败")
            return

        st.session_state["bt_result"] = result
        st.session_state["bt_engine"] = engine_tag

        _render_metrics(result["metrics"], engine_tag)

        bench_df = _get_benchmark_df(benchmark_sel, result["nav_curve"])
        st.plotly_chart(
            equity_curve(
                result["nav_curve"],
                benchmark_df=bench_df,
                benchmark_name=benchmark_sel,
            ),
            use_container_width=True,
        )
        st.plotly_chart(drawdown_chart(result["nav_curve"]), use_container_width=True)

        with st.expander("交易明细"):
            trades = result.get("trades")
            if trades is not None and len(trades):
                st.dataframe(trades, use_container_width=True)
            else:
                st.info("无交易记录")

        if use_v52 and "holdings_history" in result:
            with st.expander("每日持仓快照"):
                st.dataframe(result["holdings_history"], use_container_width=True)


def _run_v35_backtest(
    panel, lists, top_n, max_hold, hard_stop, trailing, prob_exit, capital
):
    try:
        proto = BacktestProtocol(
            top_n=top_n,
            max_hold_days=max_hold,
            hard_stop=hard_stop,
            trailing_drawdown=trailing,
            prob_exit=prob_exit,
        )
        return BacktestEngineV35(panel, proto).run(lists, initial_capital=capital)
    except Exception as exc:
        logger.error("V35 回测失败: %s", exc, exc_info=True)
        st.error(f"V35 回测失败: {exc}")
        return None


def _run_v52_backtest(
    pred_result,
    bt_start,
    bt_end,
    position_mode,
    horizon,
    top_n,
    max_hold,
    hard_stop,
    trailing,
    prob_exit,
    capital,
):
    try:
        v52_data = _load_v52_from_pred(pred_result, bt_start, bt_end)
        if v52_data is None:
            st.error("V5.2 数据加载失败")
            return None

        from app.backtest.engine import BacktestEngine

        base_config = v52_data["config"]
        base_config.position_mode = position_mode
        base_config.holding_period = max_hold
        base_config.stop_loss_main = hard_stop
        base_config.trailing_stop_min_pct = trailing
        base_config.prob_threshold = prob_exit
        base_config.initial_capital = capital

        eng = BacktestEngine(
            config=base_config,
            pred_df=v52_data["pred_df"],
            price_df=v52_data["price_df"],
            trade_dates=v52_data["trade_dates"],
            data_version_hash=v52_data["data_version_hash"],
        )
        result_df = eng.run(horizon=horizon, topk=top_n)
        trades_df = eng.get_trades()
        holdings_df = eng.get_holdings_history()
        metrics = eng.get_metrics()

        nav_curve = result_df[["date", "nav"]].copy()
        if "cash" in result_df.columns:
            nav_curve["cash"] = result_df["cash"]
            nav_curve["n_positions"] = result_df.get("num_holdings", 0)

        return {
            "nav_curve": nav_curve,
            "trades": trades_df,
            "metrics": metrics,
            "holdings_history": holdings_df,
        }
    except Exception as exc:
        logger.error("V5.2 回测失败: %s", exc, exc_info=True)
        st.error(f"V5.2 回测失败: {exc}")
        return None


# ══════════════════════════════════════════════════════════
# Tab 3: 多模式对比
# ══════════════════════════════════════════════════════════


def _render_comparison_tab(
    pred_result, bt_start, bt_end, capital, benchmark_sel
) -> None:
    st.subheader("Squad vs Sniper 多模式对比")
    st.caption("运行两种仓位模式回测, 对比绩效差异 (集中度风险/Jensen Alpha)")

    if st.button("▶ 运行多模式对比", type="primary", key="btn_compare"):
        v52_data = _load_v52_from_pred(pred_result, bt_start, bt_end)
        if v52_data is None:
            st.error("数据加载失败")
            return

        from app.backtest.comparative_analyzer import ComparativeAnalyzer
        from app.backtest.config_manager import BacktestConfig
        from app.backtest.engine import BacktestEngine

        results = {}
        navs = {}
        for mode in ["squad", "sniper"]:
            with st.spinner(f"运行 {mode} 模式..."):
                config = BacktestConfig(
                    **{k: v for k, v in vars(v52_data["config"]).items()}
                )
                config.position_mode = mode
                config.initial_capital = capital
                eng = BacktestEngine(
                    config=config,
                    pred_df=v52_data["pred_df"],
                    price_df=v52_data["price_df"],
                    trade_dates=v52_data["trade_dates"],
                    data_version_hash=v52_data["data_version_hash"],
                )
                eng.run(horizon=2, topk=5)
                results[mode] = {
                    "nav": eng.run(horizon=2, topk=5),
                    "trades": eng.get_trades(),
                    "metrics": eng.get_metrics(),
                }
                navs[mode.upper()] = results[mode]["nav"][["date", "nav"]]

        if benchmark_sel != "无":
            bench_df = _get_benchmark_df(benchmark_sel, results["squad"]["nav"])
            if bench_df is not None:
                navs[benchmark_sel] = bench_df

        st.plotly_chart(comparison_nav_chart(navs), use_container_width=True)

        comp = ComparativeAnalyzer(
            squad_result=results["squad"]["nav"],
            sniper_result=results["sniper"]["nav"],
            squad_trades=results["squad"]["trades"],
            sniper_trades=results["sniper"]["trades"],
            squad_metrics=results["squad"]["metrics"],
            sniper_metrics=results["sniper"]["metrics"],
        )
        comparison = comp.generate_comparison_report()

        col1, col2, col3 = st.columns(3)
        col1.metric("集中度风险系数", f"{comparison['concentration_risk_ratio']:.2f}")
        col2.metric("Squad 夏普", f"{comparison['squad_sharpe']:.2f}")
        col3.metric("Sniper 夏普", f"{comparison['sniper_sharpe']:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Squad 总收益", f"{comparison['squad_total_return']:+.1%}")
        col5.metric("Sniper 总收益", f"{comparison['sniper_total_return']:+.1%}")
        rec_map = {
            "concentrated": "集中策略更优",
            "diversified": "分散策略更优",
            "tighten_stop": "建议收紧止损",
        }
        col6.metric(
            "建议",
            rec_map.get(comparison["recommendation"], comparison["recommendation"]),
        )

        with st.expander("Jensen Alpha"):
            st.metric("Squad Alpha", f"{comparison['jensen_alpha_squad']:+.2%}")
            st.metric("Sniper Alpha", f"{comparison['jensen_alpha_sniper']:+.2%}")


# ══════════════════════════════════════════════════════════
# Tab 4: 参数调优
# ══════════════════════════════════════════════════════════


def _render_tuning_tab(pred_result, bt_start, bt_end) -> None:
    st.subheader("参数调优 (ParamTuner)")
    st.caption("全部 16 个 TUNABLE_BOUNDS 参数, 支持 V35/V5.2 双引擎评估")

    tune_engine = st.selectbox(
        "调参引擎",
        ["V3.5", "V5.2"],
        help="V3.5 只支持 4 个参数; V5.2 支持 8 个参数",
    )
    use_v52_tune = tune_engine == "V5.2"

    obj_col, constr_col = st.columns(2)
    with obj_col:
        objective = st.selectbox(
            "目标函数",
            ["净超额(年化)", "夏普", "总收益", "Sortino", "Calmar"],
        )
    with constr_col:
        max_dd_limit = st.slider("约束: 最大回撤限制 %", -20.0, -2.0, -10.0, 1.0) / 100
        st.caption("OOS 复验时, 最大回撤超过此限制的组合会被过滤")

    if use_v52_tune:
        available_params = [
            name for name, engine in TUNABLE_ENGINE.items() if engine in ("both", "v52")
        ]
    else:
        available_params = [
            name for name, engine in TUNABLE_ENGINE.items() if engine == "both"
        ]
    rule_only_params = [
        name for name, engine in TUNABLE_ENGINE.items() if engine == "rule_only"
    ]

    tunable = st.multiselect(
        f"调参目标 (可用 {len(available_params)} 个, 引擎={tune_engine})",
        sorted(available_params),
        default=["max_hold_days", "prob_exit"]
        if not use_v52_tune
        else ["max_hold_days", "prob_exit", "surge_pct"],
    )

    if rule_only_params:
        with st.expander(f"仅规则引擎参数 ({len(rule_only_params)} 个)"):
            st.caption("以下参数属于日内规则引擎, 无法通过回测调优:")
            st.dataframe(
                pd.DataFrame({"参数名": rule_only_params}),
                use_container_width=True,
                hide_index=True,
            )

    ranges = {}
    if tunable:
        st.caption("自定义每个参数的搜索范围")
        cols = st.columns(min(len(tunable), 4))
        for i, name in enumerate(tunable):
            lo_default, hi_default, step_default = TUNABLE_BOUNDS[name]
            with cols[i % len(cols)]:
                st.markdown(f"**{name}**")
                lo = st.number_input(
                    "min", value=float(lo_default), key=f"range_lo_{name}"
                )
                hi = st.number_input(
                    "max", value=float(hi_default), key=f"range_hi_{name}"
                )
                step = st.number_input(
                    "step", value=float(step_default), key=f"range_step_{name}"
                )
                ranges[name] = (lo, hi, step)

    if st.button("🔍 网格搜索 + OOS 复验", key="btn_tune"):
        if len(tunable) > 4:
            st.error("建议 ≤4 维 (控制组合数)")
        elif not tunable:
            st.error("请至少选择一个调参目标")
        else:
            panel, lists, _ = _load_bt_panel_from_pred(pred_result, bt_start, bt_end)
            v52_data = None
            if use_v52_tune:
                v52_data = _load_v52_from_pred(pred_result, bt_start, bt_end)

            with st.spinner("网格搜索中..."):
                original_bounds = dict(TUNABLE_BOUNDS)
                for name, (lo, hi, step) in ranges.items():
                    TUNABLE_BOUNDS[name] = (lo, hi, step)
                try:
                    obj_map = {
                        "净超额(年化)": "net_excess_annual",
                        "夏普": "sharpe",
                        "总收益": "total_return",
                        "Sortino": "sortino",
                        "Calmar": "calmar",
                    }
                    tuner = ParamTuner(
                        panel=panel,
                        daily_lists=lists,
                        v52_data=v52_data,
                    )
                    report = tuner.grid_search(
                        tunable,
                        top_k=3,
                        objective=obj_map[objective],
                        max_dd_limit=max_dd_limit,
                        engine="v52" if use_v52_tune else "v35",
                    )
                finally:
                    TUNABLE_BOUNDS.clear()
                    TUNABLE_BOUNDS.update(original_bounds)

            st.json(
                {
                    "best_params": report["best_params"],
                    "train_score": report["train_score"],
                    "oos_score": report["oos_score"],
                    "engine": report.get("engine", "v35"),
                    "fallback_to_default": report["fallback_to_default"],
                }
            )
            st.dataframe(
                pd.DataFrame(
                    [{"params": p, "train_score": s} for p, s in report["leaderboard"]]
                ),
                use_container_width=True,
            )
            st.caption(f"报告: {report['report_path']} | OOS 不达标自动回退默认值")

            if not report["fallback_to_default"] and report["best_params"]:
                if st.button("✅ 应用调参结果到规则引擎 Config", type="primary"):
                    from app.rules.config import Config

                    cfg = Config()
                    ParamTuner.apply_to_config(report["best_params"], cfg)
                    st.success("已写入规则引擎 Config (内存级)")
                    st.caption("持久化需在配置中心 → 规则参数 → 保存")

                if use_v52_tune and v52_data:
                    if st.button("✅ 应用调参结果到 V5.2 BacktestConfig"):
                        ParamTuner.apply_to_v52_config(
                            report["best_params"], v52_data["config"]
                        )
                        st.success("已写入 V5.2 BacktestConfig (内存级)")


# ══════════════════════════════════════════════════════════
# Tab 5: 报告管理
# ══════════════════════════════════════════════════════════


def _render_report_tab() -> None:
    st.subheader("回测报告管理")
    st.caption("生成持久化报告 (JSON/TXT/HTML 三格式, 含审计哈希)")

    bt_result = st.session_state.get("bt_result")
    bt_engine = st.session_state.get("bt_engine", "V3.5")

    if bt_result is None:
        st.info("请先在「回测绩效」Tab 执行回测, 再生成报告")
        return

    mode_name = st.text_input("模式名称", value=bt_engine.lower(), key="report_mode")
    output_dir = st.text_input("输出目录", value="reports", key="report_dir")

    if st.button("📄 生成报告", type="primary", key="btn_report"):
        try:
            from app.backtest.config_manager import ConfigManager
            from app.backtest.report_generator import ReportGenerator

            config = ConfigManager.load("config/backtest_config.yaml")
            gen = ReportGenerator(config=config, output_dir=output_dir)

            result_df = bt_result.get("nav_curve", pd.DataFrame())
            trades_df = bt_result.get("trades", pd.DataFrame())
            metrics = bt_result.get("metrics", {})

            if "daily_pnl_pct" not in result_df.columns:
                result_df = result_df.copy()
                result_df["daily_pnl_pct"] = result_df["nav"].pct_change().fillna(0)

            basepath = gen.generate(
                mode_name=mode_name,
                result_df=result_df,
                trades_df=trades_df if trades_df is not None else pd.DataFrame(),
                metrics=metrics,
                data_version_hash="panel_v3",
            )
            st.success(f"报告已生成: {basepath}.{{json,txt,html}}")

            for ext in [".json", ".txt", ".html"]:
                p = basepath + ext
                if os.path.exists(p):
                    st.caption(f"✅ {ext[1:].upper()}: {p}")

            with st.expander("审计信息"):
                st.code(f"配置哈希: {gen.config_hash}")
                st.code(f"生成时间: {gen._timestamp}")

        except Exception as exc:
            st.error(f"报告生成失败: {exc}")
            logger.error("报告生成失败: %s", exc, exc_info=True)

    st.divider()
    st.subheader("历史报告")
    report_dir_path = output_dir
    if os.path.isdir(report_dir_path):
        reports = sorted(
            [f for f in os.listdir(report_dir_path) if f.endswith(".json")],
            reverse=True,
        )[:10]
        if reports:
            for fname in reports:
                col1, col2 = st.columns([3, 1])
                col1.caption(fname)
                fpath = os.path.join(report_dir_path, fname)
                if col2.button("查看", key=f"view_{fname}"):
                    import json as _json

                    try:
                        with open(fpath, encoding="utf-8") as fh:
                            data = _json.load(fh)
                        st.json(data)
                    except Exception as exc:
                        st.error(f"读取失败: {exc}")
        else:
            st.info("暂无历史报告")
    else:
        st.info(f"报告目录 {report_dir_path} 不存在")


if __name__ == "__main__":
    render()
