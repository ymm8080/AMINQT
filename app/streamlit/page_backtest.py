# -*- coding: utf-8 -*-
"""
页面 3: 回测中心 (P10, V3.5 + V5.2 双引擎回测 + 参数调优)
=====================================================
- 引擎选择: V3.5 (轻量) / V5.2 (完整风控: ATR止损/时间止损/仓位模式)
- 数据选择: 演示合成数据 / 真实 v3 面板 (panel_full_enriched_v3.parquet)
- 全指标展示: 总收益/年化/净超额/最大回撤/夏普/Sortino/胜率/盈亏比/OOS IC
- 基准对比: 净值曲线叠加基准 (沪深300/中证1000)
- 多模式对比: Squad vs Sniper → ComparativeAnalyzer (集中度/Jensen Alpha)
- 参数调优: 全部 16 个 TUNABLE_BOUNDS 参数 (V35/V5.2 双引擎)
- 持久化报告: ReportGenerator 生成 JSON/TXT/HTML 三格式
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol
from app.pipeline1.param_tuner import (
    CONFIG_TO_V52,
    ParamTuner,
    TUNABLE_ENGINE,
)
from app.rules.config import TUNABLE_BOUNDS

from . import data_service as ds
from .components import (
    comparison_nav_chart,
    drawdown_chart,
    equity_curve,
)

logger = logging.getLogger(__name__)


# ---------- 演示面板 ----------


def _demo_panel(days: int = 180, seed: int = 9) -> pd.DataFrame:
    """合成演示面板 (多股 × 多日)."""
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


# ---------- 目标函数 ----------


def _objective_value(metrics: dict, objective: str) -> float:
    """根据选择的目标函数从回测指标中提取评分."""
    if objective == "净超额(年化)":
        return metrics.get("net_excess_annual", 0.0)
    if objective == "夏普":
        return metrics.get("sharpe", metrics.get("sharpe_ratio", 0.0))
    if objective == "总收益":
        return metrics.get("total_return", 0.0)
    if objective == "Sortino":
        return metrics.get("sortino", 0.0)
    if objective == "Calmar":
        return metrics.get("calmar_ratio", 0.0)
    return metrics.get("net_excess_annual", 0.0)


# ---------- 数据加载 ----------


def _load_panel(data_source: str, window: str) -> tuple[pd.DataFrame, dict, bool]:
    """加载回测面板和清单.

    Returns:
        (panel, daily_lists, is_real_data).
    """
    days = 120 if window == "最近 6 个月" else 750
    if data_source == "真实面板 (v3 parquet)":
        raw = ds.load_backtest_panel(days=days)
        if raw is not None:
            result = ds.panel_to_v35_lists(raw)
            if result is not None:
                panel, lists = result
                return panel, lists, True
        st.warning("真实面板加载失败, 回退演示数据")
    # 演示数据
    seed = 9 if window == "最近 6 个月" else 10
    panel = _demo_panel(days=days, seed=seed)
    lists = _demo_lists(panel)
    return panel, lists, False


def _load_v52_data(data_source: str, window: str) -> dict | None:
    """加载 V5.2 引擎所需的数据包.

    Returns:
        {pred_df, price_df, trade_dates, data_version_hash, config} 或 None.
    """
    days = 120 if window == "最近 6 个月" else 750
    if data_source == "真实面板 (v3 parquet)":
        raw = ds.load_backtest_panel(days=days)
        if raw is not None:
            result = ds.panel_to_v52_format(raw)
            if result is not None:
                pred_df, price_df, trade_dates, dv_hash = result
                from app.backtest.config_manager import BacktestConfig, ConfigManager
                config = ConfigManager.load("config/backtest_config.yaml")
                return {
                    "pred_df": pred_df,
                    "price_df": price_df,
                    "trade_dates": trade_dates,
                    "data_version_hash": dv_hash,
                    "config": config,
                }
        st.warning("真实面板加载失败 (V5.2), 回退演示数据")
    # 演示数据 (V5.2 格式)
    panel = _demo_panel(
        days=120 if window == "最近 6 个月" else 750,
        seed=9 if window == "最近 6 个月" else 10,
    )
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


# ---------- 指标展示 ----------


def _render_metrics(m: dict, engine_name: str) -> None:
    """渲染完整指标卡片 (V35 + V5.2 全部指标)."""
    st.subheader(f"绩效指标 ({engine_name})")
    # 第一行: 基础指标
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总收益", f"{m.get('total_return', 0):+.1%}")
    c2.metric("年化", f"{m.get('annual_return', 0):+.1%}")
    c3.metric("净超额(年化)", f"{m.get('net_excess_annual', 0):+.1%}")
    c4.metric("最大回撤", f"{m.get('max_drawdown', 0):.1%}")
    c5.metric("夏普", f"{m.get('sharpe', m.get('sharpe_ratio', 0)):.2f}")

    # 第二行: 风险调整 + 赚钱指标
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Sortino", f"{m.get('sortino', 0):.2f}")
    c7.metric("胜率", f"{m.get('win_rate', 0):.1%}")
    c8.metric("盈亏比", f"{m.get('pl_ratio', m.get('profit_loss_ratio', 0)):.2f}")
    c9.metric("期望收益", f"{m.get('expectancy', 0):+.2%}")
    c10.metric("最大连亏", f"{m.get('max_consecutive_loss', 0)} 笔")

    # 第三行: 补充指标
    c11, c12, c13, c14, c15 = st.columns(5)
    c11.metric("交易笔数", f"{m.get('total_trades', m.get('num_trades', 0))}")
    c12.metric("平均持仓天数", f"{m.get('avg_holding_days', 0):.1f}")
    c13.metric("OOS Rank IC", f"{m.get('oos_rank_ic', 0):+.4f}")
    c14.metric("Calmar", f"{m.get('calmar_ratio', 0):.2f}")
    c15.metric("回撤持续期", f"{m.get('max_drawdown_duration', 0)} 天")


# ---------- 页面入口 ----------


def render() -> None:
    st.header("回测中心 · V3.5 + V5.2 双引擎")
    st.caption(
        "V3.8 协议 (分层滑点 + Sortino 主目标) | V5.2 (ATR止损 + 仓位模式 + 日保险丝) | "
        "扣费后净超额验收"
    )

    # ---------- 侧边栏: 回测参数 ----------
    with st.sidebar:
        st.subheader("引擎与数据")
        engine_choice = st.selectbox(
            "回测引擎",
            ["V3.5 (轻量)", "V5.2 (完整风控)"],
            help="V3.5: 基础退出规则; V5.2: ATR止损/时间止损/仓位模式/日保险丝",
        )
        data_choice = st.selectbox(
            "数据来源",
            ["演示数据", "真实面板 (v3 parquet)"],
        )
        window = st.selectbox("回测窗口", ["最近 6 个月", "过去三年"])

        use_v52 = "V5.2" in engine_choice

        st.divider()
        st.subheader("回测参数")
        top_n = st.number_input("Top N", 5, 20, 15)
        max_hold = st.slider("最大持仓天数", 2, 5, 3)
        hard_stop = st.slider("硬止损 %", -8.0, -2.0, -4.0, 0.5) / 100
        trailing = st.slider("移动止盈回撤 %", 2.0, 8.0, 4.0, 0.5) / 100
        prob_exit = st.slider("概率衰减退出", 0.40, 0.60, 0.50, 0.05)
        capital = st.number_input("初始资金", 100000, 10000000, 1000000, 100000)

        if use_v52:
            st.divider()
            st.subheader("V5.2 专属参数")
            position_mode = st.selectbox(
                "仓位模式", ["squad", "sniper", "sniper_max"],
                help="squad: 分散5只; sniper: 集中2只; sniper_max: 全仓1只",
            )
            horizon = st.selectbox("持有期 (交易日)", [1, 2, 4], index=1)

        st.divider()
        benchmark_sel = st.selectbox("基准对比", ["无", "沪深300", "中证1000"])

    # ---------- Tabs ----------
    tab_bt, tab_compare, tab_tune, tab_report = st.tabs(
        ["回测", "多模式对比", "参数调优", "报告管理"]
    )

    # ---------- Tab 1: 回测 ----------
    with tab_bt:
        _render_backtest_tab(
            engine_choice, data_choice, window, use_v52,
            top_n, max_hold, hard_stop, trailing, prob_exit, capital,
            position_mode if use_v52 else "squad",
            horizon if use_v52 else 2,
            benchmark_sel,
        )

    # ---------- Tab 2: 多模式对比 ----------
    with tab_compare:
        _render_comparison_tab(data_choice, window, capital, benchmark_sel)

    # ---------- Tab 3: 参数调优 ----------
    with tab_tune:
        _render_tuning_tab(data_choice, window)

    # ---------- Tab 4: 报告管理 ----------
    with tab_report:
        _render_report_tab()


# ---------- Tab 1: 回测 ----------


def _render_backtest_tab(
    engine_choice, data_choice, window, use_v52,
    top_n, max_hold, hard_stop, trailing, prob_exit, capital,
    position_mode, horizon, benchmark_sel,
) -> None:
    panel, lists, is_real = _load_panel(data_choice, window)
    data_tag = "真实数据" if is_real else "演示数据"
    engine_tag = "V5.2" if use_v52 else "V3.5"
    st.info(f"引擎: {engine_tag} | 数据: {data_tag} | 窗口: {window}")

    if st.button("▶ 执行回测", type="primary", key="btn_run_bt"):
        with st.spinner(f"回测运行中 ({engine_tag})..."):
            if use_v52:
                result = _run_v52_backtest(
                    data_choice, window, position_mode, horizon,
                    top_n, max_hold, hard_stop, trailing, prob_exit, capital,
                )
            else:
                result = _run_v35_backtest(
                    panel, lists, top_n, max_hold, hard_stop,
                    trailing, prob_exit, capital,
                )

        if result is None:
            st.error("回测失败, 请检查数据/参数")
            return

        # 存入 session_state 供报告 Tab 使用
        st.session_state["bt_result"] = result
        st.session_state["bt_engine"] = engine_tag

        m = result["metrics"]
        _render_metrics(m, engine_tag)

        # 基准数据
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


def _run_v35_backtest(panel, lists, top_n, max_hold, hard_stop, trailing, prob_exit, capital):
    """运行 V35 回测, 返回 {nav_curve, trades, metrics}."""
    try:
        proto = BacktestProtocol(
            top_n=top_n,
            max_hold_days=max_hold,
            hard_stop=hard_stop,
            trailing_drawdown=trailing,
            prob_exit=prob_exit,
        )
        eng = BacktestEngineV35(panel, proto)
        return eng.run(lists, initial_capital=capital)
    except Exception as exc:
        logger.error("V35 回测失败: %s", exc, exc_info=True)
        st.error(f"V35 回测失败: {exc}")
        return None


def _run_v52_backtest(data_choice, window, position_mode, horizon, top_n, max_hold, hard_stop, trailing, prob_exit, capital):
    """运行 V5.2 回测, 返回 {nav_curve, trades, metrics, holdings_history}."""
    try:
        v52_data = _load_v52_data(data_choice, window)
        if v52_data is None:
            st.error("V5.2 数据加载失败")
            return None

        from app.backtest.config_manager import BacktestConfig
        from app.backtest.engine import BacktestEngine

        base_config = v52_data["config"]
        # 覆盖用户可调参数
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

        # 转换 nav_curve 格式以与 V35 一致
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


def _get_benchmark_df(benchmark_sel: str, nav_curve: pd.DataFrame) -> pd.DataFrame | None:
    """获取基准净值 DataFrame (归一化)."""
    if benchmark_sel == "无":
        return None
    # 演示基准: 用面板数据的日期生成合成基准
    try:
        dates = nav_curve["date"]
        rng = np.random.default_rng(42)
        n = len(dates)
        bench_nav = 1.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
        return pd.DataFrame({"date": dates, "nav": bench_nav})
    except Exception:
        return None


# ---------- Tab 2: 多模式对比 ----------


def _render_comparison_tab(data_choice, window, capital, benchmark_sel) -> None:
    st.subheader("Squad vs Sniper 多模式对比")
    st.caption("运行两种仓位模式回测, 对比绩效差异 (集中度风险/Jensen Alpha)")

    if st.button("▶ 运行多模式对比", type="primary", key="btn_compare"):
        v52_data = _load_v52_data(data_choice, window)
        if v52_data is None:
            st.error("数据加载失败")
            return

        from app.backtest.config_manager import BacktestConfig
        from app.backtest.engine import BacktestEngine
        from app.backtest.comparative_analyzer import ComparativeAnalyzer

        results = {}
        navs = {}
        for mode in ["squad", "sniper"]:
            with st.spinner(f"运行 {mode} 模式..."):
                config = BacktestConfig(**{
                    k: v for k, v in vars(v52_data["config"]).items()
                })
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

        # 添加基准
        if benchmark_sel != "无":
            bench_df = _get_benchmark_df(benchmark_sel, results["squad"]["nav"])
            if bench_df is not None:
                navs[benchmark_sel] = bench_df

        # 对比净值曲线
        st.plotly_chart(comparison_nav_chart(navs), use_container_width=True)

        # 对比指标表
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
        col1.metric("集中度风险系数", f"{comparison['concentration_risk_ratio']:.2f}",
                     help="Sniper最大回撤/Squad最大回撤, >1表示Sniper风险更高")
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
        col6.metric("建议", rec_map.get(
            comparison["recommendation"], comparison["recommendation"]))

        with st.expander("Jensen Alpha"):
            st.metric("Squad Alpha", f"{comparison['jensen_alpha_squad']:+.2%}")
            st.metric("Sniper Alpha", f"{comparison['jensen_alpha_sniper']:+.2%}")

        st.session_state["comparison_result"] = comparison


# ---------- Tab 3: 参数调优 ----------


def _render_tuning_tab(data_choice, window) -> None:
    st.subheader("参数调优 (ParamTuner)")
    st.caption("全部 16 个 TUNABLE_BOUNDS 参数, 支持 V35/V5.2 双引擎评估")

    # 引擎选择
    tune_engine = st.selectbox(
        "调参引擎",
        ["V3.5", "V5.2"],
        help="V3.5 只支持 4 个参数; V5.2 支持 8 个参数",
    )
    use_v52_tune = tune_engine == "V5.2"

    # 目标函数与约束
    obj_col, constr_col = st.columns(2)
    with obj_col:
        objective = st.selectbox(
            "目标函数",
            ["净超额(年化)", "夏普", "总收益", "Sortino", "Calmar"],
        )
    with constr_col:
        max_dd_limit = st.slider("约束: 最大回撤限制 %", -20.0, -2.0, -10.0, 1.0) / 100
        st.caption("OOS 复验时, 最大回撤超过此限制的组合会被过滤")

    # 参数选择: 根据引擎过滤
    if use_v52_tune:
        available_params = [
            name for name, engine in TUNABLE_ENGINE.items()
            if engine in ("both", "v52")
        ]
    else:
        available_params = [
            name for name, engine in TUNABLE_ENGINE.items()
            if engine == "both"
        ]
    rule_only_params = [
        name for name, engine in TUNABLE_ENGINE.items()
        if engine == "rule_only"
    ]

    tunable = st.multiselect(
        f"调参目标 (可用 {len(available_params)} 个, 引擎={tune_engine})",
        sorted(available_params),
        default=["max_hold_days", "prob_exit"] if not use_v52_tune
        else ["max_hold_days", "prob_exit", "surge_pct"],
    )

    if rule_only_params:
        with st.expander(f"仅规则引擎参数 ({len(rule_only_params)} 个, 不可回测调参)"):
            st.caption("以下参数属于日内规则引擎, 无法通过回测调优, 需手工设置:")
            st.dataframe(
                pd.DataFrame({"参数名": rule_only_params}),
                use_container_width=True,
                hide_index=True,
            )

    # 自定义搜索范围
    ranges = {}
    if tunable:
        st.caption("自定义每个参数的搜索范围 (留空则使用默认边界)")
        cols = st.columns(min(len(tunable), 4))
        for i, name in enumerate(tunable):
            lo_default, hi_default, step_default = TUNABLE_BOUNDS[name]
            with cols[i % len(cols)]:
                st.markdown(f"**{name}**")
                lo = st.number_input("min", value=float(lo_default), key=f"range_lo_{name}")
                hi = st.number_input("max", value=float(hi_default), key=f"range_hi_{name}")
                step = st.number_input("step", value=float(step_default), key=f"range_step_{name}")
                ranges[name] = (lo, hi, step)

    if st.button("🔍 网格搜索 + OOS 复验", key="btn_tune"):
        if len(tunable) > 4:
            st.error("建议 ≤4 维 (控制组合数)")
        elif not tunable:
            st.error("请至少选择一个调参目标")
        else:
            # 加载数据
            panel, lists, _ = _load_panel(data_choice, window)

            # V5.2 数据 (如果需要)
            v52_data = None
            if use_v52_tune:
                v52_data = _load_v52_data(data_choice, window)

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

            st.json({
                "best_params": report["best_params"],
                "train_score": report["train_score"],
                "oos_score": report["oos_score"],
                "engine": report.get("engine", "v35"),
                "fallback_to_default": report["fallback_to_default"],
            })
            st.dataframe(
                pd.DataFrame([
                    {"params": p, "train_score": s}
                    for p, s in report["leaderboard"]
                ]),
                use_container_width=True,
            )
            st.caption(f"报告: {report['report_path']} | OOS 不达标自动回退默认值")

            # 一键应用按钮 (F3)
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


# ---------- Tab 4: 报告管理 ----------


def _render_report_tab() -> None:
    st.subheader("回测报告管理")
    st.caption("生成持久化报告 (JSON/TXT/HTML 三格式, 含审计哈希)")

    bt_result = st.session_state.get("bt_result")
    bt_engine = st.session_state.get("bt_engine", "V3.5")

    if bt_result is None:
        st.info("请先在「回测」Tab 执行回测, 再生成报告")
        return

    # 报告参数
    mode_name = st.text_input("模式名称", value=bt_engine.lower(), key="report_mode")
    output_dir = st.text_input("输出目录", value="reports", key="report_dir")

    if st.button("📄 生成报告", type="primary", key="btn_report"):
        try:
            from app.backtest.report_generator import ReportGenerator
            from app.backtest.config_manager import BacktestConfig, ConfigManager

            config = ConfigManager.load("config/backtest_config.yaml")
            gen = ReportGenerator(config=config, output_dir=output_dir)

            # 适配数据格式
            result_df = bt_result.get("nav_curve", pd.DataFrame())
            trades_df = bt_result.get("trades", pd.DataFrame())
            metrics = bt_result.get("metrics", {})

            # 如果是 V35 格式, 转换 nav_curve 为 result_df 格式
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

            # 展示报告路径
            for ext in [".json", ".txt", ".html"]:
                p = basepath + ext
                if os.path.exists(p):
                    st.caption(f"✅ {ext[1:].upper()}: {p}")

            # 审计信息
            with st.expander("审计信息"):
                st.code(f"配置哈希: {gen.config_hash}")
                st.code(f"生成时间: {gen._timestamp}")

        except Exception as exc:
            st.error(f"报告生成失败: {exc}")
            logger.error("报告生成失败: %s", exc, exc_info=True)

    # 历史报告浏览
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
