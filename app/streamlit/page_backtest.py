# -*- coding: utf-8 -*-
"""
页面 3: 回测中心 (P10, V3.5 回测协议 + 参数调优)
=====================================================
- 参数表单 → BacktestEngineV35 回测 → 净值/回撤/指标/交易明细
- ParamTuner 网格调参: 选择 [TUNABLE] 参数 → 训练段选优 → OOS 复验 → 写回建议
- 无真实数据时使用合成演示面板 (显著标记)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol
from app.pipeline1.param_tuner import ParamTuner
from app.rules.config import TUNABLE_BOUNDS

from .components import drawdown_chart, equity_curve


def _demo_panel(days: int = 180, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-06-01", periods=days)
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


def render() -> None:
    st.header("回测中心 · V3.5 协议")
    st.caption(
        "成交价 T+1 open + 滑点0.05% | 佣金万2.5双边 + 印花税0.05% | "
        "等权1/15 单票≤10% 行业≤4 | 验收=扣费后净超额"
    )

    with st.sidebar:
        st.subheader("回测参数")
        top_n = st.number_input("Top N", 5, 20, 15)
        max_hold = st.slider("最大持仓天数", 2, 5, 3)
        hard_stop = st.slider("硬止损 %", -8.0, -2.0, -4.0, 0.5) / 100
        trailing = st.slider("移动止盈回撤 %", 2.0, 8.0, 4.0, 0.5) / 100
        prob_exit = st.slider("概率衰减退出", 0.40, 0.60, 0.50, 0.05)
        capital = st.number_input("初始资金", 100000, 10000000, 1000000, 100000)

    panel = _demo_panel()
    lists = _demo_lists(panel)
    st.info("演示面板 (3 股 × 180 日) — 生产接入全市场历史库")

    if st.button("▶ 执行回测", type="primary"):
        proto = BacktestProtocol(
            top_n=top_n,
            max_hold_days=max_hold,
            hard_stop=hard_stop,
            trailing_drawdown=trailing,
            prob_exit=prob_exit,
        )
        result = BacktestEngineV35(panel, proto).run(lists, initial_capital=capital)
        m = result["metrics"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("总收益", f"{m['total_return']:+.1%}")
        c2.metric("年化", f"{m['annual_return']:+.1%}")
        c3.metric("净超额(年化)", f"{m['net_excess_annual']:+.1%}")
        c4.metric("最大回撤", f"{m['max_drawdown']:.1%}")
        c5.metric("夏普", f"{m['sharpe']:.2f}")
        st.plotly_chart(equity_curve(result["nav_curve"]), use_container_width=True)
        st.plotly_chart(drawdown_chart(result["nav_curve"]), use_container_width=True)
        with st.expander("交易明细"):
            st.dataframe(result["trades"], use_container_width=True)

    # ---------- 参数调优 ----------
    st.subheader("参数调优 (ParamTuner)")
    tunable = st.multiselect(
        "调参目标 ([TUNABLE])",
        sorted(TUNABLE_BOUNDS),
        default=["max_hold_days", "prob_exit"],
    )
    if st.button("🔍 网格搜索 + OOS 复验"):
        if len(tunable) > 4:
            st.error("建议 ≤4 维 (控制组合数)")
        else:
            with st.spinner("网格搜索中..."):
                tuner = ParamTuner(panel, lists)
                report = tuner.grid_search(tunable, top_k=3)
            st.json(
                {
                    "best_params": report["best_params"],
                    "train_score": report["train_score"],
                    "oos_score": report["oos_score"],
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
