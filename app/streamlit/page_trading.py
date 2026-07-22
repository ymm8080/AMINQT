# -*- coding: utf-8 -*-
"""
页面 2: 交易看板 (P10) — 同花顺风格三栏
============================================
⚠ Pipeline-2 (5分钟模型) 设计未定稿 — 本页为演示/框架模式:
  左栏: 行情快照 | 中栏: 交易状态机 + 信号列表 | 右栏: 持仓/委托/成交
信号源: RuleEngine v2 + CompositeFeed (模拟 tick); 执行: SimExecutor.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data_service as ds
from .components import intraday_chart


def _state_machine_status() -> dict:
    """交易状态机状态 (复用 services/trading_state_machine, 失败则演示)."""
    try:
        from services.trading_state_machine import TradingStateMachine
        sm = TradingStateMachine()
        return {"auto_buy": False, "auto_sell": False, "paused": False,
                "mode": "MANUAL", "_sm": sm}
    except Exception:
        return {"auto_buy": False, "auto_sell": False, "paused": False,
                "mode": "MANUAL", "_sm": None}


def render() -> None:
    st.header("交易看板 · Pipeline 2")
    st.warning("⚠ Pipeline-2 (5分钟模型) 设计未定稿 — 本页为框架演示模式 (SimExecutor, 不真实下单)")

    left, mid, right = st.columns([3, 4, 3])

    # ---------- 左栏: 行情 ----------
    with left:
        st.subheader("行情")
        symbol = st.selectbox("标的", ds.DEMO_SYMBOLS,
                              format_func=lambda s: f"{s} {ds.DEMO_NAMES.get(s, '')}")
        df = ds.demo_intraday(symbol)
        st.metric("最新价", f"{df['price'].iloc[-1]:.2f}",
                  f"{(df['price'].iloc[-1] / df['price'].iloc[0] - 1):+.2%}")
        st.plotly_chart(intraday_chart(df, prev_close=df["price"].iloc[0]),
                        use_container_width=True)
        st.caption("五档盘口: 待 miniQMT xtdata.get_quote 接入")

    # ---------- 中栏: 状态机 + 信号 ----------
    with mid:
        st.subheader("交易控制")
        status = _state_machine_status()
        c1, c2, c3 = st.columns(3)
        c1.button("启动自动买入", key="ab_on", disabled=True)
        c2.button("启动自动卖出", key="as_on", disabled=True)
        c3.button("暂停全部", key="pause", disabled=True)
        st.caption(f"模式: {status['mode']} (自动开关在 P2 定稿后启用)")
        st.subheader("信号列表 (演示)")
        signals = pd.DataFrame({
            "时间": ["09:44", "10:12", "13:05"],
            "代码": ["600519", "300750", "601318"],
            "方向": ["BUY", "SELL_HALF", "WARN"],
            "优先级": ["L4-形态", "P7", "P10"],
            "原因": ["下探后低峰确认回升", "涨7%+高换手减半", "浮盈≥20%人工复核"],
        })
        st.dataframe(signals, use_container_width=True)
        st.button("批量确认 (手动模式)", disabled=True)

    # ---------- 右栏: 持仓/委托/成交 ----------
    with right:
        st.subheader("持仓")
        st.dataframe(pd.DataFrame({
            "代码": ["600519"], "数量": [100], "可用": [0],
            "成本": [99.40], "现价": [101.20], "盈亏%": ["+1.8"],
            "T+1": ["不可卖"]}), use_container_width=True)
        st.subheader("委托队列")
        st.dataframe(pd.DataFrame({
            "时间": ["09:44"], "代码": ["600519"], "方向": ["买"],
            "价": [99.40], "量": [100], "状态": ["已成交"]}),
            use_container_width=True)
        st.subheader("账户")
        c1, c2 = st.columns(2)
        c1.metric("总资产", "1,018,000")
        c2.metric("可用资金", "817,400")
