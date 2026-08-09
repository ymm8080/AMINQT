"""
页面 2: 交易看板 (P10) — 同花顺风格三栏
============================================
已接入 services.trading_state_machine + services.order_manager:
- 自动买/卖独立开关
- 手动模式信号确认 / 自动模式直接报单
- 委托队列、成交回报、持仓列表 (SimExecutor 演示)

变更点 (看板.docx):
- 交易控制按钮移至顶栏, 尺寸紧凑
- 行情图放大并在下方显示成交量
- 个股日曲线支持跳转查看
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import streamlit as st

from app.streamlit import data_service as ds
from app.streamlit.components import intraday_chart, kline_chart
from app.streamlit.components.trading_panel import (
    render_order_queue,
    render_position_list,
    render_signal_list,
    render_trade_list,
    render_trading_control,
)
from services.order_manager import OrderManager
from services.trading_state_machine import TradingStateMachine

# Streamlit 会话级单例: 状态机 + 委托管理器


def _get_state_machine() -> TradingStateMachine:
    if "trading_sm" not in st.session_state:
        st.session_state.trading_sm = TradingStateMachine()
    return st.session_state.trading_sm


def _get_order_manager() -> OrderManager:
    if "order_manager" not in st.session_state:
        st.session_state.order_manager = OrderManager()
    return st.session_state.order_manager


# 演示数据


def _demo_signals() -> list[dict]:
    """演示信号 (SimExecutor 模式)."""
    return [
        {
            "time": "09:44",
            "symbol": "600519",
            "side": "buy",
            "priority": "L4-形态",
            "reason": "下探后低峰确认回升",
            "price": 1680.0,
            "qty": 100,
        },
        {
            "time": "10:12",
            "symbol": "300750",
            "side": "sell",
            "priority": "P7",
            "reason": "涨7%+高换手减半",
            "price": 210.0,
            "qty": 200,
        },
        {
            "time": "13:05",
            "symbol": "601318",
            "side": "sell",
            "priority": "P10",
            "reason": "浮盈≥20%人工复核",
            "price": 48.0,
            "qty": 500,
        },
    ]


def _demo_positions() -> list[dict]:
    """演示持仓."""
    return [
        {
            "symbol": "600519",
            "qty": 100,
            "available_qty": 0,
            "cost": 99.40,
            "current_price": 101.20,
        },
        {
            "symbol": "601318",
            "qty": 300,
            "available_qty": 300,
            "cost": 47.50,
            "current_price": 48.10,
        },
    ]


def _account_snapshot() -> dict:
    """账户快照 (演示; 生产接入 Executor.get_account)."""
    return {"total_asset": 1018000.0, "available_cash": 817400.0, "frozen": 0.0}


# ---------- 页面入口 ----------


def render() -> None:
    st.header("交易看板 · Pipeline 2")
    st.caption("状态机 + 委托管理器已接入 (SimExecutor 演示模式, 不真实下单)")

    sm = _get_state_machine()
    om = _get_order_manager()

    # 顶栏: 交易控制按钮紧凑排布
    with st.container():
        st.subheader("交易控制")
        render_trading_control(sm, drawdown_pct=0.0)

    st.divider()

    # 三栏主体
    left, mid, right = st.columns([3, 4, 3])

    # ---------- 左栏: 行情 + 成交量 ----------
    with left:
        st.subheader("行情")
        default_symbol = st.session_state.get("trading_symbol", ds.DEMO_SYMBOLS[0])
        default_index = (
            ds.DEMO_SYMBOLS.index(default_symbol)
            if default_symbol in ds.DEMO_SYMBOLS
            else 0
        )
        symbol = st.selectbox(
            "标的",
            ds.DEMO_SYMBOLS,
            index=default_index,
            format_func=lambda s: f"{s} {ds.DEMO_NAMES.get(s, '')}",
        )
        # 消费掉跳转标记, 避免刷新后仍锁定
        st.session_state.pop("trading_symbol", None)
        ohlc = ds.demo_ohlc(symbol)
        last_price = float(ohlc["close"].iloc[-1])
        first_price = float(ohlc["close"].iloc[0])
        st.metric(
            "最新价", f"{last_price:.2f}", f"{(last_price / first_price - 1):+.2%}"
        )

        # 日K线图 (放大 + 成交量)
        st.plotly_chart(
            kline_chart(ohlc, ma_list=(5, 10, 20), title=f"{symbol} 日K"),
            use_container_width=True,
        )

        # 分时图 (带成交量/VWAP)
        df = ds.demo_intraday(symbol)
        st.plotly_chart(
            intraday_chart(df, prev_close=first_price, title=f"{symbol} 分时"),
            use_container_width=True,
        )
        st.caption("五档盘口: 待 miniQMT xtdata.get_quote 接入")

        st.subheader("板块涨跌幅")
        sector_df = ds.demo_sector_changes()
        st.dataframe(
            sector_df,
            column_config={
                "板块": st.column_config.TextColumn("板块"),
                "涨跌幅": st.column_config.NumberColumn(
                    "涨跌幅",
                    format="+.2%%",
                ),
                "上涨家数": st.column_config.NumberColumn("上涨家数"),
                "下跌家数": st.column_config.NumberColumn("下跌家数"),
            },
            column_order=["板块", "涨跌幅", "上涨家数", "下跌家数"],
            hide_index=True,
            use_container_width=True,
        )

    # ---------- 中栏: 信号 ----------
    with mid:
        st.subheader("信号列表")
        signals = ds.load_real_signals()
        if not signals:
            signals = _demo_signals()
            st.caption(
                "⚠ 使用演示信号 — 真实信号需先运行 Pipeline-1 选股并标记 priority"
            )
        render_signal_list(sm, om, signals)

    # ---------- 右栏: 持仓/委托/成交 ----------
    with right:
        st.subheader("账户")
        acct = ds.load_real_account()
        c1, c2 = st.columns(2)
        c1.metric("总资产", f"{acct['total_asset']:,.0f}")
        c2.metric("可用资金", f"{acct['available_cash']:,.0f}")

        st.subheader("持仓")
        positions = ds.load_real_positions()
        if not positions:
            positions = _demo_positions()
            st.caption("⚠ 使用演示持仓 — 真实持仓需先标记 priority 股票")
        render_position_list(positions)

        st.subheader("委托队列")
        render_order_queue(om)

        st.subheader("成交回报")
        render_trade_list(om)

    # ---------- 底栏: 审计日志 ----------
    st.divider()
    st.subheader("交易审计日志")
    audit_entries = ds.load_audit_log()
    if audit_entries:
        log_df = pd.DataFrame(audit_entries)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        # 记录页面加载
        ds.append_audit_log(
            {
                "时间": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=8))).strftime(
                    "%H:%M:%S"
                ),
                "操作": "页面加载",
                "代码": "-",
                "方向": "-",
                "价格": "-",
                "数量": "-",
                "结果": "OK",
                "备注": "交易看板已接入状态机/委托管理器/真实信号",
            }
        )
        st.info("审计日志已初始化 — 后续操作将自动记录")


if __name__ == "__main__":
    render()
