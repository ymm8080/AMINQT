"""
交易看板组件 (P10) — 同花顺风格三栏布局助手
================================================
与 services.trading_state_machine + services.order_manager 联动,
提供交易控制、信号确认、委托队列、持仓/成交回报的渲染函数。

注: 本模块函数调用 streamlit API; 纯逻辑已下沉到状态机/委托管理器,
因此单元测试主要覆盖状态机与委托管理器, 页面渲染通过 streamlit 启动冒烟测试。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services.order_manager import OrderManager
from services.trading_state_machine import TradingState, TradingStateMachine


def render_trading_control(sm: TradingStateMachine, drawdown_pct: float = 0.0) -> None:
    """渲染交易模式控制面板 (顶栏紧凑排布)."""
    col_state, col_buy, col_sell, col_global = st.columns([1, 1, 1, 2])

    with col_state:
        state_label = {
            TradingState.RUNNING: "🟢 运行中",
            TradingState.PAUSED: "🟡 已暂停",
            TradingState.STOPPED: "🔴 已停止",
        }[sm.state]
        mode = "手动"
        if sm.state is TradingState.RUNNING:
            if sm.auto_buy_enabled and sm.auto_sell_enabled:
                mode = "全自动"
            elif sm.auto_buy_enabled:
                mode = "仅自动买入"
            elif sm.auto_sell_enabled:
                mode = "仅自动卖出"
        st.metric("状态", f"{state_label} ({mode})")

    with col_buy:
        if sm.auto_buy_enabled:
            if st.button(
                "⏹ 停止买入",
                key="stop_auto_buy",
                type="primary",
                use_container_width=True,
            ):
                sm.disable_auto_buy()
                st.rerun()
        else:
            if st.button("▶️ 启动买入", key="start_auto_buy", use_container_width=True):
                sm.start()
                sm.enable_auto_buy()
                st.rerun()

    with col_sell:
        if sm.auto_sell_enabled:
            if st.button(
                "⏹ 停止卖出",
                key="stop_auto_sell",
                type="primary",
                use_container_width=True,
            ):
                sm.disable_auto_sell()
                st.rerun()
        else:
            if st.button("▶️ 启动卖出", key="start_auto_sell", use_container_width=True):
                sm.start()
                sm.enable_auto_sell()
                st.rerun()

    with col_global:
        c1, c2, c3 = st.columns(3)
        if c1.button("⏸ 暂停", key="pause_all", use_container_width=True):
            sm.pause()
            st.rerun()
        if c2.button("⏯ 恢复", key="resume_all", use_container_width=True):
            sm.resume()
            st.rerun()
        if c3.button(
            "⏹ 停止", key="stop_all", type="primary", use_container_width=True
        ):
            sm.stop_all()
            st.rerun()

    if drawdown_pct >= 3.0:
        st.error(f"⚠ 账户回撤 {drawdown_pct:.2f}% ≥ 3%, 已触发风控强制停止")
        sm.force_stop(f"回撤超限 {drawdown_pct:.2f}%")


def _signal_action_label(side: str) -> str:
    return {"buy": "买入", "sell": "卖出"}.get(side, side)


def render_signal_list(
    sm: TradingStateMachine,
    om: OrderManager,
    signals: list[dict],
) -> None:
    """渲染信号列表并处理手动/自动确认.

    Args:
        sm: 交易状态机。
        om: 委托管理器。
        signals: 信号列表, 每项至少含 {time, symbol, side, priority, reason, price, qty}。
    """
    if not signals:
        st.info("暂无信号")
        return

    df = pd.DataFrame(signals)
    df["方向"] = df["side"].map(_signal_action_label)
    display_cols = [
        c
        for c in ["time", "symbol", "方向", "price", "qty", "priority", "reason"]
        if c in df.columns
    ]
    st.dataframe(df[display_cols], use_container_width=True)

    # 自动模式: 为每个信号直接提交委托 (require_confirm=False)
    auto_executed = []
    if sm.state is TradingState.RUNNING:
        for sig in signals:
            if sm.should_auto_execute(sig["side"]):
                oid = om.submit(
                    symbol=sig["symbol"],
                    side=sig["side"],
                    price=sig.get("price", 0.0),
                    qty=sig.get("qty", 0),
                    require_confirm=False,
                )
                auto_executed.append(oid)
    if auto_executed:
        st.success(f"自动执行 {len(auto_executed)} 条信号")
        st.rerun()

    # 手动模式: 提供逐条/批量确认
    if sm.state in (TradingState.RUNNING, TradingState.STOPPED):
        pending_ids: dict[str, str] = {}
        cols = st.columns(min(len(signals), 4))
        for idx, sig in enumerate(signals):
            side = sig["side"]
            with cols[idx % len(cols)]:
                label = f"{_signal_action_label(side)} {sig['symbol']}"
                if st.button(label, key=f"confirm_sig_{idx}"):
                    oid = om.submit(
                        symbol=sig["symbol"],
                        side=side,
                        price=sig.get("price", 0.0),
                        qty=sig.get("qty", 0),
                        require_confirm=True,
                    )
                    pending_ids[oid] = label
        if pending_ids:
            confirmed = sum(1 for oid in pending_ids if om.manual_confirm(oid))
            st.success(f"已确认 {confirmed}/{len(pending_ids)} 笔委托")
            st.rerun()


def render_order_queue(om: OrderManager) -> None:
    """渲染委托队列 + 撤单按钮."""
    orders = om.get_pending()
    if not orders:
        st.info("暂无待确认/待成交委托")
        return

    df = pd.DataFrame(orders)
    df["状态"] = df["status"].apply(lambda s: s.value)
    df["方向"] = df["side"].map(_signal_action_label)
    st.dataframe(
        df[["time", "symbol", "方向", "price", "qty", "状态"]]
        if "time" in df.columns
        else df[["symbol", "方向", "price", "qty", "状态"]],
        use_container_width=True,
    )

    for rec in orders:
        oid = rec["order_id"]
        if st.button(f"撤单 {rec['symbol']}", key=f"cancel_{oid}"):
            if om.cancel(oid):
                st.success(f"已撤单 {rec['symbol']}")
                st.rerun()


def render_position_list(positions: list[dict]) -> None:
    """渲染持仓列表 + T+1 标记."""
    if not positions:
        st.info("当前无持仓")
        return
    df = pd.DataFrame(positions)
    df["盈亏%"] = (df["current_price"] / df["cost"] - 1) * 100
    df["T+1"] = df["available_qty"].apply(lambda x: "不可卖" if x == 0 else "可卖")
    display_cols = [
        c
        for c in [
            "symbol",
            "qty",
            "available_qty",
            "cost",
            "current_price",
            "盈亏%",
            "T+1",
        ]
        if c in df.columns
    ]
    st.dataframe(
        df[display_cols].style.format(
            {"盈亏%": "{:.2f}", "cost": "{:.2f}", "current_price": "{:.2f}"}
        ),
        use_container_width=True,
    )


def render_trade_list(om: OrderManager) -> None:
    """渲染成交回报."""
    fills = om.get_fills()
    if not fills:
        st.info("暂无成交")
        return
    df = pd.DataFrame(fills)
    df["方向"] = df["side"].map(_signal_action_label)
    st.dataframe(df[["symbol", "方向", "qty", "price"]], use_container_width=True)
