"""
页面 1: 选股看板 (P10, Pipeline-1)
======================================
三层股票视图: 选股池 (V3.5 清单) / 全市场 (演示) / 重点股。
左侧表格, 右侧详情面板: 下拉框 + 上下按钮切股, 可标记/取消重点股。
Pipeline-1 推荐买入自动标为重点股; 详情面板展示参考指标图案。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from . import data_service as ds
from .components import (
    chip_control_chart,
    find_bull_chart,
    intraday_chart,
    kline_chart,
    main_force_chips_chart,
    trend_top_bottom_chart,
)


def _pool_df() -> tuple:
    """真实清单优先, 否则演示数据 (显著标记)."""
    lst, date = ds.load_latest_list()
    if lst is not None:
        return lst, date, False
    return ds.demo_list(), "DEMO", True


def _display_cols(pool: pd.DataFrame) -> list[str]:
    """选股池展示列 (保持稳定的列顺序)."""
    candidates = [
        "priority",
        "symbol",
        "name",
        "score",
        "prob_up",
        "pred_ret_1d",
        "pred_ret_3d",
        "pred_ret_5d",
        "momentum",
        "signal_conflict",
        "industry",
    ]
    return [c for c in candidates if c in pool.columns]


def _priority_label(flag: bool) -> str:
    return "⭐ 重点股" if flag else "☆ 标为重点股"


def render() -> None:
    st.header("选股看板 · Pipeline 1 (V3.5)")
    pool, pool_date, is_demo = _pool_df()
    if is_demo:
        st.warning(
            "演示数据 — 未找到真实清单 (data/lists/), 运行 `python scripts/run_daily.py` 生成"
        )
    else:
        st.caption(f"清单日期: {pool_date} | schema V1.0 | Top {len(pool)}")

    tab_pool, tab_market, tab_priority = st.tabs(["选股池", "全市场", "重点股"])

    # ---------- Tab 1: 选股池 ----------
    with tab_pool:
        left, right = st.columns([3, 2])

        with left:
            show = pool.copy()
            if "name" not in show.columns:
                show["name"] = show["symbol"].map(ds.DEMO_NAMES).fillna("-")
            # 用可读列名替换 priority
            display = show[_display_cols(show)].copy()
            if "priority" in display.columns:
                display["重点"] = display["priority"].map({True: "⭐", False: ""})
                display = display.drop(columns=["priority"])

            st.dataframe(
                display.style.format(
                    {
                        "score": "{:.4f}",
                        "prob_up": "{:.3f}",
                        "pred_ret_1d": "{:+.2%}",
                        "pred_ret_3d": "{:+.2%}",
                        "pred_ret_5d": "{:+.2%}",
                    }
                ),
                use_container_width=True,
                height=420,
            )

        with right:
            _render_detail_panel(pool)

    # ---------- Tab 2: 全市场 (演示) ----------
    with tab_market:
        st.info("全市场视图 (演示数据) — 生产接入 akshare 实时快照")
        q = st.text_input("搜索代码/名称", key="market_search")
        df = ds.demo_list(seed=7)
        if q:
            df = df[df["symbol"].str.contains(q) | df["name"].str.contains(q)]
        st.dataframe(
            df[["symbol", "name", "priority", "prob_up", "pred_ret_1d", "industry"]],
            use_container_width=True,
        )

    # ---------- Tab 3: 重点股 ----------
    with tab_priority:
        _render_priority_tab(pool)


def _render_detail_panel(pool: pd.DataFrame) -> None:
    """右侧详情面板: 切股 + 标记 + K线/分时/指标."""
    symbols = pool["symbol"].tolist()
    if not symbols:
        st.info("暂无股票")
        return

    # 维护当前选中索引, 支持上下按钮
    if "sel_idx" not in st.session_state:
        st.session_state.sel_idx = 0
    st.session_state.sel_idx = max(0, min(st.session_state.sel_idx, len(symbols) - 1))

    selected = symbols[st.session_state.sel_idx]

    # 下拉框: 同步当前索引
    def _on_select_change():
        st.session_state.sel_idx = symbols.index(st.session_state.pool_select)

    selected = st.selectbox(
        "查看详情",
        symbols,
        index=st.session_state.sel_idx,
        format_func=lambda s: f"{s} {ds.DEMO_NAMES.get(s, '')}",
        key="pool_select",
        on_change=_on_select_change,
    )

    # 上下按钮
    c_prev, c_next, c_spacer = st.columns([1, 1, 3])
    if c_prev.button("⬆ 上一个", use_container_width=True):
        st.session_state.sel_idx = max(0, st.session_state.sel_idx - 1)
        st.rerun()
    if c_next.button("⬇ 下一个", use_container_width=True):
        st.session_state.sel_idx = min(len(symbols) - 1, st.session_state.sel_idx + 1)
        st.rerun()

    # 标记按钮
    is_priority = bool(pool.loc[pool["symbol"] == selected, "priority"].iloc[0])
    label = _priority_label(is_priority)
    if st.button(label, use_container_width=True):
        now = ds.toggle_priority(selected)
        st.toast("已标为重点股" if now else "已取消重点股标记")
        st.rerun()

    if is_priority:
        st.caption("Pipeline-1 推荐买入已自动标记; 可手动取消或补充标记。")

    # 详情图表
    _render_charts(selected)


def _render_charts(symbol: str) -> None:
    """个股 K线/分时 + 参考指标图案."""
    st.subheader(f"{symbol} {ds.DEMO_NAMES.get(symbol, '')}")
    period = st.radio("周期", ["日K", "分时"], horizontal=True, key=f"period_{symbol}")
    if period == "日K":
        df = ds.demo_ohlc(symbol)
        st.plotly_chart(
            kline_chart(df, title=f"{symbol} 日K"), use_container_width=True
        )
    else:
        df = ds.demo_intraday(symbol)
        st.plotly_chart(
            intraday_chart(df, prev_close=100.0), use_container_width=True
        )

    st.divider()
    st.subheader("参考指标")
    st.caption("来自 D:\\aminqt\\reference\\indicator (演示数据近似计算)")
    ohlc = ds.demo_ohlc(symbol)
    ind1, ind2 = st.columns(2)
    with ind1:
        st.plotly_chart(main_force_chips_chart(ohlc), use_container_width=True)
    with ind2:
        st.plotly_chart(chip_control_chart(ohlc), use_container_width=True)
    ind3, ind4 = st.columns(2)
    with ind3:
        st.plotly_chart(find_bull_chart(ohlc), use_container_width=True)
    with ind4:
        st.plotly_chart(trend_top_bottom_chart(ohlc), use_container_width=True)


def _render_priority_tab(pool: pd.DataFrame) -> None:
    """重点股管理页."""
    priority_df = pool[pool["priority"]].copy()
    if priority_df.empty:
        st.info("暂无重点股 — 在选股池详情中标记")
        return

    st.caption(f"共 {len(priority_df)} 只重点股 (次日日内操作候选池)")
    display_cols = [c for c in ["symbol", "name", "score", "pred_ret_1d", "industry"] if c in priority_df.columns]
    rename_map = {"symbol": "代码", "pred_ret_1d": "预测1日收益"}
    display_df = priority_df[display_cols].rename(columns=rename_map).reset_index(drop=True)

    st.dataframe(
        display_df.style.format({"score": "{:.4f}", "预测1日收益": "{:+.2%}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("管理")
    cols = st.columns(4)
    for idx, row in display_df.iterrows():
        symbol = row["代码"]
        with cols[idx % 4]:
            if st.button(f"取消 {symbol}", key=f"unpriority_{symbol}"):
                ds.toggle_priority(symbol)
                st.rerun()


if __name__ == "__main__":
    render()
