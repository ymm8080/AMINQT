# -*- coding: utf-8 -*-
"""
页面 1: 选股看板 (P10, Pipeline-1)
======================================
三层股票视图: 选股池 (V3.5 清单) / 全市场 (演示) / 我的关注.
双击行 → 详情 (K线/分时/因子).
"""

from __future__ import annotations

import streamlit as st

from . import data_service as ds
from .components import intraday_chart, kline_chart


def _pool_df() -> tuple:
    """真实清单优先, 否则演示数据 (显著标记)."""
    lst, date = ds.load_latest_list()
    if lst is not None:
        return lst, date, False
    return ds.demo_list(), "DEMO", True


def render() -> None:
    st.header("选股看板 · Pipeline 1 (V3.5)")
    pool, pool_date, is_demo = _pool_df()
    if is_demo:
        st.warning("演示数据 — 未找到真实清单 (data/lists/), 运行 `python scripts/run_daily.py` 生成")
    else:
        st.caption(f"清单日期: {pool_date} | schema V1.0 | Top {len(pool)}")

    tab_pool, tab_market, tab_watch = st.tabs(["选股池", "全市场", "我的关注"])

    # ---------- Tab 1: 选股池 ----------
    with tab_pool:
        show = pool.copy()
        if "name" not in show.columns:
            show["name"] = show["symbol"].map(ds.DEMO_NAMES).fillna("-")
        cols = [c for c in ("symbol", "name", "score", "prob_up", "pred_ret_1d",
                            "pred_ret_3d", "pred_ret_5d", "momentum",
                            "signal_conflict", "industry") if c in show.columns]
        st.dataframe(show[cols].style.format({
            "score": "{:.4f}", "prob_up": "{:.3f}",
            "pred_ret_1d": "{:+.2%}", "pred_ret_3d": "{:+.2%}", "pred_ret_5d": "{:+.2%}"}),
            use_container_width=True, height=420)
        sel = st.selectbox("查看详情", show["symbol"].tolist(),
                           format_func=lambda s: f"{s} {ds.DEMO_NAMES.get(s, '')}",
                           key="pool_detail")
        if sel:
            _render_detail(sel)

    # ---------- Tab 2: 全市场 (演示) ----------
    with tab_market:
        st.info("全市场视图 (演示数据) — 生产接入 akshare 实时快照")
        q = st.text_input("搜索代码/名称", key="market_search")
        df = ds.demo_list(seed=7)
        if q:
            df = df[df["symbol"].str.contains(q) | df["name"].str.contains(q)]
        st.dataframe(df[["symbol", "name", "prob_up", "pred_ret_1d", "industry"]],
                     use_container_width=True)

    # ---------- Tab 3: 我的关注 ----------
    with tab_watch:
        items = ds.load_watchlist()
        if not items:
            st.info("暂无关注股 — 在选股池详情中添加")
        else:
            for it in items:
                c1, c2, c3 = st.columns([2, 4, 1])
                c1.write(f"**{it['symbol']}** {it.get('name', '')}")
                c2.write(it.get("note", ""))
                if c3.button("移除", key=f"unwatch_{it['symbol']}"):
                    ds.toggle_watchlist(it["symbol"])
                    st.rerun()


def _render_detail(symbol: str) -> None:
    """股票详情: K线 + 分时 + 关注按钮."""
    st.subheader(f"{symbol} {ds.DEMO_NAMES.get(symbol, '')}")
    if st.button("⭐ 关注/取消", key=f"watch_btn_{symbol}"):
        now = ds.toggle_watchlist(symbol, ds.DEMO_NAMES.get(symbol, ""))
        st.toast("已关注" if now else "已取消关注")
    period = st.radio("周期", ["日K", "分时"], horizontal=True, key=f"period_{symbol}")
    if period == "日K":
        df = ds.demo_ohlc(symbol)
        st.plotly_chart(kline_chart(df, title=f"{symbol} 日K"), use_container_width=True)
    else:
        df = ds.demo_intraday(symbol)
        st.plotly_chart(intraday_chart(df, prev_close=100.0), use_container_width=True)
