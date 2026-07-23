# -*- coding: utf-8 -*-
"""
A股量化交易系统 — Streamlit 四页看板 (P10)
=====================================================
页面: 选股 (Pipeline-1 V3.5 清单) / 交易 (P2 框架演示) / 回测 (V3.5 协议+调参) / 配置.
启动: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.streamlit import page_backtest, page_config, page_selection, page_trading

st.set_page_config(page_title="A股量化交易系统", page_icon="📈", layout="wide")

PAGES = {
    "选股看板": page_selection.render,
    "交易看板": page_trading.render,
    "回测中心": page_backtest.render,
    "配置中心": page_config.render,
}


def main() -> None:
    st.sidebar.title("📈 A股量化")
    st.sidebar.caption("Pipeline-1 V3.5 (LightGBM 双轨) + 规则引擎 v2")
    page = st.sidebar.radio("页面", list(PAGES))
    st.sidebar.divider()
    st.sidebar.caption("数据: akshare | 执行: miniQMT (sim) | 模型: 本地 LightGBM")
    PAGES[page]()


if __name__ == "__main__":
    main()
