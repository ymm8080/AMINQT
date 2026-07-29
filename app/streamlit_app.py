"""
A股量化交易系统 — Streamlit 四页看板 (P10)
=====================================================
页面: 选股 (Pipeline-1 V3.5 清单) / 交易 (P2 框架演示) / 回测 (V3.5 协议+调参) / 配置.
启动: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保仓库根目录在 sys.path, 支持 `streamlit run app/streamlit_app.py` 直接启动
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import pandas as pd
import streamlit as st  # noqa: E402

from app.streamlit import (  # noqa: E402
    page_backtest,
    page_config,
    page_data_feed,
    page_selection,
    page_trading,
)

st.set_page_config(page_title="A股量化交易系统", page_icon="📈", layout="wide")

PAGES = {
    "选股看板": page_selection.render,
    "交易看板": page_trading.render,
    "回测中心": page_backtest.render,
    "数据管理": page_data_feed.render,
    "配置中心": page_config.render,
}


def main() -> None:
    st.sidebar.title("📈 A股量化")
    st.sidebar.caption("Pipeline-1 V3.5 (LightGBM 双轨) + 规则引擎 v2")
    page = st.sidebar.radio(
        "页面", list(PAGES), key="nav_page", index=list(PAGES).index("选股看板")
    )
    st.sidebar.divider()
    st.sidebar.caption("数据: akshare | 执行: miniQMT (sim) | 模型: 本地 LightGBM")
    st.sidebar.divider()
    _v3_path = "data/panel_full_enriched_v3.parquet"
    if os.path.exists(_v3_path):
        _pf = pd.read_parquet(_v3_path)
        st.sidebar.caption("v3 面板: {} 列 {} 只股票".format(len(_pf.columns), _pf["symbol"].nunique()))
    else:
        st.sidebar.caption("v3 面板: 未创建")
    PAGES[page]()


if __name__ == "__main__":
    main()
