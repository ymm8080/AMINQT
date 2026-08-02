"""
页面 1: 选股看板 (P10, Pipeline-1)
=====================================
两层股票视图: 选股池 (V3.5 清单) / 全市场 (演示).
左侧表格, 右侧详情面板: 下拉框 + 上下按钮切股, 可标记/取消日内买入标记.
Pipeline-1 推荐第二天买入标的自动标为日内买入; 标记可手工点选.
可人为添加股票到选股池.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from . import data_service as ds
from .components import (
    chip_control_chart,
    factor_radar,
    find_bull_chart,
    intraday_chart,
    kline_chart,
    main_force_chips_chart,
    trend_top_bottom_chart,
)


# ---------- 数据加载 ----------


def _pool_df() -> tuple:
    """真实清单优先, 否则演示数据 (显著标记)."""
    lst, date = ds.load_latest_list()
    if lst is not None:
        return lst, date, False
    return ds.demo_list(), "DEMO", True


# ---------- 选股池: 人为添加股票 ----------


def _add_symbol_to_pool(pool: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """把用户输入的股票代码追加到选股池 (演示模式补充基本信息)."""
    symbol = str(symbol).strip()
    if not symbol or symbol in pool["symbol"].values:
        return pool
    row = {"symbol": symbol, "name": ds.DEMO_NAMES.get(symbol, symbol)}
    # 复制同类型列的默认值
    for col in pool.columns:
        if col in row:
            continue
        if col == "priority":
            row[col] = True  # 人为添加的默认标记为日内买入候选
        elif pool[col].dtype == "object":
            row[col] = "-"
        elif pool[col].dtype in ("int64", "Int64"):
            row[col] = 0
        elif pool[col].dtype == "bool":
            row[col] = False
        else:
            row[col] = 0.0
    return pd.concat([pool, pd.DataFrame([row])], ignore_index=True)


# ---------- 显示列 ----------


def _display_cols(pool: pd.DataFrame) -> list[str]:
    """选股池展示列 (保持稳定的列顺序)."""
    candidates = [
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


# ---------- 页面入口 ----------


def render() -> None:
    st.header("选股看板 · Pipeline 1 (V3.5)")
    pool, pool_date, is_demo = _pool_df()
    if is_demo:
        st.warning(
            "演示数据 — 未找到真实清单 (data/lists/), 运行 `python scripts/run_daily.py` 生成"
        )
    else:
        st.caption(f"清单日期: {pool_date} | schema V1.0 | Top {len(pool)}")

    # 人为添加股票
    with st.expander("➕ 添加股票到选股池", expanded=False):
        new_sym = st.text_input(
            "输入股票代码", placeholder="如 600519", key="add_symbol"
        )
        if st.button("添加", key="btn_add_symbol"):
            pool = _add_symbol_to_pool(pool, new_sym)
            st.session_state["selection_pool_override"] = pool
            st.rerun()

    # 允许 session_state 覆盖 pool (本次运行有效)
    if "selection_pool_override" in st.session_state:
        override = st.session_state["selection_pool_override"]
        if isinstance(override, pd.DataFrame) and not override.empty:
            # 只追加当前 pool 中没有的 symbol
            missing = override[~override["symbol"].isin(pool["symbol"])]
            if not missing.empty:
                pool = pd.concat([pool, missing], ignore_index=True)

    tab_pool, tab_market = st.tabs(["选股池", "全市场"])

    # ---------- Tab 1: 选股池 ----------
    with tab_pool:
        left, right = st.columns([3, 2])

        with left:
            _render_pool_table(pool)

        with right:
            _render_detail_panel(pool)

        # 板块行情 (涨跌幅 + 日内曲线)
        st.divider()
        _render_sector_panel()

    # ---------- Tab 2: 全市场 (演示) ----------
    with tab_market:
        _render_market_tab()


# ---------- 选股池表格 ----------


def _render_pool_table(pool: pd.DataFrame) -> None:
    """渲染选股池表格, 含日内走势 sparkline 与买入标记."""
    show = pool.copy()
    if "name" not in show.columns:
        show["name"] = show["symbol"].map(ds.DEMO_NAMES).fillna("-")

    display = show[_display_cols(show)].copy()

    # 日内走势 sparkline: 每个 symbol 120 根分时价格
    intraday_series = {}
    for sym in show["symbol"]:
        intra = ds.demo_intraday(sym)
        # 归一化为收益率序列, 便于跨股比较
        p0 = intra["price"].iloc[0]
        intraday_series[sym] = ((intra["price"] / p0) - 1).tolist()
    display["日内走势"] = display["symbol"].map(intraday_series)

    # 日内买入标记列
    display["日内买入"] = display["symbol"].apply(
        lambda s: (
            bool(show.loc[show["symbol"] == s, "priority"].iloc[0])
            if "priority" in show.columns
            else False
        )
    )

    # 列顺序: 代码/名称/走势/评分等
    col_order = ["symbol", "name", "日内走势"] + [
        c for c in display.columns if c not in ("symbol", "name", "日内走势")
    ]

    st.dataframe(
        display[col_order].style.format(
            {
                "score": "{:.4f}",
                "prob_up": "{:.3f}",
                "pred_ret_1d": "{:+.2%}",
                "pred_ret_3d": "{:+.2%}",
                "pred_ret_5d": "{:+.2%}",
            }
        ),
        column_config={
            "symbol": st.column_config.TextColumn("代码"),
            "name": st.column_config.TextColumn("名称"),
            "日内走势": st.column_config.LineChartColumn(
                "日内走势", y_min=-0.03, y_max=0.03
            ),
        },
        use_container_width=True,
        height=420,
    )

    # 手工点选日内买入标记
    st.caption("手工标记/取消 日内买入候选")
    cols = st.columns(min(len(show), 6))
    for idx, row in show.iterrows():
        symbol = row["symbol"]
        name = row.get("name", symbol)
        is_priority = bool(row.get("priority", False))
        label = f"{'✅' if is_priority else '⬜'} {symbol} {name}"
        with cols[idx % len(cols)]:
            if st.button(label, key=f"toggle_priority_{symbol}"):
                ds.toggle_priority(symbol)
                st.rerun()


# ---------- 详情面板 ----------


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

    # 标记按钮 + 跳转交易看板
    col_mark, col_trade = st.columns(2)
    is_priority = bool(pool.loc[pool["symbol"] == selected, "priority"].iloc[0])
    label = "✅ 取消日内买入" if is_priority else "⬜ 标记日内买入"
    if col_mark.button(label, use_container_width=True):
        ds.toggle_priority(selected)
        st.toast("已取消日内买入标记" if is_priority else "已标记为日内买入")
        st.rerun()
    if col_trade.button("📈 去交易看板", use_container_width=True):
        st.session_state["trading_symbol"] = selected
        st.session_state["nav_page"] = "交易看板"
        st.rerun()

    if is_priority:
        st.caption("Pipeline-1 推荐次日买入; 可手动取消或补充标记。")

    # 详情图表
    _render_charts(selected)


# ---------- 图表区 ----------


def _render_charts(symbol: str) -> None:
    """个股 K线/分时 + 参考指标图案 + 筹码分布."""
    st.subheader(f"{symbol} {ds.DEMO_NAMES.get(symbol, '')}")
    period = st.radio("周期", ["日K", "分时"], horizontal=True, key=f"period_{symbol}")

    ohlc = ds.demo_ohlc(symbol)
    if period == "日K":
        st.plotly_chart(
            kline_chart(ohlc, ma_list=(5, 10, 20), title=f"{symbol} 日K"),
            use_container_width=True,
        )
        # MACD 副图
        st.plotly_chart(_macd_chart(ohlc), use_container_width=True)
    else:
        df = ds.demo_intraday(symbol)
        st.plotly_chart(
            intraday_chart(df, prev_close=ohlc["close"].iloc[-1]),
            use_container_width=True,
        )

    st.divider()
    st.subheader("参考指标")
    st.caption("来自 D:\\aminqt\\reference\\indicator (演示数据近似计算)")
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

    # 筹码分布
    st.divider()
    st.subheader("筹码分布")
    st.plotly_chart(_chip_distribution_chart(ohlc), use_container_width=True)

    # 因子雷达 (B2)
    st.divider()
    st.subheader("因子雷达")
    st.caption("从 OHLC 数据近似计算的 V3.5 关键因子值")
    factors = _compute_factor_values(ohlc)
    if factors:
        col_radar, col_table = st.columns([2, 1])
        with col_radar:
            st.plotly_chart(factor_radar(factors, top_n=10, title=f"{symbol} 因子雷达"), use_container_width=True)
        with col_table:
            st.dataframe(
                pd.DataFrame(
                    list(factors.items()), columns=["因子", "值"]
                ).style.format({"值": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )


def _compute_factor_values(df: pd.DataFrame) -> dict:
    """从 OHLC 数据近似计算 V3.5 关键因子值 (用于雷达图).

    Returns:
        {因子名: 标准化值 (0~1)} — 10 个关键因子.
    """
    try:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)
        result = {}

        # 1. MACD 信号强度 (DIF 归一化)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        result["MACD"] = float(dif.iloc[-1] / close.iloc[-1]) if close.iloc[-1] > 0 else 0

        # 2. RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-8)
        rsi = 100 - 100 / (1 + rs)
        result["RSI"] = float(rsi.iloc[-1] / 100) if not np.isnan(rsi.iloc[-1]) else 0.5

        # 3. KDJ K 值
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        rsv = (close - low14) / (high14 - low14).replace(0, 1e-8) * 100
        k = rsv.rolling(3).mean()
        result["KDJ"] = float(k.iloc[-1] / 100) if not np.isnan(k.iloc[-1]) else 0.5

        # 4. 布林带宽
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        boll_width = (ma20 + 2 * std20 - (ma20 - 2 * std20)) / ma20
        result["BOLL宽"] = float(boll_width.iloc[-1]) if not np.isnan(boll_width.iloc[-1]) else 0

        # 5. ATR 百分比
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1)),
            ),
        )
        atr = tr.rolling(14).mean()
        atr_pct = atr / close
        result["ATR%"] = float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else 0

        # 6. 量比
        vol_ma5 = volume.rolling(5).mean()
        vol_ratio = volume / vol_ma5.replace(0, 1e-8)
        result["量比"] = float(vol_ratio.iloc[-1]) if not np.isnan(vol_ratio.iloc[-1]) else 1

        # 7. 乖离率 (close vs MA20)
        bias = (close - ma20) / ma20
        result["乖离MA20"] = float(bias.iloc[-1]) if not np.isnan(bias.iloc[-1]) else 0

        # 8. 量价背离 (相关性)
        vol_price_corr = close.pct_change().rolling(20).corr(volume.pct_change())
        result["量价相关"] = float(vol_price_corr.iloc[-1]) if not np.isnan(vol_price_corr.iloc[-1]) else 0

        # 9. 动量 (20日收益)
        mom_20d = close / close.shift(20) - 1
        result["动量20d"] = float(mom_20d.iloc[-1]) if not np.isnan(mom_20d.iloc[-1]) else 0

        # 10. 波动率 (20日)
        vol_20d = close.pct_change().rolling(20).std()
        result["波动率20d"] = float(vol_20d.iloc[-1]) if not np.isnan(vol_20d.iloc[-1]) else 0

        return result
    except Exception:
        return {}


def _macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD 副图 (DIF/DEA/BAR)."""
    close = df["close"].astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    bar = (dif - dea) * 2

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=df["date"], y=dif, name="DIF", line={"color": "white"}))
    fig.add_trace(go.Scatter(x=df["date"], y=dea, name="DEA", line={"color": "yellow"}))
    colors = np.where(bar >= 0, "#e54545", "#26a69a")
    fig.add_trace(
        go.Bar(x=df["date"], y=bar, name="MACD", marker_color=colors, opacity=0.6)
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="MACD",
        height=220,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        showlegend=True,
    )
    return fig


def _chip_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """右侧筹码分布图 (近似)."""
    close = df["close"].astype(float)
    # 用收盘价分布近似筹码分布
    hist, bins = np.histogram(close.dropna(), bins=20)
    price_centers = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hist,
            y=price_centers,
            orientation="h",
            name="筹码",
            marker_color="#1f77b4",
        )
    )
    fig.add_vline(
        x=close.iloc[-1],
        line_dash="dash",
        line_color="red",
        annotation_text=f"当前价 {close.iloc[-1]:.2f}",
    )
    fig.update_layout(
        title="筹码分布 (近似)",
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis_title="筹码量",
        yaxis_title="价格",
    )
    return fig


def _render_sector_panel() -> None:
    """板块涨跌幅 + 小号板块日内曲线 sparkline."""
    st.subheader("板块行情")
    sector_df = ds.demo_sector_changes()
    # 为每个板块生成日内分时收益序列
    sector_df["日内走势"] = sector_df["板块"].apply(
        lambda s: (
            (
                ds.demo_sector_intraday(s)["price"]
                / ds.demo_sector_intraday(s)["price"].iloc[0]
            )
            - 1
        ).tolist()
    )
    st.dataframe(
        sector_df,
        column_config={
            "板块": st.column_config.TextColumn("板块"),
            "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="+.2%%"),
            "上涨家数": st.column_config.NumberColumn("上涨家数"),
            "下跌家数": st.column_config.NumberColumn("下跌家数"),
            "日内走势": st.column_config.LineChartColumn(
                "日内走势", y_min=-0.015, y_max=0.015
            ),
        },
        column_order=["板块", "涨跌幅", "日内走势", "上涨家数", "下跌家数"],
        hide_index=True,
        use_container_width=True,
    )


# ---------- 全市场 ----------


def _render_market_tab() -> None:
    """全市场演示 tab."""
    st.info("全市场视图 (演示数据) — 生产接入 akshare 实时快照")
    q = st.text_input("搜索代码/名称", key="market_search")
    df = ds.demo_list(seed=7)
    if q:
        df = df[df["symbol"].str.contains(q) | df["name"].str.contains(q)]
    st.dataframe(
        df[["symbol", "name", "priority", "prob_up", "pred_ret_1d", "industry"]],
        use_container_width=True,
    )


if __name__ == "__main__":
    render()
