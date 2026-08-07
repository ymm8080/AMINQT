"""
看板图表组件 (P10) — plotly 纯函数, 输入 DataFrame 输出 Figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def kline_chart(
    df: pd.DataFrame, ma_list: tuple = (5, 10, 20), title: str = ""
) -> go.Figure:
    """日K线 + 均线 + 成交量副图."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color="#e54545",
            decreasing_line_color="#26a69a",
        ),
        row=1,
        col=1,
    )
    for w in ma_list:
        if len(df) >= w:
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["close"].rolling(w).mean(),
                    name=f"MA{w}",
                    line={"width": 1},
                ),
                row=1,
                col=1,
            )
    colors = np.where(df["close"] >= df["open"], "#e54545", "#26a69a")
    fig.add_trace(
        go.Bar(
            x=df["date"],
            y=df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.6,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=520,
        xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        showlegend=True,
    )
    return fig


def intraday_chart(
    df: pd.DataFrame, prev_close: float | None = None, title: str = "分时"
) -> go.Figure:
    """分时走势线 + VWAP均价线 + 昨收价 + 成交量副图(含5日均量)."""
    vwap = (df["price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    vol_ma5 = df["volume"].rolling(window=5, min_periods=1).mean()
    colors = np.where(
        df["price"] >= df["price"].shift(1).fillna(df["price"].iloc[0]),
        "#e54545",
        "#26a69a",
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["price"],
            name="价格",
            line={"color": "#1f77b4", "width": 1.5},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=vwap,
            name="VWAP",
            line={"color": "#ff7f0e", "width": 1, "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    if prev_close:
        fig.add_hline(
            y=prev_close,
            line_dash="dash",
            line_color="gray",
            annotation_text="昨收",
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Bar(
            x=df["time"],
            y=df["volume"],
            name="成交量",
            marker_color=colors,
            opacity=0.7,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=vol_ma5,
            name="成交量MA5",
            line={"color": "#ff7f0e", "width": 1},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title=title,
        height=420,
        xaxis_rangeslider_visible=False,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def equity_curve(
    nav_df: pd.DataFrame,
    title: str = "净值曲线",
    benchmark_df: pd.DataFrame | None = None,
    benchmark_name: str = "基准",
) -> go.Figure:
    """回测净值曲线 (可选叠加基准对比线).

    Args:
        nav_df: 含 date/nav 列的 DataFrame.
        title: 图表标题.
        benchmark_df: 含 date/nav 列的基准 DataFrame (归一化为同一起点).
        benchmark_name: 基准曲线名称.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_df["date"],
            y=nav_df["nav"],
            name="策略净值",
            line={"color": "#e54545", "width": 1.5},
        )
    )
    if benchmark_df is not None and not benchmark_df.empty:
        bench = benchmark_df.copy()
        # 归一化基准到与策略同一起点
        start_nav = nav_df["nav"].iloc[0]
        bench_start = bench["nav"].iloc[0]
        if bench_start > 0:
            bench["nav"] = bench["nav"] / bench_start * start_nav
        fig.add_trace(
            go.Scatter(
                x=bench["date"],
                y=bench["nav"],
                name=benchmark_name,
                line={"color": "#1f77b4", "width": 1, "dash": "dot"},
            )
        )
    fig.update_layout(
        title=title,
        height=320,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_tickformat=".2f",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


def drawdown_chart(nav_df: pd.DataFrame, title: str = "回撤") -> go.Figure:
    """Underwater 回撤图."""
    nav = nav_df["nav"]
    dd = nav / nav.cummax() - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_df["date"],
            y=dd,
            fill="tozeroy",
            name="回撤",
            line={"color": "#26a69a"},
        )
    )
    fig.update_layout(
        title=title,
        height=220,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_tickformat=".1%",
    )
    return fig


def factor_radar(factors: dict, top_n: int = 10, title: str = "因子雷达") -> go.Figure:
    """Top-N 因子值雷达图."""
    items = list(factors.items())[:top_n]
    labels, values = zip(*items, strict=False) if items else ([], [])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=list(values) + [values[0] if values else 0],
            theta=list(labels) + [labels[0] if labels else ""],
            fill="toself",
            name="因子值",
        )
    )
    fig.update_layout(
        title=title, height=360, margin={"l": 40, "r": 40, "t": 40, "b": 10}
    )
    return fig


def comparison_nav_chart(
    navs: dict[str, pd.DataFrame],
    title: str = "多模式净值对比",
) -> go.Figure:
    """多模式净值对比图 (Squad vs Sniper vs 基准).

    Args:
        navs: {label: DataFrame(date, nav)} — 每条曲线归一化到 1.0.
        title: 图表标题.
    """
    fig = go.Figure()
    palette = ["#e54545", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
    for i, (label, df) in enumerate(navs.items()):
        if df is None or df.empty:
            continue
        norm = df.copy()
        start = norm["nav"].iloc[0]
        if start > 0:
            norm["nav"] = norm["nav"] / start
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=norm["date"],
                y=norm["nav"],
                name=label,
                line={"color": color, "width": 1.5 if i == 0 else 1},
            )
        )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.4)
    fig.update_layout(
        title=title,
        height=400,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        yaxis_tickformat=".2f",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig
