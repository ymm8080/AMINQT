# -*- coding: utf-8 -*-
"""
看板图表组件 (P10) — plotly 纯函数, 输入 DataFrame 输出 Figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def kline_chart(
    df: pd.DataFrame, ma_list: tuple = (5, 20, 60), title: str = ""
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
                    line=dict(width=1),
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
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
    )
    return fig


def intraday_chart(
    df: pd.DataFrame, prev_close: float | None = None, title: str = "分时"
) -> go.Figure:
    """分时走势线 + 均价线 + 昨收价."""
    avg = (df["price"] * df["volume"]).cumsum() / df["volume"].cumsum()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["price"],
            name="价格",
            line=dict(color="#1f77b4", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=avg,
            name="均价",
            line=dict(color="#ff7f0e", width=1, dash="dot"),
        )
    )
    if prev_close:
        fig.add_hline(
            y=prev_close, line_dash="dash", line_color="gray", annotation_text="昨收"
        )
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def equity_curve(nav_df: pd.DataFrame, title: str = "净值曲线") -> go.Figure:
    """回测净值曲线."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=nav_df["date"],
            y=nav_df["nav"],
            name="策略净值",
            line=dict(color="#e54545", width=1.5),
        )
    )
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickformat=".2f",
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
            line=dict(color="#26a69a"),
        )
    )
    fig.update_layout(
        title=title,
        height=220,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickformat=".1%",
    )
    return fig


def factor_radar(factors: dict, top_n: int = 10, title: str = "因子雷达") -> go.Figure:
    """Top-N 因子值雷达图."""
    items = list(factors.items())[:top_n]
    labels, values = zip(*items) if items else ([], [])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=list(values) + [values[0] if values else 0],
            theta=list(labels) + [labels[0] if labels else ""],
            fill="toself",
            name="因子值",
        )
    )
    fig.update_layout(title=title, height=360, margin=dict(l=40, r=40, t=40, b=10))
    return fig
