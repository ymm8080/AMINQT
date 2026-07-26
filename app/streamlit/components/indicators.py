"""
参考指标图表组件 (P10)
========================
基于 D:\aminqt\reference\indicator 四个指标公式绘制的简化版图案。
当前使用 demo OHLC 数据做近似计算; 生产环境可替换为真实日线。

四个指标:
1. 主力筹码指标 (主力轨迹 + 主力平均线)
2. 主力筹码控盘程度 N (近似筹码分布)
3. 发现牛股 (多周期 EMA 买入信号)
4. 同花顺益盟趋势顶底 (短/中/长期威廉类摆动指标)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def _ema(series: pd.Series, span: int) -> pd.Series:
    """指数移动平均 (同花顺 EMA 近似)."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """简单移动平均."""
    return series.rolling(window).mean()


# ───────────────────────────────────────────────────────────────
# 1. 主力筹码指标
# ───────────────────────────────────────────────────────────────
def main_force_chips_chart(df: pd.DataFrame, title: str = "主力筹码指标") -> go.Figure:
    """主力轨迹线 (白) + 主力平均线 (黄), 附 0 轴参考。"""
    close = df["close"].astype(float)
    n1, n2 = 9, 5
    mtm = close.diff()
    main_traj = 100 * _ema(_ema(mtm, n1), n1) / _ema(_ema(mtm.abs(), n1), n1)
    main_avg = main_traj.rolling(n2).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=main_traj,
            name="主力轨迹",
            line={"color": "white", "width": 1.2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=main_avg,
            name="主力平均",
            line={"color": "yellow", "width": 1.2},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title,
        height=260,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font={"color": "#cccccc"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


# ───────────────────────────────────────────────────────────────
# 2. 主力筹码控盘程度 N
# ───────────────────────────────────────────────────────────────
def chip_control_chart(df: pd.DataFrame, title: str = "主力筹码控盘程度 N") -> go.Figure:
    """近似版: 用价格在过去 30 日区间内的位置模拟筹码集中度。"""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    a01 = (close + df["open"].astype(float) + low + high) / 4

    n = 30
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    # 用中价在区间中的位置近似 "获利盘比例"
    a04 = 100 * (a01 - ll) / (hh - ll)
    a02 = 100 * (a01 * 1.04 - ll) / (hh - ll)
    a06 = 100 - a02
    a08 = a02 - a04

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=a04,
            name="获利盘(近)",
            fill="tozeroy",
            line={"color": "red"},
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=a02,
            name="获利盘(远)",
            fill="tonexty",
            line={"color": "green"},
            opacity=0.3,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=a06,
            name="套牢盘",
            line={"color": "cyan", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=a08,
            name="筹码差",
            line={"color": "yellow", "width": 1},
        )
    )
    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=title + " (近似)",
        height=260,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font={"color": "#cccccc"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


# ───────────────────────────────────────────────────────────────
# 3. 发现牛股
# ───────────────────────────────────────────────────────────────
def find_bull_chart(df: pd.DataFrame, title: str = "发现牛股") -> go.Figure:
    """EMA3/5/7/12/20/50 + 金叉买入信号标记。"""
    close = df["close"].astype(float)
    a1 = _ema(close, 3)
    a2 = _ema(close, 5)
    a3 = _ema(close, 7)
    a4 = _ema(close, 12)
    a5 = _ema(close, 20)
    a6 = _ema(close, 50)

    # SS: EMA3 上穿 EMA20 且阳线且涨幅 ≥ 1.8%
    prev_close = close.shift(1)
    ss = (a1 > a5) & (a1.shift(1) <= a5.shift(1))
    ss = ss & (close > df["open"].astype(float)) & (close > prev_close)
    ss = ss & (close / prev_close >= 1.018)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=a1, name="EMA3", line={"color": "white", "width": 1}))
    fig.add_trace(go.Scatter(x=df["date"], y=a2, name="EMA5", line={"color": "yellow", "width": 1}))
    fig.add_trace(go.Scatter(x=df["date"], y=a3, name="EMA7", line={"color": "magenta", "width": 1}))
    fig.add_trace(go.Scatter(x=df["date"], y=a4, name="EMA12", line={"color": "green", "width": 1}))
    fig.add_trace(go.Scatter(x=df["date"], y=a5, name="EMA20", line={"color": "red", "width": 1}))
    fig.add_trace(go.Scatter(x=df["date"], y=a6, name="EMA50", line={"color": "blue", "width": 2}))

    signal_dates = df.loc[ss, "date"]
    signal_prices = close[ss]
    if not signal_dates.empty:
        fig.add_trace(
            go.Scatter(
                x=signal_dates,
                y=signal_prices,
                mode="markers",
                name="买入信号",
                marker={"color": "gold", "size": 10, "symbol": "triangle-up"},
            )
        )

    fig.update_layout(
        title=title,
        height=260,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig


# ───────────────────────────────────────────────────────────────
# 4. 同花顺益盟趋势顶底
# ───────────────────────────────────────────────────────────────
def trend_top_bottom_chart(df: pd.DataFrame, title: str = "趋势顶底") -> go.Figure:
    """短期线/中期线/长期线 (威廉类摆动指标 0~100)。"""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    hh14 = high.rolling(14).max()
    ll14 = low.rolling(14).min()
    b = 100 * (close - hh14) / (hh14 - ll14)
    short_line = b + 100

    hh34 = high.rolling(34).max()
    ll34 = low.rolling(34).min()
    raw34 = 100 * (close - hh34) / (hh34 - ll34)
    mid_line = _ema(raw34, 4) + 100
    long_line = _sma(raw34, 19) + 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=short_line,
            name="短期线",
            line={"color": "#888888", "width": 1},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=mid_line,
            name="中期线",
            line={"color": "yellow", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=long_line,
            name="长期线",
            line={"color": "red", "width": 1},
        )
    )
    # 参考区间
    for y, color in [(20, "green"), (80, "green"), (90, "red")]:
        fig.add_hline(y=y, line_dash="dash", line_color=color, opacity=0.4)

    fig.update_layout(
        title=title,
        height=260,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font={"color": "#cccccc"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    return fig
