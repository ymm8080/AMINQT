"""
app/streamlit/components 包 (P10)
===================================
统一导出图表组件与交易面板组件, 保持 `from app.streamlit.components import ...`
向后兼容。
"""

from __future__ import annotations

from .charts import (
    drawdown_chart,
    equity_curve,
    factor_radar,
    intraday_chart,
    kline_chart,
)
from .trading_panel import (
    render_order_queue,
    render_position_list,
    render_signal_list,
    render_trade_list,
    render_trading_control,
)

__all__ = [
    "kline_chart",
    "intraday_chart",
    "equity_curve",
    "drawdown_chart",
    "factor_radar",
    "render_trading_control",
    "render_signal_list",
    "render_order_queue",
    "render_position_list",
    "render_trade_list",
]
