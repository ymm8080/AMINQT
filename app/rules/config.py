# -*- coding: utf-8 -*-
"""
规则引擎 v2 配置 (DESIGN §15.6, 实施计划 P15.1)
=====================================================
全部阈值集中此类, 改规则只改这一个类.
**调参约定**: 标记 [TUNABLE] 的字段为回测调参目标 — 预设初始值,
由 app/pipeline1/param_tuner.py 基于回测结果在 bounds 内搜索调整 (用户 2026-07-22 需求).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """L0-L4 阈值 + 调参边界. 初始值 = 用户规则文档预设."""

    # ---- 板块涨跌幅 (%) — 静态, 实盘当日用; 回测须用 f(board, date) 分段 ----
    LIMIT_MAIN: float = 10.0       # 主板 60/000
    LIMIT_GEM: float = 20.0        # 创业板/科创 30/68
    LIMIT_BSE: float = 30.0        # 北交所 8/4/92

    # ---- L3 自选标记 (盘后) ----
    control_min: float = 30.0        # [TUNABLE] 主力控盘比例下限 %  bounds=(20, 40)
    surge_pct: float = 8.0           # [TUNABLE] 两周内单日涨幅阈值  bounds=(5, 12)
    surge_window: int = 10           # 两周 = 10个交易日
    peak_lookback: int = 20          # 吸筹峰回看一个月

    # ---- L3 日线买点 ----
    turnover_max_entry: float = 50.0     # [TUNABLE] 换手率>50%不选  bounds=(30, 70)
    min_avg_amount_20d: float = 5e7      # 流动性下限

    # ---- L3 日内买点 ----
    gap_open_pct: float = 4.0            # [TUNABLE] 路径A跳空阈值  bounds=(2, 6)
    breadth_min: float = 0.60            # [TUNABLE] 路径B市场广度  bounds=(0.5, 0.7)
    buy_cutoff: str = "10:40"            # 路径B时间截止
    red_bar_min: float = 0.50            # 主力筹码红柱占比下限

    # ---- L4 入场形态 ----
    peak_confirm_drop: float = 0.008     # [TUNABLE] 峰值确认回落  bounds=(0.005, 0.015)
    peak_confirm_bars: int = 2           # 持续2根bar

    # ---- L2 盘中卖出 P1-P12 ----
    auction_gap_sell: float = 5.0        # [TUNABLE] P2 竞价高开全卖  bounds=(3, 8)
    hard_stop_intraday: float = -4.0     # [TUNABLE] P1 日内硬止损%  bounds=(-6, -2)
    trailing_drawdown: float = 0.04      # [TUNABLE] P11 移动止盈回撤  bounds=(0.02, 0.08)
    prob_exit: float = 0.50              # [TUNABLE] P12 概率衰减阈值  bounds=(0.4, 0.6)
    climax_move: float = 7.0             # [TUNABLE] 强势上涨判定线  bounds=(5, 9)
    climax_full_exit: float = 15.0       # 涨15%全卖 (仅20%/30%板块可达)
    climax_turnover: float = 40.0        # [TUNABLE] 高换手判定线  bounds=(30, 50)
    half_sell_move: float = 8.0          # 换手>40%且涨8%→全卖(P6)
    chip_profit_min: float = 40.0        # 获利盘筹码下限 %
    afternoon: str = "13:00"             # 下午分界
    warn_gain: float = 20.0              # [TUNABLE] 浮盈 WARNING 线  bounds=(15, 25)

    # ---- L2 日线卖出 (盘后标记, 次日执行) ----
    daily_close_lookback: int = 4        # [TUNABLE] 收盘低于N日前  bounds=(3, 5)
    profit_chip_min: float = 40.0        # 获利盘<40%
    max_hold_days: int = 3               # [TUNABLE] 持仓N日强制退出  bounds=(2, 5)

    # ---- L1 组合风控 ----
    single_position_max: float = 0.15
    industry_max: float = 0.30
    total_position_max: float = 0.95
    daily_loss_breaker: float = -0.02    # [TUNABLE] 单日亏损熔断  bounds=(-0.03, -0.01)
    dd_derisk_level1: float = -0.05      # [TUNABLE] 回撤降仓线  bounds=(-0.08, -0.03)
    dd_derisk_level2: float = -0.10      # 回撤清仓线
    cooldown_days: int = 5               # [TUNABLE] 冷静期  bounds=(3, 10)
    min_order_weight: float = 0.03


#: 可调参数搜索边界 (param_tuner 用): {字段: (下界, 上界, 步长)}
TUNABLE_BOUNDS: dict[str, tuple[float, float, float]] = {
    "control_min": (20.0, 40.0, 5.0),
    "surge_pct": (5.0, 12.0, 1.0),
    "turnover_max_entry": (30.0, 70.0, 10.0),
    "gap_open_pct": (2.0, 6.0, 1.0),
    "breadth_min": (0.50, 0.70, 0.05),
    "peak_confirm_drop": (0.005, 0.015, 0.005),
    "auction_gap_sell": (3.0, 8.0, 1.0),
    "hard_stop_intraday": (-6.0, -2.0, 1.0),
    "trailing_drawdown": (0.02, 0.08, 0.02),
    "prob_exit": (0.40, 0.60, 0.05),
    "climax_move": (5.0, 9.0, 1.0),
    "climax_turnover": (30.0, 50.0, 10.0),
    "warn_gain": (15.0, 25.0, 5.0),
    "daily_close_lookback": (3, 5, 1),
    "max_hold_days": (2, 5, 1),
    "daily_loss_breaker": (-0.03, -0.01, 0.01),
    "dd_derisk_level1": (-0.08, -0.03, 0.01),
    "cooldown_days": (3, 10, 1),
}


def board_of(code: str) -> str:
    """板块识别 (静态, 实盘当日用)."""
    code = str(code).split(".")[0]
    if code.startswith(("300", "301", "68")):
        return "GEM"
    if code.startswith(("8", "4", "92")):
        return "BSE"
    return "MAIN"


def price_limit(code: str) -> float:
    """静态涨跌幅限制. 历史回测必须用 cleaning_pipeline.get_limit_pct(board, date) 分段版."""
    return {"MAIN": Config.LIMIT_MAIN, "GEM": Config.LIMIT_GEM,
            "BSE": Config.LIMIT_BSE}[board_of(code)]
