"""
时段定义 (V5.1 §2, 单点引用, 杜绝双口径; 检查清单 #11)
==========================================================
买入只在两个窗口内执行; 盘中 (10:30-14:30) 只卖不买;
熊市协议状态优先级高于一切时段定义:
  DEFENSE  : 只卖不买 (buy DISABLED)
  RECOVERY : 早盘窗口关闭 + 仓位上限 30% (首周)
满仓纪律 #2: 尾盘窗口 (14:30-14:55) 为主窗口 (与 PIPELINE1 14:55+滑点口径一致);
早盘窗口 (09:45-10:30) 仅允许 A级信号 且 大盘非单边下跌 (hs300 当日跌幅<1%).
"""

from __future__ import annotations

NO_BUY_UNTIL = "09:45"  # 开盘噪声期, 禁止买入
MORNING_BUY = ("09:45", "10:30")  # 早盘窗口 (受限)
EVENING_BUY = ("14:30", "14:55")  # 尾盘窗口 (主窗口)
SELL_WINDOW = ("14:30", "14:55")  # 尾盘强制卖出窗口
CLOSE = "15:00"
MORNING_HS300_DROP_LIMIT = -0.01  # 早盘窗口: hs300 跌幅 ≥1% 禁用
RECOVERY_POSITION_CAP = 0.30  # RECOVERY 仓位上限 30% (首周, 本地强制)


def position_cap(bear_state: str, base_cap: float = 1.0) -> float:
    """熊市协议下的仓位上限 (V5.1 §2).

    Args:
        bear_state: DEFENSE / RECOVERY / NORMAL
        base_cap: 基础仓位上限 (攻击档默认 1.0; B 档 0.75 等)

    Returns:
        DEFENSE → 0 (只卖不买)
        RECOVERY → min(base_cap, 0.30)
        NORMAL  → base_cap
    """
    if bear_state == "DEFENSE":
        return 0.0
    if bear_state == "RECOVERY":
        return min(base_cap, RECOVERY_POSITION_CAP)
    return base_cap


def _in_window(t: str, window: tuple[str, str]) -> bool:
    return window[0] <= t <= window[1]


def buy_window_open(
    t: str,
    bear_state: str = "NORMAL",
    signal_grade: str = "A",
    hs300_change: float = 0.0,
) -> bool:
    """当前时刻是否允许买入 (时段 + 熊市接管 + 满仓纪律#2).

    Args:
        t: "HH:MM"
        bear_state: 熊市协议状态 (优先级高于一切时段定义)
        signal_grade: 信号等级 (早盘窗口仅 A 级)
        hs300_change: hs300 当日涨跌幅 (早盘 单边下跌 ≥1% 禁用)
    """
    if bear_state == "DEFENSE":
        return False  # 只卖不买
    if _in_window(t, EVENING_BUY):
        return True  # 尾盘主窗口 (NORMAL/RECOVERY 均开)
    if _in_window(t, MORNING_BUY):
        if bear_state == "RECOVERY":
            return False  # RECOVERY 首周早盘关闭
        return signal_grade == "A" and hs300_change > MORNING_HS300_DROP_LIMIT
    return False  # 开盘噪声期 / 盘中只卖不买


def sell_window_open(t: str) -> bool:
    """尾盘强制卖出窗口 (时间止损/持仓到期/调出在此执行)."""
    return _in_window(t, SELL_WINDOW)
