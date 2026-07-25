"""
E11 熊市作战协议 (PIPELINE1_V3.8 §四 ter, 安全网 #19, 检查清单 #81-#84)
==========================================================================
设计哲学 (用户约束定稿): 熊市不买防御股、不做期权、不做T.
熊市工具箱 = 空仓纪律 + 现金管理 + 前置风控 + 禁抄底.

状态机:
  NORMAL   : 正常交易
  DEFENSE  : HS300 跌破 20 日线 且 MACD 死叉 → 强制空仓 + 逆回购/货基
             (假信号防护: 上线前回测近3年死叉假阳性率, 若空仓后10日内收复
              20日线比例>50%, 改用"跌破+连续2日确认")
  RECOVERY : 近5日 OOS Rank IC>0.02 → 复出 (首周仓位≤30%, 单票7%, 簇12%,
             流动性 ADV20×0.5%); 第二周 IC 维持>0.02 → 恢复 NORMAL

熊市硬规则 (全状态生效):
  1. 禁止抄底: 50日MA < 200日MA → 系统级冻结一切抄底/捡尸类信号
  2. 接飞刀禁令: 任何跌幅>7%的票标记雷区, 当日禁买
  3. 板块联动熔断: 同板块≥2只持仓触发止损 → 冻结该板块全部持仓与候选
  4. 准入线 bear 收紧 (联动 E7): prob_up 0.60→0.65, pred_ret 2×成本→3×成本;
     符合票可能为 0 — 这是特性不是故障
  5. 破净家数每日入库作观察指标; 严禁作为左侧加仓触发器 (约束A, 否决)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STATE_NORMAL = "NORMAL"
STATE_DEFENSE = "DEFENSE"
STATE_RECOVERY = "RECOVERY"

# E11 bear 收紧参数 (联动 E6/E7/E8)
BEAR_PARAMS = {
    "prob_entry": 0.65,  # E7 准入 prob_up 门槛 (正常 0.60)
    "ret_entry_mult": 3.0,  # E7 准入 pred_ret 门槛 = 3×COST (正常 2×)
    "single_cap": 0.07,  # 单票上限 (正常 0.10)
    "cluster_cap": 0.12,  # 簇总权重上限 (正常 0.15)
    "liquidity_ratio": 0.005,  # E6 ADV20×0.5% (正常 1%)
    "recovery_pos_cap": 0.30,  # 复出首周仓位上限
}
RECOVERY_IC_MIN = 0.02  # 复出前置: 近5日 OOS Rank IC > 0.02
KNIFE_DROP_PCT = -0.07  # 接飞刀禁令: 跌幅>7% 当日禁买
SECTOR_STOP_COUNT = 2  # 同板块≥2只止损 → 板块联动熔断


class BearProtocol:
    """E11 熊市协议状态机. 每日盘后 update() 一次."""

    def __init__(self):
        self.state = STATE_NORMAL
        self.recovery_weeks = 0  # RECOVERY 持续的周数

    # ---------------- 状态机 ----------------
    def update(
        self,
        hs300_close: pd.Series,
        macd_hist: pd.Series,
        daily_ics_5d: list[float] | None = None,
    ) -> dict:
        """输入 HS300 收盘序列 + MACD hist 序列 (含当日), 返回动作指令.

        Returns:
            {'state', 'action', ...} action ∈ FULL_EXIT / RESUME / NORMAL / HOLD
        """
        daily_ics_5d = daily_ics_5d or []
        ma20 = hs300_close.rolling(20).mean().iloc[-1]
        below_ma20 = hs300_close.iloc[-1] < ma20
        macd_dead = (
            len(macd_hist) >= 2
            and macd_hist.iloc[-1] < 0
            and macd_hist.iloc[-2] >= 0
        )
        ic_ok = len(daily_ics_5d) > 0 and float(np.mean(daily_ics_5d)) > RECOVERY_IC_MIN

        if self.state == STATE_NORMAL and below_ma20 and macd_dead:
            self.state = STATE_DEFENSE
            self.recovery_weeks = 0
            logger.warning("E11 熊市协议: NORMAL → DEFENSE, 强制空仓 + 逆回购")
            return {"state": self.state, "action": "FULL_EXIT", "cash_to": "REVERSE_REPO"}

        if self.state == STATE_DEFENSE and ic_ok:
            self.state = STATE_RECOVERY
            self.recovery_weeks = 1
            logger.warning("E11 熊市协议: DEFENSE → RECOVERY (5日IC>0.02), 复出首周")
            return {
                "state": self.state,
                "action": "RESUME",
                "pos_cap": BEAR_PARAMS["recovery_pos_cap"],
                "single_cap": BEAR_PARAMS["single_cap"],
                "cluster_cap": BEAR_PARAMS["cluster_cap"],
                "liquidity_ratio": BEAR_PARAMS["liquidity_ratio"],
            }

        if self.state == STATE_RECOVERY:
            if ic_ok and self.recovery_weeks >= 2:
                self.state = STATE_NORMAL
                self.recovery_weeks = 0
                logger.warning("E11 熊市协议: RECOVERY → NORMAL (IC 维持>0.02)")
                return {"state": self.state, "action": "NORMAL"}
            if not ic_ok:
                # IC 失守 → 退回 DEFENSE (复出失败)
                self.state = STATE_DEFENSE
                self.recovery_weeks = 0
                logger.warning("E11 熊市协议: RECOVERY → DEFENSE (IC 失守)")
                return {"state": self.state, "action": "FULL_EXIT", "cash_to": "REVERSE_REPO"}
            self.recovery_weeks += 1

        return {"state": self.state, "action": "HOLD"}

    # ---------------- bear 收紧参数出口 (联动 E6/E7/E8) ----------------
    def tightened_params(self) -> dict:
        """bear 状态 (DEFENSE/RECOVERY) 下的收紧参数; NORMAL 返回空 dict (用默认)."""
        if self.state == STATE_NORMAL:
            return {}
        return dict(BEAR_PARAMS)

    # ---------------- 熊市硬规则 (全状态生效, 静态判定) ----------------
    @staticmethod
    def is_downtrend(ma50: float, ma200: float) -> bool:
        """禁止抄底: 50日MA < 200日MA → 下跌趋势, 冻结一切抄底/捡尸类信号."""
        return bool(ma50 < ma200)

    @staticmethod
    def knife_catching_ban(pct_change: float) -> bool:
        """接飞刀禁令: 跌幅 > 7% 标记雷区, 当日禁买 (True=禁买)."""
        return bool(pct_change <= KNIFE_DROP_PCT)

    @staticmethod
    def sector_fuse(stop_symbols_by_sector: dict[str, int]) -> set[str]:
        """板块联动熔断: 同板块 ≥2 只持仓触发止损 → 冻结该板块全部持仓与候选.

        Args:
            stop_symbols_by_sector: {行业: 当日止损只数}
        Returns:
            被冻结的行业集合 (与公告驱动的板块冻结互补, 本条为止损驱动).
        """
        frozen = {
            sec for sec, n in stop_symbols_by_sector.items() if n >= SECTOR_STOP_COUNT
        }
        if frozen:
            logger.error("E11 板块联动熔断: 冻结 %s", sorted(frozen))
        return frozen
