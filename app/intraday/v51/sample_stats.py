"""40 笔样本统计 (P21.1, D.5/D.10 — 日内侧单点转出口)
======================================================================
实施计划 v3.1 P21.1 点名本路径; 统计逻辑单点维护于
`app.pipeline1.trade_discipline` (TradeRecord/TradeJournal), 此处只做
同源转出, 禁止另写第二份统计逻辑 (双口径 = 回测赚钱实盘亏钱的头号杀手).

D.5 样本积累纪律: 攻击档前 2-3 个月目标是买数据 (40-60 笔), 不是买收益;
  满 40 笔计算真实胜率/盈亏比回喂参数; 样本不足禁止凭感觉调参 (安全网#15).
D.10 实盘解锁双闸门: 前 40 笔按 B 档执行; 40 笔满足 期望>+0.5%/笔 且
  最大连亏≤5 笔, 方可切 C 档 (回测不覆盖执行风险, 必须用实盘数据二次确认).
"""

from __future__ import annotations

from app.pipeline1.trade_discipline import (
    UNLOCK_MAX_CONSEC_LOSS,
    UNLOCK_MIN_EXPECTANCY,
    UNLOCK_MIN_TRADES,
    TradeJournal,
    TradeRecord,
)

__all__ = [
    "UNLOCK_MAX_CONSEC_LOSS",
    "UNLOCK_MIN_EXPECTANCY",
    "UNLOCK_MIN_TRADES",
    "TradeJournal",
    "TradeRecord",
]
