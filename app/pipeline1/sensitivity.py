"""2 倍滑点敏感性门禁 (P19.1 阶段二 W5/W8, 检查点: 净超额≥5%才过门禁)
======================================================================
点火验证通过标准之一: 2 倍滑点下净年化超额 ≥ 5%.
滑点假设翻倍后策略仍赚钱, 才说明超额不是成本模型误差的幻觉.

引擎侧已支持 `BacktestConfig.slippage_multiplier=2` (backtest_v35),
本模块是裁决层: 同一回测跑 1x/2x 两组, 对比扣费后净年化超额
(net_excess_annual, 与 metrics 验收口径一致).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MIN_EXCESS_2X = 0.05  # 门禁: 2 倍滑点下净年化超额 ≥ 5% (P19.1 通过标准)


def slippage_sensitivity_verdict(excess_1x: float, excess_2x: float) -> dict:
    """2 倍滑点敏感性门禁.

    Args:
        excess_1x: 1 倍滑点扣费后净年化超额 (net_excess_annual)
        excess_2x: 2 倍滑点扣费后净年化超额 (同口径)

    Returns:
        {'pass': bool, 'excess_1x', 'excess_2x', 'erosion',
         'reason': str}  erosion = 成本敏感性侵蚀幅度 (1x-2x).
    """
    excess_1x, excess_2x = float(excess_1x), float(excess_2x)
    ok = excess_2x >= MIN_EXCESS_2X
    erosion = excess_1x - excess_2x
    if not ok:
        logger.critical(
            "P19.1 敏感性否决: 2倍滑点净年化超额 %.2f%% < %.0f%% "
            "(1x=%.2f%%, 侵蚀 %.2f%%) — 超额可能是成本模型误差的幻觉",
            excess_2x * 100,
            MIN_EXCESS_2X * 100,
            excess_1x * 100,
            erosion * 100,
        )
    return {
        "pass": ok,
        "excess_1x": round(excess_1x, 4),
        "excess_2x": round(excess_2x, 4),
        "erosion": round(erosion, 4),
        "reason": (
            f"2x净超额{excess_2x:.2%} ≥ {MIN_EXCESS_2X:.0%}"
            if ok
            else f"2x净超额{excess_2x:.2%} < {MIN_EXCESS_2X:.0%}, 一票否决"
        ),
    }
