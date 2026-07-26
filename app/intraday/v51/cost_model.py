"""
成本模型 (V5.1 §5, 与 PIPELINE1 同口径, 单点维护, 检查清单 #12)
====================================================================
纪律: 成本参数只允许在本文件维护; PIPELINE1 标签、回测、本系统 B5
三处引用同一源 (slippage_tier 直接 import 自 PIPELINE1 标签引擎).
改成本 = 改三处结果, 必须重新回测.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.pipeline1.label_engine import slippage_tier  # 同源单点 (E5 分层)

__all__ = ["CostModel", "slippage_tier", "round_trip_cost"]


@dataclass(frozen=True)
class CostModel:
    """V5.1 成本参数 (frozen: 盘中绝对禁止热更新)."""

    commission: float = 0.00025  # 佣金万2.5 (可谈至万2以下, 谈判成果直接改善B5通过率)
    stamp_tax_sell: float = 0.0005  # 印花税万5 (固定)
    impact_coef: float = 0.5  # 冲击成本系数 (√(order/ADV20) 模型)


def round_trip_cost(
    adv_20d: float, order_value: float, costs: CostModel | None = None
) -> float:
    """round-trip 总成本率 = 佣金×2 + 印花税 + 滑点×2 + 冲击成本.

    impact = impact_coef × √(order_value / ADV20)
    (B6 说明: 小资金阶段冲击≈0, 资金增长后自动生效, 无需改代码).
    """
    c = costs or CostModel()
    if adv_20d <= 0:
        return float("inf")  # 无流动性数据 → 成本无穷大 (B5/B6 否决保护)
    slip = slippage_tier(adv_20d)
    impact = c.impact_coef * (order_value / adv_20d) ** 0.5
    return 2 * c.commission + c.stamp_tax_sell + 2 * slip + impact
