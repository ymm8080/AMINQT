"""
卖出规则引擎 (V5.1 §4, 8 条按优先级排序, 优先级不可调整, 检查清单 #7-#10)
==============================================================================
铁律 #1: trigger() 纯函数 (无 IO/无随机/无外部调用).
关键约束: 所有卖出规则强制校验 sellable_qty > 0 (T+1, 铁律 #0);
S8 跌停挂单若全日未成交, 次日继续挂, 直至成交, 每日记录 WORM.

卖出决策链 (1 分钟轮询):
  前置 sellable_qty>0 → S8 跌停逃生 → S3 换手异动 → S6 涨停炸板
  → S1 动态止损 (stop_price 下发制) → S2 自适应移动止盈
  → S5a 时间止损 / S5b 持仓到期 → S7/S4 调出/失效 → 继续持有
"""

from __future__ import annotations

from dataclasses import dataclass

from .position import Position
from .safe_div import safe_divide
from .sessions import sell_window_open

# 参数 (季度平原寻优; S1 stop_price 不在寻优范围 — PIPELINE1 动态下发)
S3_TURNOVER = 0.40  # S3: 换手率 > 40%
S3_CHANGE = 0.08  # S3: 且涨幅 > 8% (主力出货)
S6_OPEN_FALLBACK = 0.015  # S6: 涨停开板回落 > 1.5%
S2_ACTIVATE = 0.03  # S2: 浮盈 ≥3% 激活
S2_TRAIL_BASE = 0.03  # S2: 回撤带 max(3%, 1.0×ATR)
S2_ATR_MULT = 1.0
S5A_DAYS, S5A_MIN_RET = 2, 0.01  # S5a: 满 2 日涨幅 <1%
S5B_DAYS = 2  # S5b: 持仓满 2 个交易日 (强制轮动)


@dataclass(frozen=True)
class SellContext:
    """卖出决策输入 (调用方装配; trigger 纯函数不取数)."""

    t: str
    price: float
    limit_down_price: float  # 当日跌停价 (S8)
    limit_up_price: float = 0.0  # 当日涨停价 (S6)
    turnover_pct: float = 0.0  # 当日换手率 (S3)
    change_pct: float = 0.0  # 当日涨跌幅 (S3)
    touched_limit_up: bool = False  # 盘中曾涨停 (S6)
    atr_pct: float = 0.02
    invalidation: bool = False  # 清单失效条件下发 (S7)
    model_removed: bool = False  # 日线模型调出 (S4)


# ============================================================
# 单规则 (True = 触发; 全部需调用方保证 sellable_qty>0 前置)
# ============================================================
def s8_limit_escape(ctx: SellContext) -> bool:
    """S8 跌停逃生【最高优先级】: 持仓跌停无法成交 → 次日 09:25 集合竞价
    挂跌停价排队, 优先于一切其他信号; 未成交次日继续挂, 每日 WORM 记录.
    流动性风险 > 价格风险.
    """
    return ctx.limit_down_price > 0 and ctx.price <= ctx.limit_down_price + 1e-9


def s3_turnover_spike(ctx: SellContext) -> bool:
    """S3 换手异动: 换手 >40% 且 涨幅 >8% → 主力出货信号, 立即清仓."""
    return ctx.turnover_pct > S3_TURNOVER and ctx.change_pct > S3_CHANGE


def s6_limit_break(ctx: SellContext) -> bool:
    """S6 涨停炸板: 盘中触板后开板回落 >1.5% → 止盈出局 (A股特有场景)."""
    if not ctx.touched_limit_up or ctx.limit_up_price <= 0:
        return False
    return ctx.price < ctx.limit_up_price * (1 - S6_OPEN_FALLBACK)


def s1_dynamic_stop(ctx: SellContext, pos: Position) -> bool:
    """S1 动态止损【V5.1重构】: 现价 ≤ stop_price (PIPELINE1 每日下发).

    stop = max(1.2×ATR, target/2); 高波动票止损自动放宽,
    仓位已在 PIPELINE1 侧同步缩小. 废除全局固定 -4%.
    """
    return pos.stop_price > 0 and ctx.price <= pos.stop_price


def s2_trailing_stop(ctx: SellContext, pos: Position) -> bool:
    """S2 自适应移动止盈【V5.1重构】: 浮盈 ≥3% 激活;
    从最高价回撤 max(3%, 1.0×ATR_pct) — 回撤带随波动率伸缩.
    """
    if pos.entry_price <= 0 or pos.max_price_since_entry <= pos.entry_price:
        return False
    profit = safe_divide(pos.max_price_since_entry, pos.entry_price) - 1
    if profit < S2_ACTIVATE:
        return False
    band = max(S2_TRAIL_BASE, S2_ATR_MULT * ctx.atr_pct)
    return ctx.price <= pos.max_price_since_entry * (1 - band)


def s5a_time_stop(ctx: SellContext, pos: Position) -> bool:
    """S5a 时间止损【V5.1新增】: 买入满 2 日且累计涨幅 <1% → 尾盘卖出.

    不达预期不陪耗 (亏得干脆, 赚得耐心).
    """
    if pos.hold_days < S5A_DAYS or pos.entry_price <= 0:
        return False
    ret = safe_divide(ctx.price, pos.entry_price) - 1
    return ret < S5A_MIN_RET and sell_window_open(ctx.t)


def s5b_expiry(ctx: SellContext, pos: Position) -> bool:
    """S5b 持仓到期: 持有满 2 个交易日 → 强制轮动, 尾盘卖出."""
    return pos.hold_days >= S5B_DAYS and sell_window_open(ctx.t)


def s7_s4_remove(ctx: SellContext) -> bool:
    """S7/S4 雷区禁令与日线调出: 清单失效条件下发 / 日线模型调出 → 尾盘卖出."""
    return (ctx.invalidation or ctx.model_removed) and sell_window_open(ctx.t)


# ============================================================
# 卖出决策链 (优先级不可调整)
# ============================================================
def trigger(ctx: SellContext, pos: Position) -> dict:
    """卖出决策链. 前置: sellable_qty > 0 (T+1 物理校验, 不通过整条链跳过).

    Returns:
        {'action': 'SELL'/'AUCTION_SELL'/'HOLD', 'rule': str|None,
         'qty': int, 'reason': str}
        AUCTION_SELL = S8 集合竞价排队 (次日 09:25 挂跌停价, 优先于一切).
    """
    if not pos.can_sell():
        return {
            "action": "HOLD",
            "rule": None,
            "qty": 0,
            "reason": "T+1锁死: sellable_qty=0",
        }
    qty = pos.sellable_qty
    if s8_limit_escape(ctx):
        return {
            "action": "AUCTION_SELL",
            "rule": "S8",
            "qty": qty,
            "reason": "跌停逃生: 次日09:25集合竞价挂跌停价排队",
        }
    if s3_turnover_spike(ctx):
        return {
            "action": "SELL",
            "rule": "S3",
            "qty": qty,
            "reason": f"换手异动 {ctx.turnover_pct:.0%} 且涨 {ctx.change_pct:.1%}",
        }
    if s6_limit_break(ctx):
        return {
            "action": "SELL",
            "rule": "S6",
            "qty": qty,
            "reason": "涨停炸板回落>1.5%",
        }
    if s1_dynamic_stop(ctx, pos):
        return {
            "action": "SELL",
            "rule": "S1",
            "qty": qty,
            "reason": f"动态止损: 现价≤stop_price({pos.stop_price})",
        }
    if s2_trailing_stop(ctx, pos):
        return {
            "action": "SELL",
            "rule": "S2",
            "qty": qty,
            "reason": "移动止盈 (ATR自适应回撤带)",
        }
    if s5a_time_stop(ctx, pos):
        return {
            "action": "SELL",
            "rule": "S5a",
            "qty": qty,
            "reason": "时间止损: 满2日涨幅<1%",
        }
    if s5b_expiry(ctx, pos):
        return {
            "action": "SELL",
            "rule": "S5b",
            "qty": qty,
            "reason": f"持仓满{S5B_DAYS}日强制轮动",
        }
    if s7_s4_remove(ctx):
        return {
            "action": "SELL",
            "rule": "S7/S4",
            "qty": qty,
            "reason": "清单失效/日线调出",
        }
    return {"action": "HOLD", "rule": None, "qty": 0, "reason": "未触发"}
