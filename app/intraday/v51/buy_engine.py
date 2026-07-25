"""
买入规则引擎 (V5.1 §3, 8 条: 6 否决 + 2 正向, 检查清单 #5-#6/#5a/#5b)
==============================================================================
铁律 #1: trigger() 纯函数 — 无 IO / 无随机 / 无外部调用;
回测 / 看板 / 实盘同一份代码 (回测赚钱实盘亏钱的头号杀手就是双口径).

买入逻辑: 股票∈当日清单 且 时间∈买入窗口 且 B3-B8 否决链全通过
          且 (B1 或 B2 触发) 且 资金管理通过 → 下单.
下单执行纪律: 信号触发后以当前 Bar 收盘价 ±1 档限价挂单;
15 秒未成交撤单转市价; 当日该票只买一次, 绝不补仓.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost_model import CostModel, round_trip_cost

# 参数 (季度平原寻优对象; 盘中绝对禁止热更新)
CHASE_LIMIT = 0.07  # B3 追高否决: 涨幅>7% (与清单失效条件#4 口径一致, 杜绝双口径)
MIN_NET_EDGE = 0.005  # B5 净收益闸门 [0.3%/0.5%/0.8% 寻优]
BAR_AMOUNT_MIN = 1e6  # B6: bar 成交额 ≥ 100 万
ADV_BUY_RATIO = 0.01  # B6: 单笔 ≤ ADV20×1%
STOP_ATR_MULT = 1.2  # B7: 止损距离 < 1.2×ATR → 否决
SECTOR_CRASH_DROP = -0.05  # B8: 板块实时跌幅>5%
SECTOR_CRASH_COUNT = 2  # B8: 同板块 ≥2 只
LIMIT_TICK = 0.01  # ±1 档限价挂单


@dataclass(frozen=True)
class Bar:
    """5 分钟 Bar (输入数据, trigger 不自行取数)."""

    t: str
    close: float
    volume: float
    amount: float
    vwap: float = 0.0


@dataclass(frozen=True)
class BuyContext:
    """买入决策输入 (由调用方装配; trigger 纯函数不取数)."""

    symbol: str
    t: str
    price: float
    pre_close: float
    pred_q50: float  # PIPELINE1 分位数中位预测 (B5 毛利输入)
    atr_pct: float
    stop_price: float  # PIPELINE1 下发 (B7 止损距离)
    adv_20d: float
    order_value: float
    bar_amount: float  # 当前 bar 成交额
    sector_drop_count: int = 0  # 同板块实时跌幅>5% 的只数 (B8)
    event_mean: float = 0.0  # 事件研究均值 (B5 双输入取小)
    oos_decay: float = 0.0  # 样本外衰减折扣 (B5)


# ============================================================
# 否决链 B3-B8 (True = 否决)
# ============================================================
def b3_chase_filter(ctx: BuyContext) -> bool:
    """B3 追高过滤: 涨幅 > 7% → 否决 (涨停附近追入赔率极差)."""
    return ctx.price / ctx.pre_close - 1 > CHASE_LIMIT


def b4_open_noise(ctx: BuyContext) -> bool:
    """B4 开盘噪声过滤: 09:45 前禁买 (开盘 30 分钟信息未消化)."""
    return ctx.t < "09:45"


def b5_net_edge_veto(ctx: BuyContext, costs: CostModel | None = None) -> bool:
    """B5 净收益闸门: 扣全成本后净收益 < min_net_edge → 否决.

    毛利 = min(pred_q50, 事件研究均值) × (1 - 样本外衰减)  (保守原则);
    成本 = 佣金×2 + 印花税 + 分层滑点×2 + 冲击 (与 PIPELINE1 标签同口径).
    """
    gross = min(ctx.pred_q50, ctx.event_mean) * (1 - ctx.oos_decay)
    net = gross - round_trip_cost(ctx.adv_20d, ctx.order_value, costs)
    return net < MIN_NET_EDGE


def b6_liquidity_veto(ctx: BuyContext) -> bool:
    """B6 流动性约束: 单笔 > ADV20×1% 或 bar 成交额 < 100 万 → 否决."""
    if ctx.order_value > ctx.adv_20d * ADV_BUY_RATIO:
        return True
    return ctx.bar_amount < BAR_AMOUNT_MIN


def b7_stop_distance_veto(ctx: BuyContext) -> bool:
    """B7 止损距离否决【V5.1新增】: 当前价距 stop_price < 1.2×ATR → 放弃.

    止损位太近 = 噪音必扫损; E.2 动态止损前置为入场过滤.
    """
    if ctx.stop_price <= 0 or ctx.price <= 0:
        return True  # 无止损价 = 无保护, 否决
    distance = (ctx.price - ctx.stop_price) / ctx.price
    return distance < STOP_ATR_MULT * ctx.atr_pct


def b8_sector_crash_veto(ctx: BuyContext) -> bool:
    """B8 板块崩跌盘中否决【V5.1新增】: 同板块 ≥2 只实时跌幅>5% → 放弃.

    价格驱动的板块熔断 (第三触发源, 与公告/止损互补).
    """
    return ctx.sector_drop_count >= SECTOR_CRASH_COUNT


# ============================================================
# 正向规则 B1/B2
# ============================================================
def b1_vwap_pullback(
    bars: tuple[Bar, ...], pullback_pct: float = 0.01, stable_bars: int = 3
) -> bool:
    """B1 VWAP 回踩企稳: 价格回踩 VWAP 幅度内 且 连续 N 根 K 线站稳其上.

    经济学逻辑: 机构成本线有承接. 参数 [0.5%/1%/1.5%/2%] × [2/3/5] 平原寻优.
    """
    if len(bars) < stable_bars:
        return False
    recent = bars[-stable_bars:]
    for b in recent:
        if b.vwap <= 0:
            return False
        # 回踩幅度内 (不低于 vwap×(1-pullback)) 且收在 VWAP 上方 (企稳)
        if not (b.vwap * (1 - pullback_pct) <= b.close and b.close >= b.vwap * 0.998):
            return False
    return True


def b2_evening_strength(
    bars: tuple[Bar, ...], vol_ratio: float = 1.5, ma_window: int = 24
) -> bool:
    """B2 尾盘放量走强: 尾盘 bar 量比 ≥ 阈值 且 收盘价站上均线.

    经济学逻辑: 资金抢筹, 有隔夜动量. 参数 量比[1.2/1.5/2.0] × 均线[24/48].
    """
    if len(bars) < ma_window:
        return False
    last = bars[-1]
    avg_vol = sum(b.volume for b in bars[-ma_window:-1]) / (ma_window - 1)
    if avg_vol <= 0 or last.volume / avg_vol < vol_ratio:
        return False
    ma = sum(b.close for b in bars[-ma_window:]) / ma_window
    return last.close > ma


# ============================================================
# 买入决策链 (纯函数)
# ============================================================
def trigger(ctx: BuyContext, bars: tuple[Bar, ...]) -> dict:
    """买入决策链: 否决链 B3-B8 → 正向 B1/B2.

    Returns:
        {'pass': bool, 'vetoes': [...], 'positive': 'B1'/'B2'/None}
        pass=True 后由执行层按 ±1 档限价挂单, 15 秒未成交撤单转市价.
    """
    vetoes = []
    for name, fn in (
        ("B3", b3_chase_filter),
        ("B4", b4_open_noise),
        ("B5", b5_net_edge_veto),
        ("B6", b6_liquidity_veto),
        ("B7", b7_stop_distance_veto),
        ("B8", b8_sector_crash_veto),
    ):
        if fn(ctx):
            vetoes.append(name)
    positive = None
    if not vetoes:
        if b1_vwap_pullback(bars):
            positive = "B1"
        elif b2_evening_strength(bars):
            positive = "B2"
    return {
        "pass": not vetoes and positive is not None,
        "vetoes": vetoes,
        "positive": positive,
    }


def limit_order_price(bar_close: float, side: str = "buy") -> float:
    """下单执行纪律: 当前 Bar 收盘价 ±1 档限价 (15 秒未成交撤单转市价)."""
    tick = max(round(bar_close * 0.001, 2), LIMIT_TICK)  # 1 档≈0.1% 保底 0.01
    return round(bar_close + tick if side == "buy" else bar_close - tick, 2)
