# -*- coding: utf-8 -*-
"""同花顺(THS)自定义量价指标 → tech_ths_* 因子列.

所有指标均融合 **成交量 + 价格**（用户要求：只用量价指标）。
原版同花顺公式中 3 个指标是纯价格 (OHLC) 的，本实现按以下方式
注入成交量维度：

1. **主力筹码指标**：进场/拉高/出货信号 × 量比因子
2. **发现牛股**：SS 买入信号增加放量条件 (VOL > 1.5 × MA(VOL,5))
3. **益盟趋势顶底**：增加量价背离检测 (价创新低+量缩=底背离)
4. **主力筹码控盘程度N**：已通过 WINNER() 近似使用成交量

另新增 5 个量价派生特征：量比 / VWAP偏离 / OBV斜率 / 量价相关 / 量加权动量。

References
----------
* ``reference/indicator/主力筹码指标.docx``
* ``reference/indicator/主力筹码控盘程度N.docx``
* ``reference/indicator/发现牛股.docx``
* ``reference/indicator/同花顺益盟趋势顶底新公式.docx``
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)

# ── 公共参数 ────────────────────────────────────────────────────────
DECAY_TAU = 10  # 二值信号指数衰减半衰期（天）
WINNER_LOOKBACK = 60  # WINNER() 近似回看窗口（天）
VOL_RATIO_CAP = 3.0  # 量比封顶（防 outlier）
BULL_VOL_MULT = 1.5  # 发现牛股放量阈值倍数

# ════════════════════════════════════════════════════════════════════
#  因子列名清单（45 列 = 10 + 4 + 4 + 6 + 5 量价派生 + 1 背离 + 8 资金流向 + 3 控盘增强 + 4 筹码分布）
# ════════════════════════════════════════════════════════════════════

THS_FACTOR_COLUMNS: List[str] = [
    # ── 主力筹码指标（量价融合）──
    "tech_ths_trajectory",  # 主力轨迹
    "tech_ths_mazl",  # MA(轨迹, 5)
    "tech_ths_entry",  # 主力进场（×量比）
    "tech_ths_washout",  # 洗盘（×量比）
    "tech_ths_pullup",  # 主力拉高（×量比）
    "tech_ths_ship",  # 出货（×量比）
    "tech_ths_entry_flag_decay10",  # 进场信号（衰减）
    "tech_ths_pullup_flag_decay10",  # 拉高信号（衰减）
    "tech_ths_ship_flag_decay10",  # 出货信号（衰减）
    "tech_ths_golden_cross_decay10",  # 轨迹金叉（衰减）
    # ── 主力筹码控盘程度N ──
    "tech_ths_ctrl_low",  # 低价区筹码%
    "tech_ths_ctrl_mid",  # 中间区筹码宽度
    "tech_ths_ctrl_high",  # 高价区筹码%
    "tech_ths_ctrl_flag_decay10",  # 控盘信号（衰减）
    # ── 发现牛股（量价融合）──
    "tech_ths_ema3_dev_ema20",  # EMA3/EMA20-1
    "tech_ths_ema7_dev_ema20",  # EMA7/EMA20-1
    "tech_ths_ema12_dev_ema50",  # EMA12/EMA50-1
    "tech_ths_bull_ss_decay10",  # SS 买入信号（含放量确认）
    # ── 益盟趋势顶底（量价融合）──
    "tech_ths_trend_short",  # 短期线 (0-200)
    "tech_ths_trend_mid",  # 中期线 (0-200)
    "tech_ths_trend_long",  # 长期线 (0-200)
    "tech_ths_trend_top_decay10",  # 见顶信号（衰减）
    "tech_ths_trend_bottom_decay10",  # 底部区域信号（衰减）
    "tech_ths_trend_golden_decay10",  # 低位金叉信号（衰减）
    "tech_ths_vol_price_divergence",  # 量价背离（+底背离 / -顶背离）
    # ── 量价派生特征 ──
    "tech_ths_vol_ratio",  # 量比 = V / MA(V,20)
    "tech_ths_vwap_dev",  # VWAP 偏离度 = close/vwap - 1
    "tech_ths_obv_slope",  # OBV 5日斜率
    "tech_ths_vol_price_corr",  # 5日 收益率-成交量变化 相关系数
    "tech_ths_vol_weighted_mtm",  # 量加权动量
    # ── 主力资金流向（新增 8 列）──
    "tech_ths_flow_net",  # 主力净流入估算 = (close-open)*volume
    "tech_ths_flow_net_ma5",  # 5日主力净流入均值
    "tech_ths_flow_net_ma20",  # 20日主力净流入均值
    "tech_ths_flow_ratio",  # 主力净流入比率 = flow_net / (close*volume)
    "tech_ths_flow_accum",  # 主力净流入累积（归一化）
    "tech_ths_flow_divergence",  # 资金流向背离 (+底背离 / -顶背离)
    "tech_ths_flow_strength",  # 资金强度 = |flow_net| / ((high-low)*volume)
    "tech_ths_flow_trend",  # 资金流向趋势 (5日斜率)
    # ── 控盘增强（新增 3 列）──
    "tech_ths_ctrl_ratio",  # 控盘比例 = ctrl_low / (ctrl_low+ctrl_mid+ctrl_high)
    "tech_ths_ctrl_concentration",  # 筹码集中度 = 1 - ctrl_mid/100
    "tech_ths_ctrl_change",  # 控盘比例变化 = ctrl_ratio - ctrl_ratio.shift(5)
    # ── 筹码分布增强（新增 4 列）──
    "tech_ths_chip_profit_ratio",  # 获利盘比例 (WINNER近似, 0~1)
    "tech_ths_chip_concentration_20",  # 20日筹码集中度
    "tech_ths_chip_cost_skew",  # 筹码成本偏态
    "tech_ths_chip_low_high_ratio",  # 低高价区筹码比
]


# ════════════════════════════════════════════════════════════════════
#  公共工具函数
# ════════════════════════════════════════════════════════════════════


def safe_divide(numerator, denominator) -> pd.Series:
    """除零安全除法：分母为 0 时返回 0."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.divide(num, den, out=np.zeros_like(num), where=den != 0)
    return pd.Series(out, index=getattr(numerator, "index", None))


def ths_ema(series: pd.Series, n: int) -> pd.Series:
    """同花顺 EMA(X, N) → ewm(span=N, adjust=False)."""
    return series.ewm(span=n, adjust=False).mean()


def ths_sma(series: pd.Series, n: int, m: int) -> pd.Series:
    """同花顺 SMA(X, N, M) → ewm(alpha=M/N, adjust=False)."""
    return series.ewm(alpha=m / n, adjust=False).mean()


def ths_ref(series: pd.Series, n: int) -> pd.Series:
    """同花顺 REF(X, n) — n 天前的值."""
    return series.shift(n)


def ths_hhv(series: pd.Series, n: int) -> pd.Series:
    """同花顺 HHV(X, N) — N 周期最高."""
    return series.rolling(n, min_periods=1).max()


def ths_llv(series: pd.Series, n: int) -> pd.Series:
    """同花顺 LLV(X, N) — N 周期最低."""
    return series.rolling(n, min_periods=1).min()


def ths_cross(a: pd.Series, b: pd.Series) -> pd.Series:
    """同花顺 CROSS(A, B) — A 上穿 B."""
    return (a > b) & (a.shift(1) <= b.shift(1))


def exp_decay_encode(flag: pd.Series, tau: int = DECAY_TAU) -> pd.Series:
    """二值信号 → 指数衰减编码.

    aurumq-rl Phase 24A 教训：原始 0/1 binary 进 LayerNorm 后
    z-score 把 0/1 拉成极端 outlier，T-1 hit 从 2.11% 跌到 0.40%。
    Phase 26F 修复：改用指数衰减 τ=10d 编码后 T-1 hit 反弹到 2.27%。
    """
    f = flag.astype(float)
    kernel_len = tau * 5
    weights = np.exp(-np.arange(kernel_len) / tau)
    weights /= weights.sum()
    result = f.rolling(kernel_len, min_periods=1).apply(
        lambda x: np.dot(x, weights[-len(x) :]), raw=True
    )
    return result.fillna(0.0)


def _approx_winner(
    price: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    lookback: int = WINNER_LOOKBACK,
) -> pd.Series:
    """近似 WINNER(price) — 成交量加权获利盘比例 (0-100)."""
    typical_price = (high + low + close) / 3.0

    def _winner_at_row(idx):
        if idx < 1:
            return 0.0
        start = max(0, idx - lookback)
        tp_window = typical_price.iloc[start : idx + 1].values
        vol_window = volume.iloc[start : idx + 1].values
        p = price.iloc[idx]
        if np.nansum(vol_window) == 0:
            return 0.0
        mask = tp_window <= p
        return 100.0 * np.nansum(vol_window[mask]) / np.nansum(vol_window)

    return pd.Series(
        [_winner_at_row(i) for i in range(len(price))], index=price.index, dtype=float
    )


def _compute_vol_ratio(volume: pd.Series, ma_period: int = 20) -> pd.Series:
    """量比 = 当日成交量 / 过去 N 日均量，封顶 VOL_RATIO_CAP."""
    ma_vol = volume.rolling(ma_period, min_periods=1).mean()
    ratio = safe_divide(volume, ma_vol)
    return ratio.clip(upper=VOL_RATIO_CAP).fillna(0.0)


# ════════════════════════════════════════════════════════════════════
#  指标 1: 主力筹码指标 — 量价融合版
# ════════════════════════════════════════════════════════════════════


def compute_main_force_chip(df: pd.DataFrame) -> pd.DataFrame:
    """主力筹码指标（量价融合）.

    原公式只看价格突破(LLV/HHV)。本版将进场/拉高/出货信号 × 量比，
    使"放量突破"的信号强度远大于"缩量突破"。

    输出 10 列。
    """
    c, o = df["close"], df["open"]
    h, l = df["high"], df["low"]
    vol = df["volume"]

    N1, N2 = 9, 5
    vol_ratio = _compute_vol_ratio(vol, 20)

    # ── 主力轨迹 & MAZL ──
    mtm = c - ths_ref(c, 1)
    mtm_ema = ths_ema(ths_ema(mtm, N1), N1)
    abs_mtm_ema = ths_ema(ths_ema(mtm.abs(), N1), N1)
    df["tech_ths_trajectory"] = 100.0 * safe_divide(mtm_ema, abs_mtm_ema)
    df["tech_ths_mazl"] = df["tech_ths_trajectory"].rolling(N2, min_periods=1).mean()

    # ── 主力进场 / 洗盘 (基于 LOW) ──
    var1 = ths_ref((l + o + c + h) / 4.0, 1)
    var2 = safe_divide(
        ths_sma((l - var1).abs(), 13, 1), ths_sma((l - var1).clip(lower=0), 10, 1)
    )
    var3 = ths_ema(var2, 10)
    var4 = ths_llv(l, 33)
    var5_raw = pd.Series(np.where(l <= var4, var3, 0.0), index=df.index, dtype=float)
    var5 = ths_ema(var5_raw, 3)

    # 量价融合：信号 × 量比
    df["tech_ths_entry"] = (
        pd.Series(
            np.where(var5 > ths_ref(var5, 1), var5, 0.0), index=df.index, dtype=float
        )
        * vol_ratio
    )
    df["tech_ths_washout"] = (
        pd.Series(
            np.where(var5 < ths_ref(var5, 1), var5, 0.0), index=df.index, dtype=float
        )
        * vol_ratio
    )

    # ── 主力拉高 / 出货 (基于 HIGH) ──
    var21 = safe_divide(
        ths_sma((h - var1).abs(), 13, 1), ths_sma((h - var1).clip(upper=0), 10, 1)
    )
    var31 = ths_ema(var21, 10)
    var41 = ths_hhv(h, 33)
    var51_raw = pd.Series(np.where(h >= var41, var31, 0.0), index=df.index, dtype=float)
    var51 = ths_ema(var51_raw, 3)

    # 量价融合：信号 × 量比
    df["tech_ths_pullup"] = (
        pd.Series(
            np.where(var51 < ths_ref(var51, 1), var51, 0.0), index=df.index, dtype=float
        )
        * vol_ratio
    )
    df["tech_ths_ship"] = (
        pd.Series(
            np.where(var51 > ths_ref(var51, 1), var51, 0.0), index=df.index, dtype=float
        )
        * vol_ratio
    )

    # ── 衰减信号 ──
    df["tech_ths_entry_flag_decay10"] = exp_decay_encode(df["tech_ths_entry"] > 0)
    df["tech_ths_pullup_flag_decay10"] = exp_decay_encode(df["tech_ths_pullup"] > 0)
    df["tech_ths_ship_flag_decay10"] = exp_decay_encode(df["tech_ths_ship"] > 0)
    df["tech_ths_golden_cross_decay10"] = exp_decay_encode(
        ths_cross(df["tech_ths_trajectory"], df["tech_ths_mazl"])
    )

    logger.debug("主力筹码指标(量价): 10 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  指标 2: 主力筹码控盘程度N
# ════════════════════════════════════════════════════════════════════


def compute_chip_control(df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """主力筹码控盘程度N（已含成交量，通过 WINNER 近似）.

    输出 4 列。
    """
    c, o = df["close"], df["open"]
    h, l = df["high"], df["low"]
    vol = df["volume"]

    a01 = (c + o + l + h) / 4.0
    a02 = _approx_winner(a01 * 1.04, c, h, l, vol)
    a03 = _approx_winner(a01 * 0.96, c, h, l, vol)

    df["tech_ths_ctrl_low"] = a03
    df["tech_ths_ctrl_mid"] = a02 - a03
    df["tech_ths_ctrl_high"] = 100.0 - a02

    a04_hhv = ths_hhv(a03, 15)
    a0a_cond = (safe_divide(a04_hhv - a03, a03) * 100 > n) & (a04_hhv > 50)
    df["tech_ths_ctrl_flag_decay10"] = exp_decay_encode(a0a_cond)

    logger.debug("主力筹码控盘程度N: 4 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  指标 3: 发现牛股 — 量价融合版
# ════════════════════════════════════════════════════════════════════


def compute_bull_finder(df: pd.DataFrame) -> pd.DataFrame:
    """发现牛股（量价融合）.

    原公式 SS 信号只看价格交叉。本版增加放量条件：
    VOL > BULL_VOL_MULT × MA(VOL, 5)

    输出 4 列。
    """
    c, o = df["close"], df["open"]
    vol = df["volume"]

    a1 = ths_ema(c, 3)
    a3 = ths_ema(c, 7)
    a4 = ths_ema(c, 12)
    a5 = ths_ema(c, 20)
    a6 = ths_ema(c, 50)

    df["tech_ths_ema3_dev_ema20"] = safe_divide(a1, a5) - 1.0
    df["tech_ths_ema7_dev_ema20"] = safe_divide(a3, a5) - 1.0
    df["tech_ths_ema12_dev_ema50"] = safe_divide(a4, a6) - 1.0

    # SS 原始信号
    cross_signal = ths_cross(a1, a5)
    c_gt_o = c > o
    c_gt_ref = c > ths_ref(c, 1)
    c_ratio = safe_divide(c, ths_ref(c, 1))
    ss_price = cross_signal & c_gt_o & c_gt_ref & (c_ratio >= 1.018)

    # 量价融合：放量确认
    ma_vol5 = vol.rolling(5, min_periods=1).mean()
    vol_surge = vol > BULL_VOL_MULT * ma_vol5
    ss = ss_price & vol_surge

    df["tech_ths_bull_ss_decay10"] = exp_decay_encode(ss)

    logger.debug("发现牛股(量价): 4 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  指标 4: 益盟趋势顶底 — 量价融合版
# ════════════════════════════════════════════════════════════════════


def compute_trend_top_bottom(df: pd.DataFrame) -> pd.DataFrame:
    """益盟趋势顶底（量价融合）.

    原公式是纯价格的 William%R 变体。本版新增量价背离列：
    - 底背离：价创 20 日新低 + 量 < MA(VOL,20) → +1
    - 顶背离：价创 20 日新高 + 量 < MA(VOL,20) → -1
    - 正常：0

    输出 6 列。
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    vol = df["volume"]

    # ── 三条趋势线（原公式）──
    hhv14 = ths_hhv(h, 14)
    llv14 = ths_llv(l, 14)
    b = 100.0 * safe_divide(c - hhv14, hhv14 - llv14)

    hhv34 = ths_hhv(h, 34)
    llv34 = ths_llv(l, 34)
    raw34 = 100.0 * safe_divide(c - hhv34, hhv34 - llv34)
    d = ths_ema(raw34, 4)
    a = raw34.rolling(19, min_periods=1).mean()

    short_line = b + 100.0
    mid_line = d + 100.0
    long_line = a + 100.0

    df["tech_ths_trend_short"] = short_line
    df["tech_ths_trend_mid"] = mid_line
    df["tech_ths_trend_long"] = long_line

    # ── 见顶信号 ──
    top_cond = (
        (ths_ref(mid_line, 1) > 85)
        & (ths_ref(short_line, 1) > 85)
        & (ths_ref(long_line, 1) > 65)
        & ths_cross(long_line, short_line)
    )
    df["tech_ths_trend_top_decay10"] = exp_decay_encode(top_cond)

    # ── 底部区域信号 ──
    bottom_cond = (
        (
            (long_line < 12)
            & (mid_line < 8)
            & ((short_line < 7.2) | (ths_ref(short_line, 1) < 5))
            & (
                (mid_line > ths_ref(mid_line, 1))
                | (short_line > ths_ref(short_line, 1))
            )
        )
        | (
            (long_line < 8)
            & (mid_line < 7)
            & (short_line < 15)
            & (short_line > ths_ref(short_line, 1))
        )
        | ((long_line < 10) & (mid_line < 7) & (short_line < 1))
    )
    df["tech_ths_trend_bottom_decay10"] = exp_decay_encode(bottom_cond)

    # ── 低位金叉信号 ──
    golden_cond = (
        (long_line < 15)
        & (ths_ref(long_line, 1) < 15)
        & (mid_line < 18)
        & (short_line > ths_ref(short_line, 1))
        & ths_cross(short_line, long_line)
        & (short_line > mid_line)
        & ((ths_ref(short_line, 1) < 5) | (ths_ref(short_line, 2) < 5))
        & ((mid_line >= long_line) | (ths_ref(short_line, 1) < 1))
    )
    df["tech_ths_trend_golden_decay10"] = exp_decay_encode(golden_cond)

    # ── 量价背离（新增）──
    # 底背离：价创20日新低 + 量缩（VOL < MA(VOL,20)）→ +1
    # 顶背离：价创20日新高 + 量缩 → -1
    price_new_low = c <= ths_llv(c, 20)
    price_new_high = c >= ths_hhv(c, 20)
    ma_vol20 = vol.rolling(20, min_periods=1).mean()
    vol_shrink = vol < ma_vol20

    divergence = pd.Series(0.0, index=df.index)
    divergence[price_new_low & vol_shrink] = 1.0  # 底背离
    divergence[price_new_high & vol_shrink] = -1.0  # 顶背离
    df["tech_ths_vol_price_divergence"] = divergence

    logger.debug("益盟趋势顶底(量价): 6 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  量价派生特征
# ════════════════════════════════════════════════════════════════════


def compute_vol_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """5 个量价派生特征.

    - vol_ratio: 量比 = V / MA(V,20)
    - vwap_dev: close / VWAP - 1
    - obv_slope: OBV 5日归一化斜率
    - vol_price_corr: 5日 收益率-成交量变化 相关系数
    - vol_weighted_mtm: 量加权动量
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    vol = df["volume"]

    # 1. 量比
    df["tech_ths_vol_ratio"] = _compute_vol_ratio(vol, 20)

    # 2. VWAP 偏离度（日级 VWAP 近似 = 典型价）
    vwap = (h + l + c) / 3.0
    df["tech_ths_vwap_dev"] = safe_divide(c, vwap) - 1.0

    # 3. OBV 5日归一化斜率
    obv = (np.sign(c.diff()) * vol).cumsum()
    obv_norm = safe_divide(
        obv - obv.shift(5), obv.abs().rolling(20, min_periods=1).mean() + 1.0
    )
    df["tech_ths_obv_slope"] = obv_norm.fillna(0.0)

    # 4. 5日 收益率-成交量变化 相关系数
    ret = c.pct_change()
    vol_chg = vol.pct_change()
    df["tech_ths_vol_price_corr"] = (
        ret.rolling(5, min_periods=3).corr(vol_chg).fillna(0.0)
    )

    # 5. 量加权动量 = (C - REF(C,1)) * VOL / MA(VOL,20)
    mtm = c - ths_ref(c, 1)
    ma_vol20 = vol.rolling(20, min_periods=1).mean()
    df["tech_ths_vol_weighted_mtm"] = safe_divide(mtm * vol, ma_vol20).fillna(0.0)

    logger.debug("量价派生特征: 5 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  主力资金流向因子 (8 列) — 新增
# ════════════════════════════════════════════════════════════════════


def compute_main_force_flow(df: pd.DataFrame) -> pd.DataFrame:
    """主力资金流向估算（8 列, 纯量价）.

    无逐笔数据时，用日线量价近似主力资金流向：
    - close > open → 主力买入 (正流入)
    - close < open → 主力卖出 (负流出)
    - 流入强度 ∝ |close-open| * volume

    另检测资金流向背离：
    - 顶背离: 价涨 + 资金流出 → -1
    - 底背离: 价跌 + 资金流入 → +1
    """
    c, o = df["close"], df["open"]
    h, l = df["high"], df["low"]
    vol = df["volume"]

    # 1. 主力净流入 = (close - open) * volume
    flow_net = (c - o) * vol
    df["tech_ths_flow_net"] = flow_net

    # 2. 5日 / 20日 均值
    df["tech_ths_flow_net_ma5"] = flow_net.rolling(5, min_periods=1).mean()
    df["tech_ths_flow_net_ma20"] = flow_net.rolling(20, min_periods=1).mean()

    # 3. 主力净流入比率 = flow_net / (close * volume)
    turnover = c * vol
    df["tech_ths_flow_ratio"] = safe_divide(flow_net, turnover)

    # 4. 累积净流入 (归一化到 -1~1)
    flow_accum_raw = flow_net.cumsum()
    rolling_abs_max = flow_accum_raw.abs().rolling(60, min_periods=1).max() + 1.0
    df["tech_ths_flow_accum"] = safe_divide(flow_accum_raw, rolling_abs_max)

    # 5. 资金流向背离
    price_up = c > ths_ref(c, 1)
    price_down = c < ths_ref(c, 1)
    flow_out = flow_net < 0  # 资金流出
    flow_in = flow_net > 0  # 资金流入
    divergence = pd.Series(0.0, index=df.index)
    divergence[price_up & flow_out] = -1.0  # 顶背离: 价涨+资金流出
    divergence[price_down & flow_in] = 1.0  # 底背离: 价跌+资金流入
    df["tech_ths_flow_divergence"] = divergence

    # 6. 资金强度 = |flow_net| / ((high-low) * volume)
    price_range = (h - l) * vol
    df["tech_ths_flow_strength"] = safe_divide(flow_net.abs(), price_range)

    # 7. 资金流向趋势 (5日线性回归斜率)
    def _slope(x):
        y = np.arange(len(x))
        return linregress(y, x).slope

    df["tech_ths_flow_trend"] = (
        df["tech_ths_flow_net_ma5"]
        .rolling(5, min_periods=5)
        .apply(_slope, raw=True)
        .fillna(0.0)
    )

    logger.debug("主力资金流向: 8 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  控盘增强因子 (3 列) — 新增
# ════════════════════════════════════════════════════════════════════


def compute_ctrl_enhancement(df: pd.DataFrame) -> pd.DataFrame:
    """控盘增强因子（3 列）.

    在原有 4 列控盘指标 (ctrl_low/mid/high/flag) 基础上增强：
    - ctrl_ratio: 控盘比例，低价区筹码占比越高→主力控盘越强
    - ctrl_concentration: 筹码集中度，中间区域越窄→筹码越集中
    - ctrl_change: 控盘比例 5 日变化，正→主力加仓

    必须在 compute_chip_control() 之后调用。
    """
    ctrl_low = df.get("tech_ths_ctrl_low", pd.Series(0.0, index=df.index))
    ctrl_mid = df.get("tech_ths_ctrl_mid", pd.Series(0.0, index=df.index))
    ctrl_high = df.get("tech_ths_ctrl_high", pd.Series(0.0, index=df.index))

    # 控盘比例 = ctrl_low / (ctrl_low + ctrl_mid + ctrl_high)
    total = ctrl_low + ctrl_mid + ctrl_high
    df["tech_ths_ctrl_ratio"] = safe_divide(ctrl_low, total)

    # 筹码集中度 = 1 - ctrl_mid / 100
    df["tech_ths_ctrl_concentration"] = 1.0 - safe_divide(ctrl_mid, 100.0)

    # 控盘比例 5 日变化
    df["tech_ths_ctrl_change"] = df["tech_ths_ctrl_ratio"] - df[
        "tech_ths_ctrl_ratio"
    ].shift(5)
    df["tech_ths_ctrl_change"] = df["tech_ths_ctrl_change"].fillna(0.0)

    logger.debug("控盘增强: 3 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  筹码分布增强因子 (4 列) — 新增
# ════════════════════════════════════════════════════════════════════


def compute_chip_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """筹码分布增强因子（4 列）.

    在原有 7 列控盘指标 (4+3) 基础上，增加更细粒度的筹码分布特征：
    - chip_profit_ratio: 获利盘比例，当前价以下筹码占比 (0~1)
    - chip_concentration_20: 20日筹码集中度，值越高→价格波动越小→筹码越集中
    - chip_cost_skew: 筹码成本偏态，当前价在60日高低区间的相对位置，>0.5偏高
    - chip_low_high_ratio: 低高价区筹码比，越大→低位筹码越多→主力成本越低

    必须在 compute_chip_control() 之后调用。
    """
    c = df["close"]
    h = df["high"]
    l = df["low"]
    vol = df["volume"]

    # 1. 获利盘比例 = WINNER(close) / 100 (近似，复用已有函数)
    df["tech_ths_chip_profit_ratio"] = _approx_winner(c, c, h, l, vol) / 100.0

    # 2. 20日筹码集中度 = 1 - STD(close,20) / MEAN(close,20)
    std_20 = c.rolling(20, min_periods=1).std()
    mean_20 = c.rolling(20, min_periods=1).mean()
    df["tech_ths_chip_concentration_20"] = 1.0 - safe_divide(std_20, mean_20)
    df["tech_ths_chip_concentration_20"] = df["tech_ths_chip_concentration_20"].clip(
        lower=0.0
    )

    # 3. 筹码成本偏态 = (close - LLV(low,60)) / (HHV(high,60) - LLV(low,60))
    llv_60 = ths_llv(l, 60)
    hhv_60 = ths_hhv(h, 60)
    df["tech_ths_chip_cost_skew"] = safe_divide(c - llv_60, hhv_60 - llv_60)
    df["tech_ths_chip_cost_skew"] = df["tech_ths_chip_cost_skew"].clip(
        lower=0.0, upper=1.0
    )

    # 4. 低高价区筹码比 = ctrl_low / (ctrl_high + 1)
    ctrl_low = df.get("tech_ths_ctrl_low", pd.Series(0.0, index=df.index))
    ctrl_high = df.get("tech_ths_ctrl_high", pd.Series(0.0, index=df.index))
    df["tech_ths_chip_low_high_ratio"] = safe_divide(ctrl_low, ctrl_high + 1.0)

    logger.debug("筹码分布增强: 4 列")
    return df


# ════════════════════════════════════════════════════════════════════
#  统一入口
# ════════════════════════════════════════════════════════════════════


def add_all_ths_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算全部同花顺量价指标，输出 45 列 tech_ths_* 因子.

    Args:
        df: 日线数据，必须包含 ``open, high, low, close, volume``.

    Returns:
        原 df 附加 45 个 ``tech_ths_*`` 列.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"THS 指标缺少必需列: {missing}")

    logger.info("开始计算同花顺量价指标 (45 列)...")

    df = compute_main_force_chip(df)
    df = compute_chip_control(df)
    df = compute_bull_finder(df)
    df = compute_trend_top_bottom(df)
    df = compute_vol_price_features(df)
    df = compute_main_force_flow(df)
    df = compute_ctrl_enhancement(df)
    df = compute_chip_distribution(df)

    # NaN → 0
    for col in THS_FACTOR_COLUMNS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    logger.info("同花顺量价指标计算完成: %d 列 (含筹码分布)", len(THS_FACTOR_COLUMNS))
    return df
