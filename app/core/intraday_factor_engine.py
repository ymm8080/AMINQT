# -*- coding: utf-8 -*-
"""交易级五分钟量价因子体系 (P7, ARCH §5.10).

25 维五分钟因子 = 16 个股 (含 13 微观结构) + 5 大盘 + 4 板块。
注意: 执行循环每 2 分钟, 因子粒度 5 分钟 K 线 (48 根/日) — 每次循环
用最新已完成 5min bar + 当前未完成 bar 实时快照重算。

防未来函数: 所有窗口只回看当日前/当日已完成的 bar。
除零防护: 统一走 ``_safe_div``。
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STOCK_FACTOR_DIM = 16    # 个股微观因子 (含 13 维微观结构)
MARKET_FACTOR_DIM = 5    # 大盘盘中因子
SECTOR_FACTOR_DIM = 4    # 板块盘中因子
TOTAL_DIM = STOCK_FACTOR_DIM + MARKET_FACTOR_DIM + SECTOR_FACTOR_DIM  # 25

STOCK_FACTOR_NAMES: List[str] = [
    "intraday_vwap_dev",          # 1  VWAP偏离度
    "intraday_momentum_5",        # 2  5根动量 (≈25min)
    "intraday_momentum_15",       # 3  15根动量
    "intraday_volume_ratio",      # 4  分钟量比
    "intraday_amount_surge",      # 5  成交额脉冲
    "intraday_price_vol_corr",    # 6  量价相关性 (12根)
    "intraday_buy_pressure",      # 7  买入压力
    "intraday_sell_pressure",     # 8  卖出压力
    "intraday_spread",            # 9  振幅 (12根)
    "intraday_trend_strength",    # 10 趋势强度 (线性斜率)
    "intraday_reversal",          # 11 反转信号
    "intraday_vol_concentration", # 12 成交集中度 (top3 bar)
    "intraday_open_strength",     # 13 开盘强弱
    "intraday_close_strength",    # 14 尾盘强弱
    "intraday_position_in_range", # 15 日内区间位置
    "intraday_vol_trend",         # 16 量能趋势
]

MARKET_FACTOR_NAMES: List[str] = [
    "market_5min_momentum",       # 17 大盘5分钟动量
    "market_5min_volume_ratio",   # 18 大盘分钟量比
    "market_5min_vwap_dev",       # 19 大盘VWAP偏离
    "market_5min_breadth",        # 20 大盘涨跌家数比
    "market_5min_trend",          # 21 大盘日内趋势
]

SECTOR_FACTOR_NAMES: List[str] = [
    "sector_5min_momentum",       # 22 板块5分钟动量
    "sector_5min_volume_ratio",   # 23 板块分钟量比
    "sector_5min_dev",            # 24 个股vs板块偏离
    "sector_5min_flow",           # 25 板块资金流向 (15min)
]

ALL_FACTOR_NAMES = STOCK_FACTOR_NAMES + MARKET_FACTOR_NAMES + SECTOR_FACTOR_NAMES

_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _safe_div(num: float, den: float) -> float:
    """除零安全除法: 分母为 0 / NaN → 0."""
    if den is None or not np.isfinite(den) or den == 0:
        return 0.0
    out = num / den
    return float(out) if np.isfinite(out) else 0.0


def _slope_norm(close: np.ndarray, window: int = 12) -> float:
    """最近 window 根收盘价的归一化线性斜率 (slope / last_close)."""
    seg = close[-window:] if len(close) >= window else close
    if len(seg) < 2:
        return 0.0
    x = np.arange(len(seg), dtype=float)
    slope = np.polyfit(x, seg, 1)[0]
    return _safe_div(slope, seg[-1])


class IntradayFactorEngine:
    """五分钟因子计算器."""

    def __init__(self, config: dict = None) -> None:
        self.config = config or {}

    # ────────────────────────────────────────────────────────────────
    #  内部工具
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(bars: pd.DataFrame, name: str) -> pd.DataFrame:
        """校验 K 线列; 返回原表 (允许空表)."""
        missing = [c for c in _REQUIRED_COLS if c not in bars.columns]
        if missing:
            raise KeyError(f"{name} 缺少必需列: {missing}")
        return bars

    @staticmethod
    def _with_snapshot(bars: pd.DataFrame,
                       current_snapshot: Optional[dict]) -> pd.DataFrame:
        """把当前未完成 bar 快照拼到已完成 bar 末尾 (可选).

        快照可含 open/high/low/close/volume/amount; 缺 close 时取 price。
        """
        if not current_snapshot:
            return bars
        snap = dict(current_snapshot)
        if "close" not in snap and "price" in snap:
            snap["close"] = snap["price"]
        if "close" not in snap:
            return bars
        row = {
            "open": snap.get("open", snap["close"]),
            "high": snap.get("high", snap["close"]),
            "low": snap.get("low", snap["close"]),
            "close": snap["close"],
            "volume": snap.get("volume", 0.0),
        }
        if "amount" in snap:
            row["amount"] = snap["amount"]
        extra = pd.DataFrame([row])
        extra = extra.reindex(columns=bars.columns)
        return pd.concat([bars, extra], ignore_index=True)

    @staticmethod
    def _ohlcva(bars: pd.DataFrame):
        """提取 numpy 数组; amount 缺失时用 close*volume 近似."""
        o = bars["open"].to_numpy(dtype=float)
        h = bars["high"].to_numpy(dtype=float)
        l = bars["low"].to_numpy(dtype=float)
        c = bars["close"].to_numpy(dtype=float)
        v = bars["volume"].to_numpy(dtype=float)
        if "amount" in bars.columns:
            a = bars["amount"].to_numpy(dtype=float)
        else:
            a = c * v
        return o, h, l, c, v, a

    # ────────────────────────────────────────────────────────────────
    #  A. 个股微观因子 (16 维)
    # ────────────────────────────────────────────────────────────────

    def compute_stock_factors(self, bars_5min: pd.DataFrame,
                              current_snapshot: dict = None) -> Dict[str, float]:
        """个股微观因子 (16 维): VWAP偏离/动量/量比/额脉冲/量价相关/
        买卖压力/振幅/趋势强度/反转 等 (ARCH §5.10.2.A).

        Args:
            bars_5min: 当日已完成 5 分钟 K 线。
            current_snapshot: 当前未完成 bar 实时快照 (可选)。

        Returns:
            {factor_name: value} 16 维。
        """
        bars = self._with_snapshot(self._validate(bars_5min, "bars_5min"),
                                   current_snapshot)
        zero = {name: 0.0 for name in STOCK_FACTOR_NAMES}
        if len(bars) == 0:
            logger.warning("个股 5min K 线为空 — 16 维个股因子置 0")
            return zero

        o, h, l, c, v, a = self._ohlcva(bars)
        n = len(c)
        last = c[-1]

        def _ret(k: int) -> float:
            """k 根前动量 close[-1]/close[-1-k] - 1 (不足则取最早)."""
            base = c[-1 - k] if n > k else c[0]
            return _safe_div(last, base) - 1.0

        w12 = slice(max(0, n - 12), n)
        rng12 = (h[w12] - l[w12])
        range_sum = float(rng12.sum())
        total_vol = float(v.sum())
        total_amt = float(a.sum())

        out: Dict[str, float] = {}

        # 1. VWAP 偏离
        vwap = _safe_div(total_amt, total_vol)
        out["intraday_vwap_dev"] = _safe_div(last - vwap, vwap)

        # 2/3. 动量 (5 根 ≈ 25min, 15 根)
        out["intraday_momentum_5"] = _ret(5)
        out["intraday_momentum_15"] = _ret(15)

        # 4. 分钟量比: 最近 5 根均量 / 最近 20 根均量
        v5 = float(v[-5:].mean()) if n >= 1 else 0.0
        v20 = float(v[-20:].mean()) if n >= 1 else 0.0
        out["intraday_volume_ratio"] = _safe_div(v5, v20)

        # 5. 成交额脉冲: 当前 bar / 近 5 根均值
        a5 = float(a[-5:].mean()) if n >= 1 else 0.0
        out["intraday_amount_surge"] = _safe_div(a[-1], a5)

        # 6. 量价相关 (12 根收益率 vs 成交量)
        if n >= 3:
            ret12 = np.diff(c[w12]) / np.where(c[w12][:-1] == 0, 1.0, c[w12][:-1])
            vol12 = v[w12][1:]
            if len(ret12) >= 2 and np.std(ret12) > 0 and np.std(vol12) > 0:
                out["intraday_price_vol_corr"] = float(np.corrcoef(ret12, vol12)[0, 1])
            else:
                out["intraday_price_vol_corr"] = 0.0
        else:
            out["intraday_price_vol_corr"] = 0.0

        # 7/8. 买卖压力 (12 根)
        buy_amt = float(np.clip(c[w12] - o[w12], 0.0, None).sum())
        sell_amt = float(np.clip(o[w12] - c[w12], 0.0, None).sum())
        out["intraday_buy_pressure"] = _safe_div(buy_amt, range_sum)
        out["intraday_sell_pressure"] = _safe_div(sell_amt, range_sum)

        # 9. 振幅 (12 根)
        hhv12 = float(h[w12].max())
        llv12 = float(l[w12].min())
        out["intraday_spread"] = _safe_div(hhv12 - llv12, last)

        # 10. 趋势强度 (12 根线性斜率 / close)
        out["intraday_trend_strength"] = _slope_norm(c, 12)

        # 11. 反转信号: (close_now - close_12bars_ago) / range_12bars
        base12 = c[-13] if n > 12 else c[0]
        out["intraday_reversal"] = _safe_div(last - base12, hhv12 - llv12)

        # 12. 成交集中度: top3 bar 成交额 / 全日成交额
        top3 = float(np.sort(a)[-3:].sum()) if n >= 1 else 0.0
        out["intraday_vol_concentration"] = _safe_div(top3, total_amt)

        # 13. 开盘强弱: (open - prev_close)/prev_close + 前 15 分钟收益
        prev_close = float(self.config.get("prev_close", 0.0) or 0.0)
        if prev_close <= 0 and current_snapshot:
            prev_close = float(current_snapshot.get("prev_close", 0.0) or 0.0)
        gap = _safe_div(o[0] - prev_close, prev_close) if prev_close > 0 else 0.0
        first_15 = _safe_div(c[min(2, n - 1)], o[0]) - 1.0
        out["intraday_open_strength"] = gap + first_15

        # 14. 尾盘强弱: 近 15 分钟收益 + 末根量比
        last_15 = _safe_div(last, c[max(0, n - 4)]) - 1.0
        last_vol_ratio = _safe_div(v[-1], float(v[-5:].mean()))
        out["intraday_close_strength"] = last_15 + last_vol_ratio

        # 15. 日内区间位置
        day_high = float(h.max())
        day_low = float(l.min())
        out["intraday_position_in_range"] = _safe_div(last - day_low,
                                                      day_high - day_low)

        # 16. 量能趋势: 近 5 根均量 / 近 20 根均量 (同 4 但用均值口径,
        #     此处取 5 根总量/20 根总量归一 — 与量比互补)
        out["intraday_vol_trend"] = _safe_div(
            float(v[-5:].sum()), float(v[-20:].sum()) / 4.0
        )

        # 缓存最近个股 5 根动量, 供 sector_5min_dev 使用
        self._last_stock_momentum_5 = out["intraday_momentum_5"]

        return {k: float(np.nan_to_num(val, nan=0.0)) for k, val in out.items()}

    # ────────────────────────────────────────────────────────────────
    #  B. 大盘盘中因子 (5 维)
    # ────────────────────────────────────────────────────────────────

    def compute_market_factors(self, index_bars: pd.DataFrame) -> Dict[str, float]:
        """大盘盘中因子 (5 维).

        Args:
            index_bars: 上证指数当日 5 分钟 K 线。涨跌家数比无法从 K 线
                推出, 通过 ``config['market_breadth']`` 或列 ``breadth``
                注入, 缺省 1.0 (中性)。

        Returns:
            {factor_name: value} 5 维。
        """
        self._validate(index_bars, "index_bars")
        zero = {name: 0.0 for name in MARKET_FACTOR_NAMES}
        zero["market_5min_breadth"] = float(
            self.config.get("market_breadth", 1.0)
        )
        if len(index_bars) == 0:
            logger.warning("大盘 5min K 线为空 — 大盘因子置 0 (breadth 用缺省)")
            return zero

        _, _, _, c, v, a = self._ohlcva(index_bars)
        n = len(c)
        last = c[-1]

        out: Dict[str, float] = {}

        # 17. 大盘 5 根动量
        base = c[-6] if n > 5 else c[0]
        out["market_5min_momentum"] = _safe_div(last, base) - 1.0

        # 18. 大盘分钟量比
        v5 = float(v[-5:].mean())
        v20 = float(v[-20:].mean())
        out["market_5min_volume_ratio"] = _safe_div(v5, v20)

        # 19. 大盘 VWAP 偏离
        vwap = _safe_div(float(a.sum()), float(v.sum()))
        out["market_5min_vwap_dev"] = _safe_div(last - vwap, vwap)

        # 20. 大盘涨跌家数比 (外部注入)
        breadth = float(self.config.get("market_breadth", 1.0))
        if "breadth" in index_bars.columns and len(index_bars) > 0:
            breadth = float(index_bars["breadth"].iloc[-1])
        out["market_5min_breadth"] = breadth

        # 21. 大盘日内趋势
        out["market_5min_trend"] = _slope_norm(c, 12)

        return {k: float(np.nan_to_num(val, nan=0.0)) for k, val in out.items()}

    # ────────────────────────────────────────────────────────────────
    #  C. 板块盘中因子 (4 维)
    # ────────────────────────────────────────────────────────────────

    def compute_sector_factors(self, sector_bars: pd.DataFrame,
                               stock_bars: pd.DataFrame = None) -> Dict[str, float]:
        """板块盘中因子 (4 维).

        Args:
            sector_bars: 板块指数当日 5 分钟 K 线。
            stock_bars: 个股 5 分钟 K 线 (可选, 用于 sector_5min_dev;
                缺省时回退到最近一次 compute_stock_factors 的缓存)。

        Returns:
            {factor_name: value} 4 维。
        """
        self._validate(sector_bars, "sector_bars")
        zero = {name: 0.0 for name in SECTOR_FACTOR_NAMES}
        if len(sector_bars) == 0:
            logger.warning("板块 5min K 线为空 — 板块因子置 0")
            return zero

        o, _, _, c, v, a = self._ohlcva(sector_bars)
        n = len(c)
        last = c[-1]

        out: Dict[str, float] = {}

        # 22. 板块 5 根动量
        base = c[-6] if n > 5 else c[0]
        sector_mom5 = _safe_div(last, base) - 1.0
        out["sector_5min_momentum"] = sector_mom5

        # 23. 板块分钟量比
        v5 = float(v[-5:].mean())
        v20 = float(v[-20:].mean())
        out["sector_5min_volume_ratio"] = _safe_div(v5, v20)

        # 24. 个股 vs 板块 当根偏离: stock_ret - sector_ret (最近 5 根)
        if stock_bars is not None and len(stock_bars) > 0:
            sc = stock_bars["close"].to_numpy(dtype=float)
            s_base = sc[-6] if len(sc) > 5 else sc[0]
            stock_mom5 = _safe_div(sc[-1], s_base) - 1.0
        else:
            stock_mom5 = float(getattr(self, "_last_stock_momentum_5", 0.0))
        out["sector_5min_dev"] = stock_mom5 - sector_mom5

        # 25. 板块资金流向 (近 15min = 3 根): 优先 net_flow 列,
        #     否则用 (close-open)*volume / amount 代理
        if "net_flow" in sector_bars.columns:
            flow_num = float(sector_bars["net_flow"].to_numpy(dtype=float)[-3:].sum())
        else:
            flow_num = float(((c - o) * v)[-3:].sum())
        flow_den = float(a[-3:].sum())
        out["sector_5min_flow"] = _safe_div(flow_num, flow_den)

        return {k: float(np.nan_to_num(val, nan=0.0)) for k, val in out.items()}

    # ────────────────────────────────────────────────────────────────
    #  统一入口
    # ────────────────────────────────────────────────────────────────

    def compute_5min(self, stock_bars: pd.DataFrame,
                     index_bars: pd.DataFrame,
                     sector_bars: pd.DataFrame,
                     current_snapshot: dict = None) -> np.ndarray:
        """合并计算 25 维五分钟因子向量.

        Args:
            stock_bars: 个股 5min K 线。
            index_bars: 上证指数 5min K 线。
            sector_bars: 板块指数 5min K 线。
            current_snapshot: 当前未完成 bar 快照 (可选)。

        Returns:
            (25,) ndarray = [16 个股 + 5 大盘 + 4 板块]。
        """
        stock = self.compute_stock_factors(stock_bars, current_snapshot)
        market = self.compute_market_factors(index_bars)
        sector = self.compute_sector_factors(sector_bars, stock_bars)

        vec = np.array(
            [stock[k] for k in STOCK_FACTOR_NAMES]
            + [market[k] for k in MARKET_FACTOR_NAMES]
            + [sector[k] for k in SECTOR_FACTOR_NAMES],
            dtype=float,
        )
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        assert vec.shape == (TOTAL_DIM,), f"因子维度错误: {vec.shape}"
        logger.debug("5min 因子向量计算完成: %d 维", TOTAL_DIM)
        return vec
