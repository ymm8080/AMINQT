"""
特征引擎 V3.5 — 14 维度 (DESIGN §14.3, 安全网 #5/#6/#13)
=============================================================
铁律: 所有 rolling/shift/cumsum 必须 groupby("symbol") (安全网 #5);
      一切计算前 sort_values([symbol, date]) (安全网 #13);
      NaN 不填充直接入 LightGBM, 关键因子加 missingness 指示.
技术指标用 pandas 实现 (MACD/RSI/ATR/BBANDS), 不依赖 TA-Lib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cleaning_pipeline import get_limit_pct

MA_WINDOWS = (5, 10, 20, 60, 120, 250)
# 行业中性化目标列 (申万一级行业内 rank)
NEUTRALIZE_COLS = ["PE_log", "PB_LF", "turnover_rate", "chip_concentration"]
# 关键因子 missingness 指示
MISSINGNESS_COLS = ["main_money_flow", "chip_concentration"]


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _apply_per_stock(df: pd.DataFrame, fn) -> pd.DataFrame:
    """逐股应用特征函数 — groupby(symbol) 强制 (安全网 #5).

    用显式循环而非 groupby.apply: pandas 2.2+/3.x 对 apply 丢弃分组列的行为
    不一致, 显式循环跨版本稳定且保列.
    """
    parts = [fn(g.copy()) for _, g in df.groupby("symbol")]
    return pd.concat(parts).sort_values(["symbol", "date"]).reset_index(drop=True)


def _safe_divide(numerator, denominator) -> pd.Series:
    """Safe division: NaN where denominator is 0 (防除零, 保留 NaN 语义)."""
    return numerator / denominator.replace(0, np.nan)


class FeatureEngineV35:
    """14 维特征. 输入: 清洗后的面板 (含 hfq+raw 双价格 + 财务/筹码/资金流已 merge 的列)."""

    # ---------------- 总装 ----------------
    def build(
        self, df: pd.DataFrame, float_shares_map: dict | None = None
    ) -> pd.DataFrame:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
        df = self.dim01_price_volume(df)
        df = self.dim02_volatility(df)
        df = self.dim03_fundamentals(df)
        df = self.dim07_limit_gene(df)
        df = self.dim04_sector_effect(df)
        df = self.dim_active_pit(df)  # §14.2.2 安全网 #15
        df = self.dim08_calendar_month(df)
        df = self.dim09_custom_formulas(df, float_shares_map)
        df = self.dim10_money_flow(df)
        df = self.dim12_ma_system(df)
        df = self.dim13_holiday(df)
        df = self.dim14_market_sentiment(df)
        df = self.dim15_alpha_factors(df)
        df = self.dim16_candlestick(df)
        df = self.dim17_extended_factors(df)
        df = self.dim18_lhb(df)
        df = self.dim19_amihud(df)  # E6
        df = self.industry_neutralize(df)
        df = self.add_missingness_flags(df)
        return df

    # ---------------- ①价量动能 ----------------
    def dim01_price_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD(12,26,9) / RSI(14) / KDJ(9,3,3) / 60日乖离率 / 量价背离."""

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            c = g["close_hfq"]
            ema12, ema26 = _ema(c, 12), _ema(c, 26)
            g["MACD"] = ema12 - ema26
            g["MACD_signal"] = _ema(g["MACD"], 9)
            g["MACD_hist"] = g["MACD"] - g["MACD_signal"]
            # RSI(14)
            delta = c.diff()
            up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            g["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            # KDJ(9,3,3)
            hhv9 = (
                g["high_hfq"].rolling(9, min_periods=9).max()
                if "high_hfq" in g
                else g["high"].rolling(9, min_periods=9).max()
            )
            llv9 = (
                g["low_hfq"].rolling(9, min_periods=9).min()
                if "low_hfq" in g
                else g["low"].rolling(9, min_periods=9).min()
            )
            rsv = (c - llv9) / (hhv9 - llv9).replace(0, np.nan) * 100
            g["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
            g["D"] = g["K"].ewm(alpha=1 / 3, adjust=False).mean()
            g["J"] = 3 * g["K"] - 2 * g["D"]
            # 60日乖离率 (需上市>=250天才有完整值 → 步骤1 已保证)
            g["bias_60"] = c / c.rolling(60, min_periods=60).mean() - 1
            # 量价背离: 价涨量缩=1 / 价跌量增=-1
            pc = c.pct_change()
            vc = g["volume"].pct_change()
            g["pv_divergence"] = np.where(
                (pc > 0) & (vc < 0), 1, np.where((pc < 0) & (vc > 0), -1, 0)
            )
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ②波动率 ----------------
    def dim02_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR(14)/收盘价 (归一化) + 布林带宽度 (20日, 2σ). 振幅已删 (与 ATR 共线)."""

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            h = g.get("high_hfq", g["high"])
            low = g.get("low_hfq", g["low"])
            c = g["close_hfq"]
            tr = pd.concat(
                [h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()], axis=1
            ).max(axis=1)
            g["ATR_pct"] = tr.rolling(14, min_periods=14).mean() / c
            ma20 = c.rolling(20, min_periods=20).mean()
            sd20 = c.rolling(20, min_periods=20).std()
            g["BB_width"] = (4 * sd20) / ma20
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ③基本面 ----------------
    def dim03_fundamentals(
        self, df: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """PE_log + is_negative_pe / PB / 净利营收增速 — announce_date PIT 对齐 (merge_asof, 严禁 ffill 跨季).

        fundamentals: 需含 symbol, announce_date, PE_TTM, PB_LF [, netprofit_yoy, revenue_yoy]
        """
        if fundamentals is not None and len(fundamentals):
            f = fundamentals.copy()
            f["announce_date"] = pd.to_datetime(f["announce_date"])
            df = df.sort_values("date")
            f = f.sort_values("announce_date")
            df = pd.merge_asof(
                df,
                f,
                left_on="date",
                right_on="announce_date",
                by="symbol",
                direction="backward",
            )
        if "PE_TTM" in df.columns:
            df["PE_log"] = np.where(df["PE_TTM"] > 0, np.log(df["PE_TTM"]), np.nan)
            df["is_negative_pe"] = (df["PE_TTM"] < 0).astype(int)
        df["is_STAR"] = (df["board"] == "STAR").astype(int)
        # 相对涨停强度 = 涨跌幅 / 涨停幅度 (历史 limit_pct, 安全网 #6)
        df["ret_pct"] = df.groupby("symbol")["close"].pct_change()
        df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])]
        df["relative_limit_strength"] = df["ret_pct"] / df["limit_pct"]
        return df

    # ---------------- ⑦涨停基因 + ⑪连板高度 ----------------
    def dim07_limit_gene(self, df: pd.DataFrame) -> pd.DataFrame:
        """过去 10/20 日涨停天数 / 炸板率 / 连板高度 (0-4 截断, 4 为独立类别, V3.5 修正)."""
        if "is_limit_up" not in df.columns:
            lu_price = (df["pre_close"] * (1 + df["limit_pct"])).round(2)
            df["is_limit_up"] = (
                abs(df["close"] - lu_price) < np.maximum(0.01, lu_price * 0.001)
            ).astype(int)  # B5: 相对容差
        g = df.groupby("symbol")["is_limit_up"]
        df["limit_up_days_10"] = g.rolling(10).sum().reset_index(level=0, drop=True)
        df["limit_up_days_20"] = g.rolling(20).sum().reset_index(level=0, drop=True)
        # 炸板率: 盘中触板但收盘未板 (需 intraday touch 数据, 无则 NaN)
        if "touched_limit_up" in df.columns:
            gb = df.groupby("symbol")
            touched = (
                gb["touched_limit_up"].rolling(20).sum().reset_index(level=0, drop=True)
            )
            closed = df["limit_up_days_20"]
            df["break_board_rate"] = (touched - closed) / touched.replace(0, np.nan)
        # 连板高度: 连续涨停计数, 截断到 4 (>=5 合并到 4)
        cons = g.apply(
            lambda x: x * (x.groupby((x == 0).cumsum()).cumcount() + 1)
        ).reset_index(level=0, drop=True)
        df["consecutive_board"] = cons.clip(upper=4)
        return df

    # ---------------- ④板块效应 ----------------
    def dim04_sector_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        """板块内涨停家数 + 板块指数涨幅 (行业内均值代理).

        PIT 注记: industry 列即当日成分快照, 历史成分由上游数据负责 (严禁用今日成分回算).
        """
        if "industry" not in df.columns or "is_limit_up" not in df.columns:
            return df
        grp = ["date", "industry"]
        df["sector_limit_up_count"] = df.groupby(grp)["is_limit_up"].transform("sum")
        df["sector_return"] = df.groupby(grp)["ret_pct"].transform("mean")
        df["sector_return_5d"] = df.groupby("industry")["sector_return"].transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
        return df

    # ---------------- §14.2.2 PIT 活跃度 (安全网 #15) ----------------
    def dim_active_pit(self, df: pd.DataFrame) -> pd.DataFrame:
        """活跃度 PIT 标签: 仅用 [T-252, T-1] 已实现数据, 严禁引用 T 及之后.

        is_active = (turnover_rate_252d.mean() > 0.05) & (amplitude_252d.mean() > 0.05)
        §14.2.4 #6: 8 模型回退时 is_active 降级为输入特征, 不做分训维度.
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            tr = g["turnover_rate"]
            # amplitude = (high - low) / pre_close, shift(1) 确保 PIT (不含 T)
            amp = (g["high"] - g["low"]) / g["pre_close"].replace(0, np.nan)
            g["is_active"] = (
                tr.rolling(252, min_periods=60).mean().shift(1) > 0.05
            ) & (amp.rolling(252, min_periods=60).mean().shift(1) > 0.05)
            g["is_active"] = g["is_active"].astype(int)
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑧日历效应-月份 ----------------
    @staticmethod
    def dim08_calendar_month(df: pd.DataFrame) -> pd.DataFrame:
        """月份分类特征 (季节性防火隔离已删, 交 IC 筛选裁决)."""
        df["month"] = pd.to_datetime(df["date"]).dt.month
        return df

    # ---------------- ⑨自定义技术指标 (4 公式, 审计通过 DESIGN §8.2) ----------------
    def dim09_custom_formulas(
        self, df: pd.DataFrame, float_shares_map: dict | None = None
    ) -> pd.DataFrame:
        """NECESSARY INDICATOR 4 公式 → 特征列 (P16 复刻实现, 安全网 #4 已审计).

        产出: 主力轨迹/MAZL/吸筹/拉高/出货 (zhuli) + 益盟三线/红蓝距离 (yimeng)
              + SS金叉/多头排列 (faxian) + A04红柱/A08/获利盘 (chip, 需 float_shares_map)
        吸筹峰为盘后专用信号, 不入特征列 (含前瞻).
        """
        from app.indicators.chip_distribution import ChipDistribution
        from app.indicators.faxian_niugu import faxian_niugu
        from app.indicators.yimeng_dingdi import yimeng_dingdi
        from app.indicators.zhuli_lasheng import zhuli_lasheng

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = zhuli_lasheng(g)
            g = yimeng_dingdi(g)
            g = faxian_niugu(g)
            if float_shares_map and g["symbol"].iloc[0] in float_shares_map:
                g = ChipDistribution().build(g, float_shares_map[g["symbol"].iloc[0]])
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑤筹码 + ⑩资金流 (无条件 shift(1)) ----------------
    def dim10_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """筹码集中度/获利盘 + 主力净流入/超大单 — 无条件 groupby(symbol).shift(1).

        代价评估: IC 降 0.003-0.005, 换取工程一致性 (数据延迟根治).
        """
        shift_cols = [
            c
            for c in (
                "chip_concentration",
                "profit_ratio",
                "main_money_flow",
                "super_large_order_net",
            )
            if c in df.columns
        ]
        for col in shift_cols:
            df[col] = df.groupby("symbol")[col].shift(1)
        # ⑪ is_in_yesterday_list (Holding Bonus 用, 由 list_generator 每日回填)
        if "is_in_yesterday_list" not in df.columns:
            df["is_in_yesterday_list"] = 0
        return df

    # ---------------- ⑫均线系统 ----------------
    def dim12_ma_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """5/10/20/60/120/250 均线距离 + 多头/空头排列."""

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            c = g["close_hfq"]
            for w in MA_WINDOWS:
                g[f"MA{w}_dist"] = c / c.rolling(w, min_periods=w).mean() - 1
            g["ma_bull_align"] = (
                (g["MA5_dist"] > g["MA10_dist"]) & (g["MA10_dist"] > g["MA20_dist"])
            ).astype(int)
            g["ma_bear_align"] = (
                (g["MA5_dist"] < g["MA10_dist"]) & (g["MA10_dist"] < g["MA20_dist"])
            ).astype(int)
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑬日历效应-长假 ----------------
    @staticmethod
    def dim13_holiday(df: pd.DataFrame, holidays: list | None = None) -> pd.DataFrame:
        """days_to/after_holiday, is_pre/post_holiday (3 日阈值).

        holidays: 法定长假日期列表 (春节/国庆区间); None → 列填 NaN (交 IC 筛选).
        """
        if not holidays:
            df["days_to_holiday"] = np.nan
            df["days_after_holiday"] = np.nan
        else:
            hol = pd.to_datetime(pd.Series(holidays)).sort_values().values
            dates = pd.to_datetime(df["date"]).values
            nxt = [np.searchsorted(hol, d, side="left") for d in dates]
            df["days_to_holiday"] = [
                (hol[i] - d) / np.timedelta64(1, "D") if i < len(hol) else np.nan
                for d, i in zip(dates, nxt)
            ]
            prv = [np.searchsorted(hol, d, side="right") - 1 for d in dates]
            df["days_after_holiday"] = [
                (d - hol[i]) / np.timedelta64(1, "D") if i >= 0 else np.nan
                for d, i in zip(dates, prv)
            ]
        df["is_pre_holiday"] = (df["days_to_holiday"] <= 3).astype(int)
        df["is_post_holiday"] = (df["days_after_holiday"] <= 3).astype(int)
        return df

    # ---------------- ⑭全市场情绪 ----------------
    def dim14_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """两市总成交额 + 5d/20d 比值 + 全市场涨/跌停家数."""
        daily = df.groupby("date").agg(
            market_turnover=("amount", "sum"), market_limit_up=("is_limit_up", "sum")
        )
        daily["market_turnover_ratio_5d"] = (
            daily["market_turnover"] / daily["market_turnover"].rolling(5).mean()
        )
        daily["market_turnover_ratio_20d"] = (
            daily["market_turnover"] / daily["market_turnover"].rolling(20).mean()
        )
        if "limit_pct" in df.columns:
            ld_price = (df["pre_close"] * (1 - df["limit_pct"])).round(2)
            df["_is_limit_down"] = (abs(df["close"] - ld_price) < 0.01).astype(int)
            daily["market_limit_down"] = df.groupby("date")["_is_limit_down"].sum()
            df = df.drop(columns=["_is_limit_down"])
        return df.merge(daily.reset_index(), on="date", how="left")

    # ---------------- ⑮ Alpha101 + GTJA191 因子 (源自 aurumq-rl, pandas 复刻) ----------------
    def dim15_alpha_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """WorldQuant Alpha101 + 国泰君安 GTJA191 精选因子 (pandas 复刻, 无 polars 依赖).

        筛选原则: 公式简洁 + A 股日频适用 + 与现有 14 维低共线.
        最终因子集由 IC 筛选器 (§14.3) 裁决去留, 此处仅提供原料.
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            o, c, v = g["open"], g["close"], g["volume"]
            # --- Alpha101 ---
            # alpha006: -corr(open, volume, 10) — 量价背离信号
            g["alpha006"] = -o.rolling(10, min_periods=10).corr(v)
            # alpha012: sign(Δvol) * (-Δclose) — 放量下跌反转信号
            g["alpha012"] = np.sign(v.diff(1)) * (-c.diff(1))
            # alpha041: max(Δclose_10, 5)^2 * sign(Δclose_5) — 动量加速/减速
            dc10 = c.diff(10)
            dc5 = c.diff(5)
            g["alpha041"] = (dc10.where(dc10 > 5, 5)) ** 2 * np.sign(dc5)
            # alpha042: rank(ADV15) * corr(ADV15, close) — 成交额与收盘价相关性
            adv15 = v.rolling(15, min_periods=15).mean()
            g["alpha042_ts"] = adv15.rolling(15, min_periods=15).corr(c)
            # alpha054: (-1 * ret_5d) + corr(open, vol, 5) * std(ret_5d, 5)
            ret5 = c.pct_change(5)
            g["alpha054_ts"] = (
                -ret5
                + o.rolling(5, min_periods=5).corr(v)
                * ret5.rolling(5, min_periods=5).std()
            )
            # --- GTJA191 ---
            # gtja_001: -corr(rank(Δlog(vol)), rank((close-open)/open), 6)
            dlv = np.log(v.replace(0, np.nan)).diff(1)
            intra_ret = (c - o) / o.replace(0, np.nan)
            # rank 在 per_stock 内是时序 rank; 横截面 rank 在 build 后由 industry_neutralize 补
            g["gtja_001_ts"] = -dlv.rolling(6, min_periods=6).corr(intra_ret)
            # gtja_004: -ts_rank(close, 10) — 均值回归信号
            g["gtja_004"] = -c.rolling(10, min_periods=10).rank(pct=True)
            return g

        df = _apply_per_stock(df, per_stock)
        # alpha042/alpha054/gtja_001 的横截面 rank (按 date 分组)
        for col in ("alpha042_ts", "alpha054_ts", "gtja_001_ts"):
            if col in df.columns:
                df[col] = df.groupby("date")[col].rank(pct=True)
        return df

    # ---------------- ⑯ K线形态 (源自 ohlcpattern, pandas 复刻) ----------------
    def dim16_candlestick(self, df: pd.DataFrame) -> pd.DataFrame:
        """精选 6 个 K 线形态 (二值特征), 源自 ohlcpattern skill.

        形态: bullish_engulfing / bearish_engulfing / hammer / shooting_star / morning_star / evening_star
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            o, h, lo, c = g["open"], g["high"], g["low"], g["close"]
            body = (c - o).abs()
            upper_shadow = h - np.maximum(o, c)
            lower_shadow = np.minimum(o, c) - lo
            is_white = (c > o).astype(int)
            is_black = (c < o).astype(int)
            # 看涨吞没: 前黑后白, 当前实体包覆前一实体
            g["bullish_engulfing"] = (
                (is_black.shift(1) == 1)
                & (is_white == 1)
                & (o <= c.shift(1))
                & (c >= o.shift(1))
            ).astype(int)
            # 看跌吞没: 前白后黑, 当前实体包覆前一实体
            g["bearish_engulfing"] = (
                (is_white.shift(1) == 1)
                & (is_black == 1)
                & (o >= c.shift(1))
                & (c <= o.shift(1))
            ).astype(int)
            # 锤子线: 小实体 + 长下影 + 短上影
            g["hammer"] = (
                (body < 0.3 * (h - lo).replace(0, np.nan))
                & (lower_shadow > 2 * body)
                & (upper_shadow < 0.3 * body)
            ).astype(int)
            # 射击之星: 小实体 + 长上影 + 短下影
            g["shooting_star"] = (
                (body < 0.3 * (h - lo).replace(0, np.nan))
                & (upper_shadow > 2 * body)
                & (lower_shadow < 0.3 * body)
            ).astype(int)
            # 晨星 (3 根): 前黑 + 中小实体 + 后白且收盘 > 前中点
            mid_prev = (o.shift(2) + c.shift(2)) / 2
            g["morning_star"] = (
                (is_black.shift(2) == 1)
                & (body.shift(1) < 0.5 * body.shift(2))
                & (is_white == 1)
                & (c > mid_prev)
            ).astype(int)
            # 暮星 (3 根): 前白 + 中小实体 + 后黑且收盘 < 前中点
            g["evening_star"] = (
                (is_white.shift(2) == 1)
                & (body.shift(1) < 0.5 * body.shift(2))
                & (is_black == 1)
                & (c < mid_prev)
            ).astype(int)
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑰ 扩展因子 (源自 quant-ohlcv-feature, pandas 复刻) ----------------
    def dim17_extended_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """精选 3 个低共线因子: Amihud 流动性 + Fisher 变换 + 恐慌贪婪指数."""

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            h, lo, c, v = g["high"], g["low"], g["close"], g["volume"]
            amount = g.get("amount", v * c)
            # --- Amihud 非流动性 ( adapted from quant-ohlcv-feature/Amihud.py) ---
            ret_abs = c.pct_change().abs()
            g["amihud_illiq"] = (
                (ret_abs / amount.replace(0, np.nan)).rolling(10, min_periods=5).mean()
            )
            # --- Fisher 变换 ( adapted from Fisher_v3.py) ---
            price = (h + lo) / 2
            n = 20
            min_low = lo.rolling(n, min_periods=n).min()
            max_high = h.rolling(n, min_periods=n).max()
            rng = (max_high - min_low).replace(0, np.nan)
            price_ch = 0.33 * 2 * ((price - min_low) / rng - 0.5)
            price_ch = price_ch.clip(-0.999, 0.999)
            fisher = 0.5 * np.log((1 + price_ch) / (1 - price_ch))
            g["fisher_transform"] = fisher.ewm(alpha=0.5, adjust=False).mean()
            # --- 恐慌贪婪指数 ( adapted from FearGreed_Yidai_v1.py) ---
            tr = pd.concat(
                [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
            ).max(axis=1)
            sma_close = c.rolling(10, min_periods=1).mean()
            str_ = tr / sma_close.replace(0, np.nan)
            tr_up = np.where(c > c.shift(1), str_, 0)
            tr_dn = np.where(c < c.shift(1), str_, 0)
            wma_up = pd.Series(tr_up).rolling(10, min_periods=2).mean()
            wma_dn = pd.Series(tr_dn).rolling(10, min_periods=2).mean()
            g["fear_greed"] = (wma_up - wma_dn) / (wma_up + wma_dn).replace(0, np.nan)
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑱ 龙虎榜特征 (源自 uzi-skill lhb-analyzer) ----------------
    def dim18_lhb(self, df: pd.DataFrame) -> pd.DataFrame:
        """龙虎榜特征: 条件列, 无 LHB 数据时填 NaN (交 IC 筛选裁决).

        需上游 data_supply merge: lhb_net_buy / lhb_institutional_count / lhb_hot_money_rank
        """
        lhb_cols = {
            "lhb_net_buy": "lhb_net_buy_5d",
            "lhb_institutional_count": "lhb_inst_count_5d",
            "lhb_hot_money_rank": "lhb_hot_money_5d",
        }
        for src, dest in lhb_cols.items():
            if src in df.columns:
                df[dest] = df.groupby("symbol")[src].transform(
                    lambda s: s.rolling(5, min_periods=1).mean()
                )
                # PIT: shift(1) 确保不含 T
                df[dest] = df.groupby("symbol")[dest].shift(1)
            else:
                df[dest] = np.nan
        return df

    # ---------------- ⑲ Amihud 非流动性 (E6, V3.8) ----------------
    def dim19_amihud(self, df: pd.DataFrame) -> pd.DataFrame:
        """[E6] amihud_illiquidity = |ret_1d| / amount (20 日均值, 越高越危险);
        adv20 = 20 日日均成交额 (E5 滑点分层 / E6 liquidity_cap 输入, 非特征).

        纳入 IC 筛选与 B8 剪枝范围 (V3.8 §三).
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            c = g["close_hfq"] if "close_hfq" in g else g["close"]
            # shift(1) 后向取前值 (仅用 t-1 数据, 无 look-ahead bias)
            prev_c = c.shift(1)
            # _safe_divide 防除零 (prev_c 为 0 时结果为 NaN)
            safe_ret = _safe_divide(c - prev_c, prev_c)
            ret_abs = safe_ret.abs()
            # _safe_divide 防除零 (amount 为 0 时结果为 NaN)
            raw_amihud = _safe_divide(ret_abs, g["amount"])
            # NaN 保留入 LightGBM (CLAUDE.md 铁律); np.nan_to_num 在 _train_one 下游应用
            g["amihud_illiquidity"] = np.nan_to_num(
                raw_amihud.rolling(20, min_periods=20).mean(), nan=np.nan
            )
            g["adv20"] = np.nan_to_num(
                g["amount"].rolling(20, min_periods=20).mean(), nan=np.nan
            )
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- 行业中性化 ----------------
    @staticmethod
    def industry_neutralize(df: pd.DataFrame, cols: list | None = None) -> pd.DataFrame:
        """行业差异大的因子做申万一级行业内 rank (rank within industry, 按 date+industry)."""
        if "industry" not in df.columns:
            return df
        for col in cols or NEUTRALIZE_COLS:
            if col in df.columns:
                df[f"{col}_industry_rank"] = df.groupby(["date", "industry"])[col].rank(
                    pct=True
                )
        return df

    # ---------------- 缺失值策略 ----------------
    @staticmethod
    def add_missingness_flags(
        df: pd.DataFrame, cols: list | None = None
    ) -> pd.DataFrame:
        """关键因子加 missingness 指示变量; NaN 不填充直接入 LightGBM."""
        for col in cols or MISSINGNESS_COLS:
            if col in df.columns:
                df[f"is_missing_{col}"] = df[col].isna().astype(int)
        return df

    # ---------------- 特征列清单 ----------------
    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        """返回特征列 (排除标识/标签/原始行情/中间量)."""
        exclude_prefix = ("label_", "is_limit_up", "is_one_word", "limit_up_price")
        id_cols = {
            "symbol",
            "date",
            "board",
            "industry",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "amount",
            "turnover_rate",
            "free_float_turnover_rate",
            "is_suspended",
            "is_st",
            "list_days",
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "close_hfq",
            "limit_pct",
            "announce_date",
            "PE_TTM",
            "touched_limit_up",
            "score_rank",
            "rank_amount",
            "rank_ff_turnover",
            "liquidity_score",
            "churn_suspect",
            "is_virtual",  # B18 标识列, 非特征
            "price_1455",  # B9 执行价列, 非特征
            "adv20",  # E6 中间量 (滑点分层/liquidity_cap 输入), 非特征
            # dim09 中间量与前瞻信号: 吸筹峰含 REF(X,-1) 前瞻, 严禁入特征 (安全网 #4)
            "吸筹峰",
            "VAR5",
            "VAR51",
            "time",
            "红在蓝上",
        }
        return [
            c
            for c in df.columns
            if c not in id_cols and not c.startswith(exclude_prefix)
        ]
