# -*- coding: utf-8 -*-
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
        df = self.dim08_calendar_month(df)
        df = self.dim09_custom_formulas(df, float_shares_map)
        df = self.dim10_money_flow(df)
        df = self.dim12_ma_system(df)
        df = self.dim13_holiday(df)
        df = self.dim14_market_sentiment(df)
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
            l = g.get("low_hfq", g["low"])
            c = g["close_hfq"]
            tr = pd.concat(
                [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1
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
            df["is_limit_up"] = (abs(df["close"] - lu_price) < 0.01).astype(int)
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
        from app.indicators.zhuli_lasheng import zhuli_lasheng
        from app.indicators.yimeng_dingdi import yimeng_dingdi
        from app.indicators.faxian_niugu import faxian_niugu
        from app.indicators.chip_distribution import ChipDistribution

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
