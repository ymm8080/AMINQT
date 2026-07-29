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
BIAS_PERIODS = (
    5,
    10,
    20,
    60,
    120,
    250,
)  # 乖离率周期 (与 MA_WINDOWS 一致, 独立常量防耦合)
# 行业中性化目标列 (申万一级行业内 rank)
NEUTRALIZE_COLS = ["turnover_rate", "chip_concentration", "conc_90", "benefit_part"]
# 关键因子 missingness 指示
MISSINGNESS_COLS = [
    "main_money_flow",
    "chip_concentration",
    "conc_90",
    "benefit_part",
    "margin_balance",
    "north_net_buy_sh",
    "holder_count",
]


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
        self,
        df: pd.DataFrame,
        float_shares_map: dict | None = None,
        cross_sectional_rank: bool = False,
    ) -> pd.DataFrame:
        """构建特征面板.

        cross_sectional_rank: 主板 IC 对截面排名负敏感, 默认关闭, 仅双创开启.
        """
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)  # 安全网 #13
        df = self.dim01_price_volume(df)
        df = self.dim02_volatility(df)
        df = self.dim03_fundamentals(df)
        df = self.dim07_limit_gene(df)
        df = self.dim04_sector_effect(df)
        df = self.dim05_turnover_liquidity(df)  # Tushare daily_basic 换手率/量比/股息率
        df = self.dim06_valuation_size(df)  # Tushare daily_basic PE/PB/PS/市值
        df = self.dim_active_pit(df)  # §14.2.2 安全网 #15
        df = self.dim08_calendar_month(df)
        df = self.dim09_custom_formulas(df, float_shares_map)
        df = self.dim10_money_flow(df)
        df = self.dim11_float_limits(
            df
        )  # Tushare stk_limit 涨跌停价 + daily_basic 流通股本
        df = self.dim12_ma_system(df)
        df = self.dim13_holiday(df)
        df = self.dim14_market_sentiment(df)
        df = self.dim15_alpha_factors(df)
        df = self.dim16_candlestick(df)
        df = self.dim17_extended_factors(df)
        df = self.dim20_short_horizon(df)  # 短周期特征 (专攻 1d 预测信噪比)
        df = self.dim18_lhb(df)
        df = self.dim19_amihud(df)  # E6
        df = self.dim21_chip_tushare(
            df
        )  # 真实筹码分布 (Tushare cyq_perf), 无CYQ时用OHLCV代理补位 (DIM20)
        # dim20_chip_proxy 已合并到 dim21 — CYQ NaN 时自动回退 OHLCV 推导
        df = self.dim22_fundamental_pit(df)  # [Alt-3] 基本面PIT (fina_indicator)
        df = self.dim23_shareholder_structure(df)  # [Alt-5] 股东户数+户均持股
        df = self.dim24_margin_trading(df)  # [Alt-2] 融资融券
        df = self.dim25_northbound(df)  # [Alt-1] 北向资金
        df = self.dim26_lhb_enhanced(df)  # [Alt-4] 龙虎榜增强
        df = self.dim27_industry_flow(df)  # [Alt-7a] 行业级 alt 数据聚合
        df = self.dim28_sector_index(df)  # [Alt-7b] 申万行业指数动量/轮动
        df = self.dim29_holdertrade(df)  # [Alt-6] 股东增减持
        df = self.dim30_kline_geometry(df)  # K线几何特征 (缺口/实体/影线/连续)
        df = self.dim31_announcement(df)  # 公告事件特征 (距上次公告天数/公告频率)
        df = self.industry_neutralize(df)
        df = self.add_missingness_flags(df)
        # 时序变化特征: 对每个非标识列自动计算 N 日变化 (_chgN)
        df = self._add_time_series_changes(df)
        # 清理 inf (逐列替换, 避免全 DataFrame replace 触发 numpy vstack OOM)
        for col in df.columns:
            if df[col].dtype in ("float64", "float32"):
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if cross_sectional_rank:
            df = self._add_cross_sectional_ranks(df)
        return df

    # ---------------- 时序变化特征 (IC 筛选器逐因子裁决) ----------------
    @staticmethod
    def _add_time_series_changes(df: pd.DataFrame) -> pd.DataFrame:
        """对每个数值特征生成 N 日时序变化 (_chgN / _pct_chgN).

        _chgN     = X(t) - X(t-N)        绝对值差分 (适合价格/量级特征)
        _pct_chgN = X(t)/X(t-N) - 1      百分比变化 (适合 ratio-scale 特征如 conc_90/benefit_part)

        全量产出, IC 筛选器逐因子评估:
          - IC 正且显著 (grade=strong/weak) → 参与训练 & 推理
          - IC 为负或不显著 (grade=dead)    → 自动排除, 不参与训练/推理

        每次滚动重训时由 ICScreener.screen() 重新评估, 窗口内 IC 决定去留,
        不依赖历史结论。若某窗口全量 dead → 等效不生效。
        """
        skip_prefix = (
            "label_",
            "is_",
            "limit_up_",
            "churn_",
            "market_state",
            "schema_version",
            "name",
            "tradestatus",
        )
        skip_suffix = (
            "_xrank",
            "_chg1",
            "_chg3",
            "_chg5",
            "_chg10",
            "_chg20",
            "_pct_chg1",
            "_pct_chg3",
            "_pct_chg5",
            "_pct_chg10",
            "_pct_chg20",
        )
        skip_exact = {
            "symbol",
            "date",
            "board",
            "industry",
            "month",
            "is_pre_holiday",
            "is_post_holiday",
            "list_days",
            "announce_date",
            "touched_limit_up",
            "is_virtual",
            "price_1455",
            "adv20",
            "limit_pct",
            "time",
            "PE_TTM",
            "score_rank",
            "rank_amount",
            "rank_ff_turnover",
            "liquidity_score",
        }

        src_cols = [
            c
            for c in df.columns
            if c not in skip_exact
            and not any(c.startswith(p) for p in skip_prefix)
            and not any(c.endswith(s) for s in skip_suffix)
            and df[c].dtype in ("float64", "float32", "int64", "int32")
            and df[c].isna().mean() < 0.7
        ]

        if not src_cols:
            return df

        df = df.sort_values(["symbol", "date"])
        WINDOWS = (1, 3, 5, 10, 20)
        for col in src_cols:
            grp = df.groupby("symbol")[col]
            for w in WINDOWS:
                # T 日收盘后跑 pipeline, 预测 T+1 收益 → T 日数据可用, 无前视偏差
                # 绝对值差分
                abs_chg = grp.diff(w)
                if abs_chg.notna().sum() > 100:
                    df[f"{col}_chg{w}"] = abs_chg
                # 百分比变化 (ratio-scale 特征更合理: conc_90 从 0.05→0.10 和 0.50→0.55 含义不同)
                pct_chg = grp.pct_change(w, fill_method=None)
                if pct_chg.notna().sum() > 100:
                    df[f"{col}_pct_chg{w}"] = pct_chg

        return df

    # ---------------- ⑳ 截面排名特征 (IC提升最快路径) ----------------
    @staticmethod
    def _add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """对每个数值特征生成同板块同日内的百分位排名 (_xrank).

        原始特征提供绝对量纲, 截面排名提供相对位置 — 两者互补.
        仅对已有 >50% 非 NaN 且 std>0 的列生成排名.
        """
        from .cleaning_pipeline import board_of

        df = df.copy()
        if "board" not in df.columns:
            df["board"] = df["symbol"].map(board_of)

        skip_prefix = (
            "label_",
            "is_limit_up",
            "is_one_word",
            "limit_up_",
            "is_missing_",
            "is_pre_",
            "is_post_",
            "is_st",
            "is_suspended",
        )
        skip_exact = {
            "symbol",
            "date",
            "board",
            "industry",
            "name",
            "tradestatus",
            "close_hfq",
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "list_days",
            "announce_date",
            "churn_suspect",
            "score_rank",
            "rank_amount",
            "rank_ff_turnover",
            "liquidity_score",
            "market_state",
            "schema_version",
            "PE_TTM",
            "is_virtual",
            "price_1455",
            "adv20",
            "limit_pct",
            "touched_limit_up",
            "time",
            "VAR5",
            "VAR51",
        }

        src_cols = [
            c
            for c in df.columns
            if c not in skip_exact
            and not c.startswith(skip_prefix)
            and not c.endswith("_xrank")  # 不做二阶排名
            and df[c].dtype in ("float64", "float32", "int64", "int32")
        ]
        # 只对非NaN率>50%的列生成排名 (全NaN列排名无意义)
        valid = [c for c in src_cols if df[c].isna().mean() < 0.5 and df[c].std() > 0]

        if not valid:
            return df

        # 批量计算: date+board 分组的百分位排名, 直接赋值回原 df
        grp = df.groupby(["date", "board"], observed=True)
        for col in valid:
            df[f"{col}_xrank"] = grp[col].rank(pct=True)

        return df

    # ---------------- ①价量动能 (多周期ROC + 振荡器 + 背离) ----------------
    def dim01_price_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD(12,26,9) / RSI(14) / KDJ(9,3,3) / 乖离率 (5/10/20/60/120/250) / 量价背离 / 交叉信号 / 量比.

        乖离率/bias 列如果面板已预计算 (Agent 1 enrichment), 直接使用, 跳过重算.
        """

        # 检查哪些列已预计算 (逐周期, 灵活兼容部分预计算)
        _needed_bias = [w for w in BIAS_PERIODS if f"bias_{w}" not in df.columns]
        _need_5_20_cross = "bias_5_20_cross" not in df.columns
        _need_20_60_cross = "bias_20_60_cross" not in df.columns
        _need_vol_ratio = "ma_vol_ratio_5_20" not in df.columns

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            o, h, l, c = g["open"], g["high"], g["low"], g["close"]  # noqa: E741
            v = g["volume"]
            hfq_c = g.get("close_hfq", c)
            hfq_h = g.get("high_hfq", h)
            hfq_l = g.get("low_hfq", l)

            # ── 1. 多周期乖离率 (最强信号, 0.4+) ──
            for w in (5, 10, 20, 60, 120):
                g[f"bias_{w}d"] = hfq_c / hfq_c.rolling(w, min_periods=w).mean() - 1
            # ── 2. 多周期ROC (动量剖面) ──
            for w in (1, 3, 5, 10, 20, 60):
                g[f"ROC_{w}d"] = hfq_c / hfq_c.shift(w) - 1
            # 动量加速度 (ROC变化 — 二阶导)
            for w in (5, 20):
                g[f"ROC_{w}d_accel"] = g[f"ROC_{w}d"] - g[f"ROC_{w}d"].shift(w)
            # 动量一致性: 短-长ROC符号一致度
            g["momentum_consistency"] = ((g["ROC_5d"] > 0) & (g["ROC_20d"] > 0)).astype(
                float
            ) - ((g["ROC_5d"] < 0) & (g["ROC_20d"] < 0)).astype(float)

            # ── 2. 经典振荡器 ──
            # MACD(12,26,9)
            ema12, ema26 = _ema(hfq_c, 12), _ema(hfq_c, 26)
            g["MACD"] = ema12 - ema26
            g["MACD_signal"] = _ema(g["MACD"], 9)
            g["MACD_hist"] = g["MACD"] - g["MACD_signal"]
            # RSI(14)
            delta = hfq_c.diff()
            up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            dn = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
            g["RSI"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            # KDJ(9,3,3)
            hhv9 = hfq_h.rolling(9, min_periods=9).max()
            llv9 = hfq_l.rolling(9, min_periods=9).min()
            rsv = (hfq_c - llv9) / (hhv9 - llv9).replace(0, np.nan) * 100
            g["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
            g["D"] = g["K"].ewm(alpha=1 / 3, adjust=False).mean()
            g["J"] = 3 * g["K"] - 2 * g["D"]
            # 乖离率 (bias) — 仅补预计算未覆盖的周期
            for w in _needed_bias:
                g[f"bias_{w}"] = c / c.rolling(w, min_periods=w).mean() - 1
            # 交叉信号 (依赖 bias 列, 已确保存在)
            if _need_5_20_cross:
                g["bias_5_20_cross"] = (
                    np.sign(g["bias_5"] - g["bias_20"]).diff().fillna(0)
                )
            if _need_20_60_cross:
                g["bias_20_60_cross"] = (
                    np.sign(g["bias_20"] - g["bias_60"]).diff().fillna(0)
                )
            # 量价背离: 价涨量缩=1 / 价跌量增=-1
            pc = c.pct_change()
            vc = g["volume"].pct_change()
            g["pv_divergence"] = np.where(
                (pc > 0) & (vc < 0), 1, np.where((pc < 0) & (vc > 0), -1, 0)
            )
            # 量比 (5日均量 / 20日均量)
            if _need_vol_ratio:
                v = g["volume"]
                g["ma_vol_ratio_5_20"] = _safe_divide(
                    v.rolling(5, min_periods=3).mean(),
                    v.rolling(20, min_periods=10).mean(),
                )
            # Williams %R(14): (highest_high - close) / (highest_high - lowest_low) * -100
            hh14 = hfq_h.rolling(14, min_periods=14).max()
            ll14 = hfq_l.rolling(14, min_periods=14).min()
            g["WilliamsR"] = (hh14 - hfq_c) / (hh14 - ll14).replace(0, np.nan) * -100
            # CCI(20): (typical_price - SMA) / (0.015 * mean_deviation)
            tp = (hfq_h + hfq_l + hfq_c) / 3
            sma20 = tp.rolling(20, min_periods=20).mean()
            md20 = tp.rolling(20, min_periods=20).apply(
                lambda x: np.abs(x - x.mean()).mean()
            )
            g["CCI"] = (tp - sma20) / (0.015 * md20.replace(0, np.nan))
            # Bollinger %B(20): (close - lower) / (upper - lower)
            bb_ma = hfq_c.rolling(20, min_periods=20).mean()
            bb_std = hfq_c.rolling(20, min_periods=20).std()
            g["BB_upper"] = bb_ma + 2 * bb_std
            g["BB_lower"] = bb_ma - 2 * bb_std
            g["BB_pctB"] = (hfq_c - g["BB_lower"]) / (
                g["BB_upper"] - g["BB_lower"]
            ).replace(0, np.nan)
            g["BB_width"] = (g["BB_upper"] - g["BB_lower"]) / bb_ma.replace(0, np.nan)

            # ── 3. 动量质量 ──
            # 波动率调整动量 (Sharpe-like): ROC / 滚动波动率
            for w in (5, 20):
                vol_w = g["ROC_1d"].rolling(w, min_periods=w).std()
                g[f"vol_adj_mom_{w}d"] = g[f"ROC_{w}d"] / vol_w.replace(0, np.nan)
            # 日内强度: (close-open)/(high-low) 的趋势
            intra_range = (hfq_h - hfq_l).replace(0, np.nan)
            intra_strength = (hfq_c - o) / intra_range
            g["intra_strength"] = intra_strength
            g["intra_strength_ma5"] = intra_strength.rolling(5, min_periods=1).mean()
            # 新高/新低计数
            g["new_high_20d"] = (
                hfq_c == hfq_c.rolling(20, min_periods=20).max()
            ).astype(float)
            g["new_low_20d"] = (
                hfq_c == hfq_c.rolling(20, min_periods=20).min()
            ).astype(float)
            g["new_high_ratio_20d"] = g["new_high_20d"].rolling(20, min_periods=1).sum()

            # ── 4. 量价确认 ──
            # OBV 趋势: OBV SMA 偏离
            obv_dir = np.sign(hfq_c.diff())
            obv = (obv_dir * v).cumsum()
            g["OBV_trend"] = obv / obv.rolling(20, min_periods=20).mean() - 1
            g["OBV_trend_chg"] = g["OBV_trend"] - g["OBV_trend"].shift(5)
            # MFI(14): Money Flow Index (量价结合的RSI)
            tp_mfi = (hfq_h + hfq_l + hfq_c) / 3
            mf = tp_mfi * v
            pos_mf = mf.where(tp_mfi > tp_mfi.shift(1), 0)
            neg_mf = mf.where(tp_mfi < tp_mfi.shift(1), 0)
            pos_sum = pos_mf.rolling(14, min_periods=14).sum()
            neg_sum = neg_mf.rolling(14, min_periods=14).sum()
            mfr = pos_sum / neg_sum.replace(0, np.nan)
            g["MFI"] = 100 - 100 / (1 + mfr)
            # 量价同步: 价ROC方向 × 量变化方向 (连续值)
            for w in (5, 20):
                vol_roc = v / v.shift(w) - 1
                g[f"pv_sync_direct_{w}d"] = np.sign(g[f"ROC_{w}d"]) * np.sign(vol_roc)
            # 放量上涨/缩量下跌 (A股特有信号)
            g["vol_breakout"] = (
                (v > v.rolling(20, min_periods=20).mean() * 1.5) & (g["ROC_5d"] > 0.02)
            ).astype(float)
            g["vol_dry_up"] = (
                (v < v.rolling(20, min_periods=20).mean() * 0.5) & (g["ROC_5d"] < -0.02)
            ).astype(float)
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ②波动率 ----------------
    def dim02_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR(14)/收盘价 (归一化) + 布林带宽度 (20日, 2σ) + 5日振幅.

        amplitude_5d 如面板已预计算则跳过重算.
        """

        _need_amplitude_5d = "amplitude_5d" not in df.columns

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            h = g.get("high_hfq", g["high"])
            low = g.get("low_hfq", g["low"])
            c = g.get("close_hfq", g["close"])
            tr = pd.concat(
                [h - low, (h - c.shift(1)).abs(), (low - c.shift(1)).abs()], axis=1
            ).max(axis=1)
            g["ATR_pct"] = tr.rolling(14, min_periods=14).mean() / c
            ma20 = c.rolling(20, min_periods=20).mean()
            sd20 = c.rolling(20, min_periods=20).std()
            g["BB_width"] = (4 * sd20) / ma20
            # 5日振幅均值 (预计算则跳过)
            if _need_amplitude_5d:
                amp = (g["high"] - g["low"]) / g["pre_close"].replace(0, np.nan)
                g["amplitude_5d"] = amp.rolling(5, min_periods=3).mean()
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ③基本面 ----------------
    def dim03_fundamentals(
        self, df: pd.DataFrame, fundamentals: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """基础设施列: ret_pct (日收益) + limit_pct (涨停幅度).

        原 PE_log/is_negative_pe/is_STAR/relative_limit_strength 已移除 (IC<0.02, 无增量).
        fundamentals 参数保留兼容, 但不再产出特征.
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
        close_col = "close_hfq" if "close_hfq" in df.columns else "close"
        df["ret_pct"] = df.groupby("symbol")[close_col].pct_change()
        df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])]
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

    # ---------------- ⑤ 换手率与流动性 (Tushare daily_basic) ----------------
    @staticmethod
    def dim05_turnover_liquidity(df: pd.DataFrame) -> pd.DataFrame:
        """换手率/量比/股息率 — 6 个流动性因子.

        上游列: turnover_rate_f (自由流通换手率), volume_ratio (量比),
                dv_ratio (股息率), dv_ttm (股息率TTM)
        产出:
          1. turnover_f_chg_5d    — 自由流通换手率 5 日变化
          2. vol_ratio_ma5        — 量比 5 日均值
          3. vol_ratio_extreme    — 量比极端值 (>=2 视为放量)
          4. turnover_f_vs_ma20   — 换手率偏离 20 日均值
          5. dv_ratio_signal      — 股息率信号 (dv_ratio > 2% = 高股息)
          6. liquidity_score      — 综合流动性得分 (换手率+量比标准化)
        """
        out_cols = [
            "turnover_f_chg_5d",
            "vol_ratio_ma5",
            "vol_ratio_extreme",
            "turnover_f_vs_ma20",
            "dv_ratio_signal",
            "liquidity_score_v2",
        ]
        has_data = any(
            c in df.columns for c in ["turnover_rate_f", "volume_ratio", "dv_ratio"]
        )
        if not has_data:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")

            if "turnover_rate_f" in g.columns:
                tff = g["turnover_rate_f"]
                g["turnover_f_chg_5d"] = tff - tff.shift(5)
                g["turnover_f_vs_ma20"] = (
                    tff / tff.rolling(20, min_periods=5).mean() - 1
                )
            else:
                g["turnover_f_chg_5d"] = np.nan
                g["turnover_f_vs_ma20"] = np.nan

            if "volume_ratio" in g.columns:
                vr = g["volume_ratio"]
                g["vol_ratio_ma5"] = vr.rolling(5, min_periods=3).mean()
                g["vol_ratio_extreme"] = (vr >= 2.0).astype(int)
            else:
                g["vol_ratio_ma5"] = np.nan
                g["vol_ratio_extreme"] = np.nan

            if "dv_ratio" in g.columns:
                g["dv_ratio_signal"] = (g["dv_ratio"] > 2.0).astype(int)
            else:
                g["dv_ratio_signal"] = np.nan

            # 综合流动性得分: turnover_f zscore + vol_ratio zscore
            components = []
            if "turnover_rate_f" in g.columns:
                mu = g["turnover_rate_f"].rolling(60, min_periods=10).mean()
                sd = g["turnover_rate_f"].rolling(60, min_periods=10).std()
                components.append((g["turnover_rate_f"] - mu) / sd.replace(0, np.nan))
            if "volume_ratio" in g.columns:
                mu = g["volume_ratio"].rolling(60, min_periods=10).mean()
                sd = g["volume_ratio"].rolling(60, min_periods=10).std()
                components.append((g["volume_ratio"] - mu) / sd.replace(0, np.nan))
            if components:
                g["liquidity_score_v2"] = sum(components) / len(components)
            else:
                g["liquidity_score_v2"] = np.nan

            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ⑥ 估值与市值 (Tushare daily_basic) ----------------
    @staticmethod
    def dim06_valuation_size(df: pd.DataFrame) -> pd.DataFrame:
        """PE/PB/PS TTM + 市值 — 7 个估值因子.

        上游列: pe_ttm, pb, ps_ttm, total_mv, circ_mv
        产出:
          1. pe_pct           — PE TTM 截面百分位
          2. pb_pct           — PB 截面百分位
          3. pe_pb_ratio      — PE/PB 比 (成长/价值风格)
          4. mv_log           — log(总市值)
          5. mv_ma20_dev      — 市值偏离 20 日均线
          6. small_mv_premium — 小市值溢价 (截面市值倒数 rank)
          7. value_composite  — 价值综合得分 (低PE+低PB+高股息)
        """
        out_cols = [
            "pe_pct",
            "pb_pct",
            "pe_pb_ratio",
            "mv_log",
            "mv_ma20_dev",
            "small_mv_premium",
            "value_composite",
        ]
        has_data = any(c in df.columns for c in ["pe_ttm", "pb", "total_mv"])
        if not has_data:
            for c in out_cols:
                df[c] = np.nan
            return df

        # 截面百分位 (每日)
        for src, dst in [("pe_ttm", "pe_pct"), ("pb", "pb_pct")]:
            if src in df.columns:
                df[dst] = df.groupby("date")[src].rank(pct=True)

        # PE/PB ratio
        if "pe_ttm" in df.columns and "pb" in df.columns:
            df["pe_pb_ratio"] = df["pe_ttm"] / df["pb"].replace(0, np.nan)

        # 市值时序
        if "total_mv" in df.columns:
            df["mv_log"] = np.log(df["total_mv"].replace(0, np.nan))
            # 小市值溢价: 市值倒数截面rank
            df["small_mv_premium"] = df.groupby("date")["total_mv"].transform(
                lambda s: (1 / s.replace(0, np.nan)).rank(pct=True)
            )

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            if "total_mv" in g.columns:
                g["mv_ma20_dev"] = (
                    g["total_mv"] / g["total_mv"].rolling(20, min_periods=5).mean() - 1
                )
            else:
                g["mv_ma20_dev"] = np.nan
            return g

        df = _apply_per_stock(df, per_stock)

        # 价值综合: 低PE rank + 低PB rank + dv_ratio (如果存在)
        value_comps = []
        for col in ["pe_pct", "pb_pct"]:
            if col in df.columns:
                value_comps.append(1 - df[col].fillna(0.5))  # 低PE=高分
        if "dv_ratio" in df.columns:
            dv_rank = df.groupby("date")["dv_ratio"].rank(pct=True).fillna(0)
            value_comps.append(dv_rank)
        if value_comps:
            df["value_composite"] = sum(value_comps) / len(value_comps)
        else:
            df["value_composite"] = np.nan

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
              + SS金叉/多头排列 (faxian) + A04红柱/A08/获利盘 (chip)

        P18 精简:
          1. chip 优先使用面板 float_share 列, 回退 float_shares_map
          2. 二元信号 → 连续密度 (5d/10d/20d): 唯一有 IC 增益的变换
          3. VAR5/VAR51 解锁为连续信号强度特征 (原误标为中间量)

        试验过但未保留: z-score (丢失水平信息, IC 反降), days_since (冗余),
        跨指标交互 (冗余于原始值), 加速度 (噪声).
        吸筹峰为盘后专用信号, 不入特征列 (含前瞻).
        """
        from app.indicators.chip_distribution import ChipDistribution
        from app.indicators.faxian_niugu import faxian_niugu
        from app.indicators.yimeng_dingdi import yimeng_dingdi
        from app.indicators.zhuli_lasheng import zhuli_lasheng

        has_panel_float = "float_share" in df.columns

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            # ── 原始指标计算 ──
            g = zhuli_lasheng(g)
            g = yimeng_dingdi(g)
            g = faxian_niugu(g)

            # ── chip_distribution: 优先面板 float_share, 回退 float_shares_map ──
            float_shares = None
            sym = g["symbol"].iloc[0]
            if has_panel_float:
                fs = g["float_share"].dropna()
                if len(fs) > 0:
                    float_shares = fs.iloc[-1]
            if float_shares is None and float_shares_map and sym in float_shares_map:
                float_shares = float_shares_map[sym]
            if float_shares is not None:
                g = ChipDistribution().build(g, float_shares)

            # ── 二元信号 → 连续密度 (唯一经 IC 验证有增益的变换) ──
            density_sigs = [
                "吸筹",
                "洗盘",
                "拉高",
                "出货",
                "见顶",
                "顶部",
                "底部区域",
                "低位金叉",
                "SS",
                "多头排列",
                "A0A",
            ]
            for sig in density_sigs:
                if sig not in g.columns:
                    continue
                s = g[sig].astype(float)
                for w in (5, 10, 20):
                    g[f"{sig}_density_{w}d"] = s.rolling(w, min_periods=1).sum()

            # ── 剔除死权重列: 原始布尔信号 (IC=0) + 中间量 + 冗余列 ──
            drop_cols = [
                "吸筹",
                "洗盘",
                "拉高",
                "出货",
                "见顶",
                "顶部",
                "底部区域",
                "低位金叉",
                "轨迹死叉",
                "上方死叉出货",
                "红在蓝上",
                "SS",
                "多头排列",
                "A0A",
                "吸筹峰",  # 含 REF(X,-1) 前瞻, 严禁入特征
                "A01",
                "A02",
                "A03",  # chip 中间量 (A04=A03 副本)
            ]
            g = g.drop(columns=[c for c in drop_cols if c in g.columns])

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

    # ---------------- ⑪ 流通股本与涨跌停价格 (Tushare daily_basic + stk_limit) ----------------
    @staticmethod
    def dim11_float_limits(df: pd.DataFrame) -> pd.DataFrame:
        """流通股本/涨跌停价格 — 5 个因子.

        上游列: float_share, free_share (daily_basic),
                up_limit_raw, down_limit_raw (stk_limit)
        产出:
          1. float_share_ratio   — 流通股本/总股本 (筹码供给压力)
          2. free_float_pct      — 自由流通占比
          3. limit_dist_pct      — 现价距涨停价距离 (%)
          4. limit_down_dist_pct — 现价距跌停价距离 (%)
          5. limit_asymmetry     — 涨跌停距离不对称 (涨停距/跌停距, >1=偏多)
        """
        out_cols = [
            "float_share_ratio",
            "free_float_pct",
            "limit_dist_pct",
            "limit_down_dist_pct",
            "limit_asymmetry",
        ]
        has_data = any(c in df.columns for c in ["float_share", "up_limit_raw"])
        if not has_data:
            for c in out_cols:
                df[c] = np.nan
            return df

        # 流通股本/总股本比
        if "float_share" in df.columns and "total_share" in df.columns:
            df["float_share_ratio"] = df["float_share"] / df["total_share"].replace(
                0, np.nan
            )
        elif "float_share" in df.columns:
            df["float_share_ratio"] = np.nan

        # 自由流通占比
        if "free_share" in df.columns and "total_share" in df.columns:
            df["free_float_pct"] = df["free_share"] / df["total_share"].replace(
                0, np.nan
            )
        elif "free_share" in df.columns:
            df["free_float_pct"] = np.nan

        # 涨跌停距离
        if "up_limit_raw" in df.columns and "down_limit_raw" in df.columns:
            close = df["close"] if "close" in df.columns else df["close_hfq"]
            df["limit_dist_pct"] = (df["up_limit_raw"] - close) / close.replace(
                0, np.nan
            )
            df["limit_down_dist_pct"] = (close - df["down_limit_raw"]) / close.replace(
                0, np.nan
            )
            df["limit_asymmetry"] = (df["up_limit_raw"] - close) / (
                close - df["down_limit_raw"]
            ).replace(0, np.nan)
        elif "up_limit_raw" in df.columns:
            close = df["close"] if "close" in df.columns else df["close_hfq"]
            df["limit_dist_pct"] = (df["up_limit_raw"] - close) / close.replace(
                0, np.nan
            )

        return df

    # ---------------- ⑫均线系统 (多周期关系) ----------------
    def dim12_ma_system(self, df: pd.DataFrame) -> pd.DataFrame:
        """MA偏离 + 多头得分 + 量价同步.

        MA距离6个已捕捉全部趋势信号(|IC| 0.16-0.48),
        LightGBM树自动学习交叉/斜率组合, 无需显式派生。
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            c = g.get("close_hfq", g["close"])
            v = g["volume"]

            for w in (5, 10, 20, 60, 120, 250):
                ma = c.rolling(w, min_periods=w).mean()
                g[f"MA{w}_dist"] = c / ma - 1

            mas = {w: c.rolling(w, min_periods=w).mean() for w in (5, 10, 20, 60, 120)}
            g["ma_bull_score"] = (
                (mas[5] > mas[10]).astype(float) * 4
                + (mas[10] > mas[20]).astype(float) * 3
                + (mas[20] > mas[60]).astype(float) * 2
                + (mas[60] > mas[120]).astype(float) * 1
            )

            for w in (5, 20):
                vol_ma = v.rolling(w, min_periods=w).mean()
                g[f"vol_ratio_MA{w}"] = v / vol_ma.replace(0, np.nan)
                price_dir = np.sign(g[f"MA{w}_dist"])
                vol_dir = np.sign(v / vol_ma.replace(0, np.nan) - 1)
                g[f"pv_sync_{w}d"] = price_dir * vol_dir

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

    # ---------------- ⑯ K线形态 (连续强度, 非二值) ----------------
    def dim16_candlestick(self, df: pd.DataFrame) -> pd.DataFrame:
        """K线形态 → 连续强度特征 (行业标准: 罕见二值→平滑密度, IC 可测).

        日频二值模式 (hammer/engulfing/star) 先检测, 再半衰期累积为密度得分:
          - bullish_intensity: 看涨形态加权密度 (hammer + bullish_engulfing + morning_star)
          - bearish_intensity: 看跌形态加权密度
          - net_intensity:     bullish - bearish (多空净信号)
          - body_shadow_ratio_ma: 实体/影线比值的均线 (波动率+市场质量)
          - reversal_intensity: 反转形态密度 (hammer + shooting_star 总密度)
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            o, h, lo, c = g["open"], g["high"], g["low"], g["close"]
            body = (c - o).abs()
            total_range = (h - lo).replace(0, np.nan)
            upper_shadow = h - np.maximum(o, c)
            lower_shadow = np.minimum(o, c) - lo
            is_white = (c > o).astype(int)
            is_black = (c < o).astype(int)

            # --- 日频二值检测 (仅中间变量, 不输出) ---
            bullish_engulfing = (
                (is_black.shift(1) == 1)
                & (is_white == 1)
                & (o <= c.shift(1))
                & (c >= o.shift(1))
            ).astype(float)
            bearish_engulfing = (
                (is_white.shift(1) == 1)
                & (is_black == 1)
                & (o >= c.shift(1))
                & (c <= o.shift(1))
            ).astype(float)
            hammer = (
                (body < 0.3 * total_range)
                & (lower_shadow > 2 * body)
                & (upper_shadow < 0.3 * body)
            ).astype(float)
            shooting_star = (
                (body < 0.3 * total_range)
                & (upper_shadow > 2 * body)
                & (lower_shadow < 0.3 * body)
            ).astype(float)
            mid_prev = (o.shift(2) + c.shift(2)) / 2
            morning_star = (
                (is_black.shift(2) == 1)
                & (body.shift(1) < 0.5 * body.shift(2))
                & (is_white == 1)
                & (c > mid_prev)
            ).astype(float)
            evening_star = (
                (is_white.shift(2) == 1)
                & (body.shift(1) < 0.5 * body.shift(2))
                & (is_black == 1)
                & (c < mid_prev)
            ).astype(float)

            bullish_raw = bullish_engulfing + hammer + morning_star
            bearish_raw = bearish_engulfing + shooting_star + evening_star
            g["bullish_engulfing"] = bullish_engulfing
            g["bearish_engulfing"] = bearish_engulfing
            g["hammer"] = hammer
            g["shooting_star"] = shooting_star
            g["morning_star"] = morning_star
            g["evening_star"] = evening_star

            # --- 半衰期累积密度 (10/20/60 日, half-life=5) ---
            decay = 0.5 ** (1.0 / 5)  # 5日半衰
            bull_density = bullish_raw.ewm(
                alpha=1 - decay, min_periods=1, adjust=False
            ).mean()
            bear_density = bearish_raw.ewm(
                alpha=1 - decay, min_periods=1, adjust=False
            ).mean()
            for w in (10, 20, 60):
                g[f"bullish_intensity_{w}d"] = bull_density.rolling(
                    w, min_periods=1
                ).mean()
                g[f"bearish_intensity_{w}d"] = bear_density.rolling(
                    w, min_periods=1
                ).mean()
                g[f"net_intensity_{w}d"] = (
                    g[f"bullish_intensity_{w}d"] - g[f"bearish_intensity_{w}d"]
                )

            # --- 反转形态密度 (hammer+shooting_star 总活跃度) ---
            reversal_raw = hammer + shooting_star
            rev_density = reversal_raw.ewm(
                alpha=1 - decay, min_periods=1, adjust=False
            ).mean()
            g["reversal_intensity_20d"] = rev_density.rolling(20, min_periods=1).mean()

            # --- 实体/影线均值 (K线质量 + 波动率特征) ---
            bs_ratio = body / total_range
            g["body_shadow_ratio_5d"] = bs_ratio.rolling(5, min_periods=1).mean()
            g["body_shadow_ratio_20d"] = bs_ratio.rolling(20, min_periods=1).mean()

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

    # ---------------- ⑳ 短周期特征 (专攻 1d 预测信噪比) ----------------
    def dim20_short_horizon(self, df: pd.DataFrame) -> pd.DataFrame:
        """短周期特征: 隔夜跳空/日内动量/连涨连跌/均值回复 — 1d 预测专用信号.

        1d 预测信噪比极低 (日波动 ~3.4% vs 信号 ~0.1%), 需要捕捉日内微观结构.
        全量计算用 groupby("symbol") rolling, 无前瞻偏差.
        """

        def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            c, o, h, l, v = g["close"], g["open"], g["high"], g["low"], g["volume"]  # noqa: E741
            pc = c.shift(1)

            # 隔夜跳空方向 (前收→今开)
            g["overnight_ret"] = (o / pc - 1).replace([np.inf, -np.inf], np.nan) * 100

            # 日内动量 (今开→今收)
            g["intraday_momentum"] = (c / o - 1).replace(
                [np.inf, -np.inf], np.nan
            ) * 100

            # 日内振幅 (high-low spread, 反映日内博弈烈度)
            g["intraday_amplitude"] = (h / l - 1).replace(
                [np.inf, -np.inf], np.nan
            ) * 100

            # 连涨/连跌天数 (最近 5 日)
            up = (c > pc).astype(int)
            streak = up.copy()
            for i in range(1, min(6, len(g))):
                streak.iloc[i:] = (streak.iloc[i:] + 1) * (
                    up.iloc[i:] == up.iloc[i:].shift(i)
                )
            g["up_streak"] = up * streak  # 正=连涨天数, 0=当日下跌
            g["dn_streak"] = (1 - up) * streak  # 正=连跌天数, 0=当日上涨

            # 5 日收益率均值回复信号 (反转效应)
            ret_5d = c / c.shift(5) - 1
            g["ret_reversal_5d"] = -ret_5d  # 正=超跌反弹预期, 负=超涨回调预期

            # 隔夜跳空 vs 5日均值 (异常跳空检测)
            gap_ma5 = (o / pc - 1).rolling(5, min_periods=5).mean()
            g["gap_vs_ma5"] = ((o / pc - 1) - gap_ma5).replace(
                [np.inf, -np.inf], np.nan
            ) * 100

            # 尾盘拉升检测 (最后30分钟无法直接计算, 用日内动量+次日开盘代理)
            # close vs high of day: 收盘在日内高位 → 尾盘强势
            g["close_vs_high"] = (c / h - 1).replace([np.inf, -np.inf], np.nan) * 100
            g["close_vs_low"] = (c / l - 1).replace([np.inf, -np.inf], np.nan) * 100

            # 量价背离: 涨但缩量 / 跌但放量 (1d 反转信号)
            ret = c / pc - 1
            vol_chg = v / v.shift(1).replace(0, np.nan)
            g["vol_price_divergence"] = np.sign(ret) * (1.0 / vol_chg.clip(0.1, 10) - 1)

            return g

        return _apply_per_stock(df, _per_stock)

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

    # (原 DIM20 chip_proxy 已合并到 DIM21 — CYQ NaN 时 OHLCV 代理补位)

    # ---------------- ㉑ CYQ 筹码分布 (OHLCV 代理补位) ----------------
    @staticmethod
    def dim21_chip_tushare(df: pd.DataFrame) -> pd.DataFrame:
        """CYQ 筹码特征 (7→15 个) — Tushare cyq_perf 主源, OHLCV 代理补位.

        Tushare cyq_perf 提供: cost_5pct/15pct/50pct/85pct/95pct, weight_avg,
        his_low, his_high, winner_rate (获利盘比例 %, 等价 benefit_part×100).
        benefit_part 和 pct_90_con 从 Tushare 列直接推导:
          benefit_part = winner_rate / 100
          pct_90_con   = (cost_95pct - cost_5pct) / cost_50pct

        Tushare 无数据时降级为旧 calculator 列, 再缺则降级为 OHLCV 代理.
        """
        TUSHARE_REQUIRED = ["cost_50pct", "winner_rate"]
        CALC_REQUIRED = ["pct_90_con", "benefit_part", "cost_50pct"]
        has_tushare = all(c in df.columns for c in TUSHARE_REQUIRED)
        has_calc = all(c in df.columns for c in CALC_REQUIRED)

        def per_stock_cyq(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            c = g.get("close_hfq", g["close"])
            c50 = g["cost_50pct"]
            bp = g["benefit_part"]
            conc90 = g["pct_90_con"]

            c5 = g.get("cost_5pct", pd.Series(np.nan, index=g.index))
            c95 = g.get("cost_95pct", pd.Series(np.nan, index=g.index))
            c15 = g.get("cost_15pct", pd.Series(np.nan, index=g.index))
            c85 = g.get("cost_85pct", pd.Series(np.nan, index=g.index))

            g["conc_90"] = conc90
            g["benefit_part"] = bp
            g["cost_bias"] = (c - c50) / c50.replace(0, np.nan)
            g["cost_spread"] = (c95 - c5) / c50.replace(0, np.nan)
            if c15.notna().any() and c85.notna().any():
                g["chip_skew"] = (c50 - c15) / (c85 - c50).replace(0, np.nan)
            else:
                g["chip_skew"] = np.nan
            g["conc_trend_20d"] = conc90 / conc90.shift(20).replace(0, np.nan)
            g["benefit_trend_5d"] = bp - bp.shift(5)

            is_shrinking = (conc90 < conc90.shift(1)).astype(int)
            g["conc_streak"] = is_shrinking.groupby(
                (is_shrinking == 0).cumsum()
            ).cumsum()
            g["conc_streak_3d"] = (g["conc_streak"] >= 3).astype(int)
            pct70 = g.get("pct_70_con", conc90)
            is_shrinking_70 = (pct70 < pct70.shift(1)).astype(int)
            g["conc70_streak"] = is_shrinking_70.groupby(
                (is_shrinking_70 == 0).cumsum()
            ).cumsum()
            g["conc70_streak_3d"] = (g["conc70_streak"] >= 3).astype(int)
            was_streak3 = (g["conc_streak"].shift(1) >= 3).astype(int)
            g["conc_reversal"] = was_streak3 * (1 - is_shrinking)
            g["cost50_rank"] = 0.0
            bp_ma60 = bp.rolling(60, min_periods=20).mean()
            g["benefit_vs_ma60"] = (bp - bp_ma60) / bp_ma60.replace(0, np.nan)
            g["benefit_dir_5d"] = np.sign(bp - bp.shift(5))
            return g

        def per_stock_proxy(g: pd.DataFrame) -> pd.DataFrame:
            """OHLCV 筹码代理 (原 DIM20) — CYQ NaN 时补位."""
            g = g.sort_values("date")
            c, v = g["close"], g["volume"]
            to = g.get("turnover_rate", v / v.rolling(20).mean())
            g["conc_90"] = 1 - to.rolling(20).std() / to.rolling(20).mean().replace(
                0, 1
            )
            g["benefit_part"] = np.nan  # OHLCV 无法推导获利盘
            g["cost_bias"] = c / c.rolling(60).mean() - 1  # 价格偏离60日均线替代
            g["cost_spread"] = np.nan
            g["chip_skew"] = np.nan
            g["conc_trend_20d"] = g["conc_90"] / g["conc_90"].shift(20).replace(0, 1)
            g["benefit_trend_5d"] = np.nan
            # streak/拐点 对 OHLCV 代理无意义, 填 NaN
            for col in [
                "conc_streak",
                "conc_streak_3d",
                "conc70_streak",
                "conc70_streak_3d",
                "conc_reversal",
                "benefit_vs_ma60",
                "benefit_dir_5d",
            ]:
                g[col] = np.nan
            g["cost50_rank"] = 0.0
            return g

        if has_tushare:
            # Tushare cyq_perf 主源: 从 winner_rate + cost 列推导 benefit_part/pct_90_con
            if "benefit_part" not in df.columns:
                df["benefit_part"] = df["winner_rate"] / 100.0
            if "pct_90_con" not in df.columns:
                df["pct_90_con"] = (df["cost_95pct"] - df["cost_5pct"]) / df[
                    "cost_50pct"
                ].replace(0, np.nan)
            df = _apply_per_stock(df, per_stock_cyq)
        elif has_calc:
            # 旧 calculator 列 (无 Tushare 时的降级)
            df = _apply_per_stock(df, per_stock_cyq)
        else:
            # 无任何 CYQ 数据 → 全量 OHLCV 代理
            df = _apply_per_stock(df, per_stock_proxy)

        # cost50_rank (截面)
        if "cost_50pct" in df.columns:
            if "board" in df.columns:
                grp = df.groupby(["date", "board"], observed=True)
                df["cost50_rank"] = grp["cost_50pct"].rank(pct=True).fillna(0.5)
            else:
                df["cost50_rank"] = (
                    df.groupby("date")["cost_50pct"].rank(pct=True).fillna(0.5)
                )

        # NaN 列的 OHLCV 回退: 对每个 stock, conc_90/cost_bias 等为 NaN 时用 proxy 值
        nan_mask = df["conc_90"].isna()
        if nan_mask.any() and (has_tushare or has_calc):
            # 只对含 NaN 的 stock 计算 proxy
            nan_symbols = df.loc[nan_mask, "symbol"].unique()
            proxy_dfs = []
            for sym in nan_symbols:
                g = df[df["symbol"] == sym].copy()
                g = per_stock_proxy(g)
                proxy_dfs.append(g)
            if proxy_dfs:
                proxy = pd.concat(proxy_dfs)
                # 只填充 NaN 位置 (按 index 对齐, 安全网 #5)
                for col in ["conc_90", "cost_bias", "conc_trend_20d"]:
                    if col in proxy.columns and col in df.columns:
                        fill_mask = df[col].isna()
                        common = fill_mask & fill_mask.index.isin(proxy.index)
                        df.loc[common, col] = proxy.loc[common, col].values

        return df

    # ---------------- ㉒ 基本面PIT (fina_indicator, Alt-3) ----------------
    @staticmethod
    def dim22_fundamental_pit(df: pd.DataFrame) -> pd.DataFrame:
        """基本面PIT — 时序变化为主 (QoQ/YoY/趋势), 截面为辅助.

        CLAUDE.md: 特征因子看连续交易日时序变化, 不是截面.
        上游列: roe, roa, gross_margin, net_margin, eps_yoy, rev_yoy, profit_yoy,
                op_cf_ratio, debt_ratio, current_ratio, asset_turnover, inventory_turnover
        产出 (12 时序因子):
          1. roe_qoq           — ROE 季环比变化
          2. roa_qoq           — ROA 季环比变化
          3. margin_chg         — 毛利率季环比
          4. growth_accel       — 营收增速加速度 (rev_yoy - rev_yoy_shift)
          5. profit_accel       — 利润增速加速度
          6. debt_leveraging    — 负债率季环比 (正=加杠杆)
          7. efficiency_chg     — 资产周转率季环比
          8. ocf_stability      — 经营现金流/营收的稳定性 (约20季=1260个交易日滚动CV)
          9. roe_trend_4q       — ROE 4季趋势方向
         10. margin_trend_4q    — 毛利率 4季趋势
         11. rev_yoy_trend      — 营收增速 4季趋势斜率
         12. quality_momentum   — 质量动量: ROE趋势 + 毛利率趋势 - 负债趋势
        """
        fin_cols = [
            "roe",
            "roa",
            "gross_margin",
            "net_margin",
            "eps_yoy",
            "rev_yoy",
            "profit_yoy",
            "op_cf_ratio",
            "debt_ratio",
            "current_ratio",
            "asset_turnover",
            "inventory_turnover",
        ]
        out_cols = [
            "roe_qoq",
            "roa_qoq",
            "margin_chg",
            "growth_accel",
            "profit_accel",
            "debt_leveraging",
            "efficiency_chg",
            "ocf_stability",
            "roe_trend_4q",
            "margin_trend_4q",
            "rev_yoy_trend",
            "quality_momentum",
        ]
        has_fin = any(c in df.columns for c in fin_cols)
        if not has_fin:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            q = 63  # ~1 quarter in trading days

            # === 时序变化 (PRIMARY) ===
            # QoQ changes (季环比)
            for col, out in [
                ("roe", "roe_qoq"),
                ("roa", "roa_qoq"),
                ("gross_margin", "margin_chg"),
                ("debt_ratio", "debt_leveraging"),
                ("asset_turnover", "efficiency_chg"),
            ]:
                if col in g.columns:
                    g[out] = g[col] - g[col].shift(q)

            # Growth acceleration (增速加速度)
            for col, out in [
                ("rev_yoy", "growth_accel"),
                ("profit_yoy", "profit_accel"),
            ]:
                if col in g.columns:
                    g[out] = g[col] - g[col].shift(q)

            # === 趋势 (4季窗口) ===
            for col, out in [
                ("roe", "roe_trend_4q"),
                ("gross_margin", "margin_trend_4q"),
                ("rev_yoy", "rev_yoy_trend"),
            ]:
                if col in g.columns:
                    # Linear regression slope over last 4 quarters
                    s = g[col]
                    g[out] = (s - s.shift(4 * q)) / 4  # simple proxy: total change / 4

            # === 稳定性 (20 季度滚动 CV, 约 20*63=1260 个交易日) ===
            # 数据经 merge_asof(direction=backward) 为日频填充, 常量段内 rolling 无意义,
            # 必须用足够长的窗口 (>= 4 季度) 以捕捉真实季度间变化.
            if "op_cf_ratio" in g.columns:
                roll_std = g["op_cf_ratio"].rolling(20 * q, min_periods=4 * q).std()
                roll_mean = g["op_cf_ratio"].rolling(20 * q, min_periods=4 * q).mean()
                g["ocf_stability"] = -roll_std / roll_mean.replace(0, np.nan)

            # === 质量动量 (复合) ===
            components = []
            if "roe_trend_4q" in g.columns:
                components.append(g["roe_trend_4q"].fillna(0))
            if "margin_trend_4q" in g.columns:
                components.append(g["margin_trend_4q"].fillna(0))
            if "debt_leveraging" in g.columns:
                components.append(-g["debt_leveraging"].fillna(0))  # 去杠杆=正信号
            if components:
                g["quality_momentum"] = sum(components) / len(components)

            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ㉓ 股东户数+户均持股 (Alt-5) ----------------
    @staticmethod
    def dim23_shareholder_structure(df: pd.DataFrame) -> pd.DataFrame:
        """从股东户数 PIT 对齐后的列提取 8 个筹码结构因子.

        上游列: holder_count, avg_shares_per_holder (由 panel_builder 按 announce_date merge_asof)

        产出:
          1. holder_count_log    — log(股东户数)
          2. holder_count_qoq    — 股东户数环比变化
          3. holder_count_yoy    — 股东户数同比变化
          4. holder_qoq_accel    — 环比变化加速 (二阶导)
          5. avg_shares_log      — log(户均持股)
          6. avg_shares_qoq      — 户均持股环比
          7. avg_shares_yoy      — 户均持股同比
          8. holder_concentration_zscore — 截面集中度标准化
        """
        has_holder = (
            "holder_count" in df.columns or "avg_shares_per_holder" in df.columns
        )
        out_cols = [
            "holder_count_log",
            "holder_count_qoq",
            "holder_count_yoy",
            "holder_qoq_accel",
            "avg_shares_log",
            "avg_shares_qoq",
            "avg_shares_yoy",
            "holder_concentration_zscore",
        ]
        if not has_holder:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")

            if "holder_count" in g.columns:
                hc = g["holder_count"]
                g["holder_count_log"] = np.log(hc.replace(0, np.nan))
                # 环比: 需 shift 约 63 个交易日 (一个季度)
                g["holder_count_qoq"] = (hc - hc.shift(63)) / hc.shift(63).replace(
                    0, np.nan
                )
                g["holder_count_yoy"] = (hc - hc.shift(252)) / hc.shift(252).replace(
                    0, np.nan
                )
                # 二阶导: QoQ 的变化
                g["holder_qoq_accel"] = g["holder_count_qoq"] - g[
                    "holder_count_qoq"
                ].shift(63)

            if "avg_shares_per_holder" in g.columns:
                ash = g["avg_shares_per_holder"]
                g["avg_shares_log"] = np.log(ash.replace(0, np.nan))
                g["avg_shares_qoq"] = (ash - ash.shift(63)) / ash.shift(63).replace(
                    0, np.nan
                )
                g["avg_shares_yoy"] = (ash - ash.shift(252)) / ash.shift(252).replace(
                    0, np.nan
                )

            return g

        df = _apply_per_stock(df, per_stock)

        # 截面集中度 zscore (按 date 分组)
        if "holder_count_log" in df.columns:
            mu = df.groupby("date")["holder_count_log"].transform("mean")
            sd = df.groupby("date")["holder_count_log"].transform("std")
            df["holder_concentration_zscore"] = -(
                df["holder_count_log"] - mu
            ) / sd.replace(0, np.nan)

        return df

    # ---------------- ㉔ 融资融券 (Alt-2) ----------------
    @staticmethod
    def dim24_margin_trading(df: pd.DataFrame) -> pd.DataFrame:
        """从融资融券日频数据提取 7 个杠杆情绪因子.

        上游列: margin_balance, short_balance, margin_buy_amt (由 panel_builder merge)

        产出:
          1. margin_balance_chg_1d    — 融资余额日变动
          2. margin_balance_chg_5d    — 融资余额 5 日变动
          3. short_balance_ratio      — 融券/融资比
          4. margin_buy_ratio         — 融资买入占成交比
          5. margin_balance_ma20_dev  — 融资余额偏离 20 日均线
          6. margin_balance_yoy       — 融资余额同比
          7. margin_pressure_score    — 杠杆压力: (融券比+融资增速) 综合
        """
        has_margin = "margin_balance" in df.columns
        out_cols = [
            "margin_balance_chg_1d",
            "margin_balance_chg_5d",
            "short_balance_ratio",
            "margin_buy_ratio",
            "margin_balance_ma20_dev",
            "margin_balance_yoy",
            "margin_pressure_score",
        ]
        if not has_margin:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            mb = g.get("margin_balance", pd.Series(np.nan, index=g.index))
            sb = g.get("short_balance", pd.Series(np.nan, index=g.index))
            mba = g.get("margin_buy_amt", pd.Series(np.nan, index=g.index))
            amount = g.get("amount", pd.Series(np.nan, index=g.index))

            g["margin_balance_chg_1d"] = mb.diff(1) / mb.shift(1).replace(0, np.nan)
            g["margin_balance_chg_5d"] = mb.diff(5) / mb.shift(5).replace(0, np.nan)
            g["short_balance_ratio"] = sb / mb.replace(0, np.nan)
            g["margin_buy_ratio"] = mba / amount.replace(0, np.nan)
            # 融资余额偏离 20 日均线
            g["margin_balance_ma20_dev"] = mb / mb.rolling(20, min_periods=5).mean() - 1
            g["margin_balance_yoy"] = mb.diff(252) / mb.shift(252).replace(0, np.nan)
            # 杠杆压力综合
            g["margin_pressure_score"] = (
                g["short_balance_ratio"].fillna(0) * 0.4
                + g["margin_balance_chg_5d"].fillna(0).clip(-0.5, 0.5) * 0.6
            )
            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ㉕ 北向资金 (Alt-1) ----------------
    @staticmethod
    def dim25_northbound(df: pd.DataFrame) -> pd.DataFrame:
        """从北向资金日频数据提取 7 个外资流向因子.

        上游列: north_net_buy, north_net_buy_sh, north_net_buy_sz,
                north_buy_amt_sh, north_buy_amt_sz (由 panel_builder merge)

        注意: 北向是市场级数据, 所有个股同值.

        产出:
          1. north_net_buy_5d       — 5日累计净买入
          2. north_net_buy_20d      — 20日累计净买入
          3. north_net_buy_streak   — 连续净买入天数 (近似=符号连续为正的天数)
          4. north_buy_ratio        — 北向买入/成交额 (代理)
          5. north_sh_sz_divergence — 沪深分化 (沪净买-深净买)
          6. north_momentum_5d      — 北向净买入 5 日变化加速度
          7. north_flow_zscore      — 时序标准化净流入 (rolling 20d zscore, 非截面)
        """
        has_north = any(c in df.columns for c in ["north_net_buy", "north_net_buy_sh"])
        out_cols = [
            "north_net_buy_5d",
            "north_net_buy_20d",
            "north_net_buy_streak",
            "north_buy_ratio",
            "north_sh_sz_divergence",
            "north_momentum_5d",
            "north_flow_zscore",
        ]
        if not has_north:
            for c in out_cols:
                df[c] = np.nan
            return df

        # 如果有分市场列, 合并出总净买入
        if "north_net_buy" not in df.columns:
            sh = df.get("north_net_buy_sh", pd.Series(0, index=df.index)).fillna(0)
            sz = df.get("north_net_buy_sz", pd.Series(0, index=df.index)).fillna(0)
            df["north_net_buy"] = sh + sz

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            nb = g.get("north_net_buy", pd.Series(np.nan, index=g.index))

            g["north_net_buy_5d"] = nb.rolling(5, min_periods=1).sum()
            g["north_net_buy_20d"] = nb.rolling(20, min_periods=1).sum()
            # 连续净买入天数: 正号分组累计, 负号/零归零 (向量化)
            sign_pos = (np.sign(nb.fillna(0)) > 0).astype(int)
            g["north_net_buy_streak"] = sign_pos.groupby(
                (sign_pos == 0).cumsum()
            ).cumsum()
            # 动量加速度
            g["north_momentum_5d"] = nb.diff(5) - nb.diff(10).shift(5)
            return g

        df = _apply_per_stock(df, per_stock)

        # 沪深分化 (全市场日频列, 非 stock-level — 需按日期广播)
        if "north_net_buy_sh" in df.columns and "north_net_buy_sz" in df.columns:
            df["north_sh_sz_divergence"] = df["north_net_buy_sh"].fillna(0) - df[
                "north_net_buy_sz"
            ].fillna(0)

        # 北向买入比 (代理: 净买入绝对值/全市场成交额)
        if "north_net_buy" in df.columns and "amount" in df.columns:
            daily_total_amt = df.groupby("date")["amount"].transform("sum")
            df["north_buy_ratio"] = df["north_net_buy"].abs() / daily_total_amt.replace(
                0, np.nan
            )

        # 北向是市场级数据 (所有个股同值), 截面 zscore = 0/0 = NaN.
        # 改用时间序列 zscore: 对单序列 rolling 20 日标准化.
        if "north_net_buy" in df.columns:
            nb_by_date = df.groupby("date")["north_net_buy"].first()
            nb_by_date = nb_by_date.sort_index()
            nb_mu = nb_by_date.rolling(20, min_periods=5).mean()
            nb_sd = nb_by_date.rolling(20, min_periods=5).std()
            ts_z = (nb_by_date - nb_mu) / nb_sd.replace(0, np.nan)
            df["north_flow_zscore"] = df["date"].map(ts_z)

        return df

    # ---------------- ㉖ 龙虎榜增强 (Alt-4) ----------------
    @staticmethod
    def dim26_lhb_enhanced(df: pd.DataFrame) -> pd.DataFrame:
        """从龙虎榜数据提取 5 个机构行为因子 (dim18 增强版).

        上游列: lhb_net_buy, lhb_institutional_net_buy, lhb_institutional_count,
                lhb_buy_amt, lhb_sell_amt (由 panel_builder merge)

        产出:
          1. lhb_inst_net_buy_5d   — 近 5 日机构净买入 (取最近一次上榜值前向填充后 rolling sum)
          2. lhb_inst_net_buy_20d  — 近 20 日机构净买入
          3. lhb_inst_count_5d     — 近 5 日机构席位上榜次数
          4. lhb_inst_buy_ratio    — 机构买入/总上榜买入
          5. lhb_abnormal_score    — 异常上榜得分 (偏离均值 2σ)
        """
        has_lhb = any(
            c in df.columns for c in ["lhb_net_buy", "lhb_institutional_net_buy"]
        )
        out_cols = [
            "lhb_inst_net_buy_5d",
            "lhb_inst_net_buy_20d",
            "lhb_inst_count_5d",
            "lhb_inst_buy_ratio",
            "lhb_abnormal_score",
        ]
        if not has_lhb:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")

            # 机构净买入 (取最近上榜值 ffill)
            if "lhb_institutional_net_buy" in g.columns:
                inst_net = g["lhb_institutional_net_buy"].replace(0, np.nan).ffill()
                g["lhb_inst_net_buy_5d"] = inst_net.rolling(5, min_periods=1).sum()
                g["lhb_inst_net_buy_20d"] = inst_net.rolling(20, min_periods=1).sum()
            else:
                g["lhb_inst_net_buy_5d"] = np.nan
                g["lhb_inst_net_buy_20d"] = np.nan

            # 机构席位出现次数
            if "lhb_institutional_count" in g.columns:
                g["lhb_inst_count_5d"] = (
                    g["lhb_institutional_count"]
                    .fillna(0)
                    .rolling(5, min_periods=1)
                    .sum()
                )
            else:
                g["lhb_inst_count_5d"] = np.nan

            # 机构买入占比
            if "lhb_buy_amt" in g.columns and "lhb_institutional_buy" in g.columns:
                total_buy = g["lhb_buy_amt"].replace(0, np.nan).ffill()
                inst_buy = g.get(
                    "lhb_institutional_buy", pd.Series(np.nan, index=g.index)
                ).fillna(0)
                g["lhb_inst_buy_ratio"] = inst_buy / total_buy.replace(0, np.nan)
            elif "lhb_inst_buy_ratio" in g.columns:
                pass  # 已由 upstream 提供
            else:
                g["lhb_inst_buy_ratio"] = np.nan

            # 异常上榜得分
            if "lhb_net_buy" in g.columns:
                net = g["lhb_net_buy"].fillna(0)
                roll_mu = net.rolling(60, min_periods=20).mean()
                roll_sd = net.rolling(60, min_periods=20).std()
                g["lhb_abnormal_score"] = (
                    (net - roll_mu) / roll_sd.replace(0, np.nan)
                ).abs()
            else:
                g["lhb_abnormal_score"] = np.nan

            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ㉗ 行业级 alt 数据聚合 (Alt-7a) ----------------
    @staticmethod
    def dim27_industry_flow(df: pd.DataFrame) -> pd.DataFrame:
        """行业级 alt 数据聚合 — 时序变化为主 (行业资金流方向/加速度).

        CLAUDE.md: 看连续交易日时序变化, 不只看截面.
        上游: industry列 + margin_balance / holder_count_qoq / north_net_buy / lhb_net_buy

        产出 (6 时序行业因子):
          1. ind_margin_chg_5d     — 行业融资余额 5 日平均变化率 (正=行业加杠杆)
          2. ind_margin_accel      — 行业融资变化加速度 (5日 vs 20日)
          3. ind_holder_trend_20d  — 行业股东户数 20 日趋势 (正=散户化/筹码分散)
          4. ind_north_chg_5d      — 行业北向资金 5 日净流入变化
          5. ind_lhb_net_flow_5d   — 行业龙虎榜机构净买入 5 日累计
          6. ind_capital_flow      — 行业资金流综合: 融资+北向+股东方向
        """
        out_cols = [
            "ind_margin_chg_5d",
            "ind_margin_accel",
            "ind_holder_trend_20d",
            "ind_north_chg_5d",
            "ind_lhb_net_flow_5d",
            "ind_capital_flow",
        ]
        if "industry" not in df.columns:
            for c in out_cols:
                df[c] = np.nan
            return df

        # 1. 融资融券行业时序
        if "margin_balance" in df.columns:

            def mb_stock(g):
                g = g.sort_values("date")
                mb = g["margin_balance"]
                g["_mb_chg_1d"] = (mb - mb.shift(1)) / mb.shift(1).replace(0, np.nan)
                return g

            df = _apply_per_stock(df, mb_stock)
            # Aggregate to industry
            ind_mb = (
                df.groupby(["date", "industry"], observed=True)["_mb_chg_1d"]
                .mean()
                .reset_index()
            )
            ind_mb = ind_mb.sort_values(["industry", "date"])
            ind_mb["ind_margin_chg_5d"] = (
                ind_mb.groupby("industry", observed=True)["_mb_chg_1d"]
                .rolling(5, min_periods=3)
                .mean()
                .reset_index(level=0, drop=True)
            )
            ind_mb["ind_margin_accel"] = ind_mb.groupby("industry", observed=True)[
                "ind_margin_chg_5d"
            ].shift(10)
            ind_mb = ind_mb.reset_index(drop=True)
            df = df.merge(
                ind_mb[["date", "industry", "ind_margin_chg_5d", "ind_margin_accel"]],
                on=["date", "industry"],
                how="left",
            )
            df = df.drop(columns=["_mb_chg_1d"], errors="ignore")
        else:
            df["ind_margin_chg_5d"] = np.nan
            df["ind_margin_accel"] = np.nan

        # 2. 股东户数行业时序
        if "holder_count_qoq" in df.columns:
            ind_hc = (
                df.groupby(["date", "industry"], observed=True)["holder_count_qoq"]
                .mean()
                .reset_index()
            )
            ind_hc = ind_hc.sort_values(["industry", "date"])
            ind_hc["ind_holder_trend_20d"] = (
                ind_hc.groupby("industry", observed=True)["holder_count_qoq"]
                .rolling(20, min_periods=5)
                .mean()
                .reset_index(level=0, drop=True)
            )
            ind_hc = ind_hc.reset_index(drop=True)
            df = df.merge(
                ind_hc[["date", "industry", "ind_holder_trend_20d"]],
                on=["date", "industry"],
                how="left",
            )
        else:
            df["ind_holder_trend_20d"] = np.nan

        # 3. 北向资金行业时序
        if "north_net_buy" in df.columns:
            ind_nb = (
                df.groupby(["date", "industry"], observed=True)["north_net_buy"]
                .sum()
                .reset_index()
            )
            ind_nb = ind_nb.sort_values(["industry", "date"])
            ind_nb["ind_north_chg_5d"] = ind_nb.groupby("industry", observed=True)[
                "north_net_buy"
            ].diff(5)
            ind_nb = ind_nb.reset_index(drop=True)
            df = df.merge(
                ind_nb[["date", "industry", "ind_north_chg_5d"]],
                on=["date", "industry"],
                how="left",
            )
        else:
            df["ind_north_chg_5d"] = np.nan

        # 4. 龙虎榜行业时序
        lhb_col = (
            "lhb_institutional_net_buy"
            if "lhb_institutional_net_buy" in df.columns
            else ("lhb_net_buy" if "lhb_net_buy" in df.columns else None)
        )
        if lhb_col:
            ind_lhb = (
                df.groupby(["date", "industry"], observed=True)[lhb_col]
                .sum()
                .reset_index()
            )
            ind_lhb = ind_lhb.sort_values(["industry", "date"])
            ind_lhb["ind_lhb_net_flow_5d"] = (
                ind_lhb.groupby("industry", observed=True)[lhb_col]
                .rolling(5, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            ind_lhb = ind_lhb.reset_index(drop=True)
            df = df.merge(
                ind_lhb[["date", "industry", "ind_lhb_net_flow_5d"]],
                on=["date", "industry"],
                how="left",
            )
        else:
            df["ind_lhb_net_flow_5d"] = np.nan

        # 5. 行业资金流综合方向 (正=资金流入行业)
        comps = []
        if "ind_margin_chg_5d" in df.columns:
            comps.append(df["ind_margin_chg_5d"].fillna(0))
        if "ind_north_chg_5d" in df.columns:
            comps.append(
                df["ind_north_chg_5d"].fillna(0)
                / df["ind_north_chg_5d"]
                .abs()
                .replace(0, 1)
                .rolling(20)
                .mean()
                .fillna(1)
            )
        if comps:
            df["ind_capital_flow"] = sum(comps) / len(comps)
        else:
            df["ind_capital_flow"] = np.nan

        return df

    # ---------------- ㉘ 申万行业指数动量/轮动 (Alt-7b) ----------------
    @staticmethod
    def dim28_sector_index(df: pd.DataFrame) -> pd.DataFrame:
        """从申万一级行业指数日线提取行业动量/轮动特征.

        上游: 需 sector_index 面板 merge 到个股 (由 panel_builder.enrich_alt_data 扩展).
        行业指数列: sw_ret_1d, sw_ret_5d, sw_ret_20d, sw_vol_20d,
                    sw_relative_strength, sw_rotation_position

        产出 (行业级特征, 同行业个股共享):
          1. sw_ret_1d              — 行业指数日收益率
          2. sw_ret_5d              — 行业指数 5 日动量
          3. sw_ret_20d             — 行业指数 20 日动量
          4. sw_vol_20d             — 行业指数 20 日波动率
          5. sw_relative_strength   — 行业/沪深300 相对强弱 (20日)
          6. sw_rotation_position   — 行业轮动位置 (5日动量截面排名)
          7. sw_momentum_accel      — 行业动量加速度 (5日-20日)
          8. sw_turnover_anomaly    — 行业成交额异常 (vs 自身20日均值)
        """
        # 这些列需由 panel_builder 将行业指数 merge 到个股 (按 date+industry 对齐)
        # 如果面板中存在行业指数列 (index_code match → industry match), 直接使用
        # 否则尝试从 OHLCV 计算行业平均回报作为代理
        out_cols = [
            "sw_ret_1d",
            "sw_ret_5d",
            "sw_ret_20d",
            "sw_vol_20d",
            "sw_relative_strength",
            "sw_rotation_position",
            "sw_momentum_accel",
            "sw_turnover_anomaly",
        ]

        # 方案 A: 面板已有行业指数 merge 列 (上游已处理, sw_ret_1d 来自 sector_index)
        # 方案 B: 从 OHLCV 面板计算行业平均收益作为代理
        # 统一路径: sw_ret_1d 优先 (真实行业指数), 否则用 OHLCV 代理
        if "industry" not in df.columns:
            for c in out_cols:
                df[c] = np.nan
            return df

        if "sw_ret_1d" not in df.columns:
            # 方案 B: OHLCV 代理 — 行业内个股等权平均收益率
            df = df.sort_values(["symbol", "date"])
            close_col = "close_hfq" if "close_hfq" in df.columns else "close"
            df["_ret_1d"] = df.groupby("symbol")[close_col].pct_change()
            grp = df.groupby(["date", "industry"], observed=True)
            ind_ret = grp["_ret_1d"].mean().reset_index()
            ind_ret = ind_ret.rename(columns={"_ret_1d": "sw_ret_1d"})
            # 行业成交额代理
            ind_amount = grp["amount"].sum().reset_index()
            ind_amount = ind_amount.rename(columns={"amount": "_ind_amount"})
            ind = ind_ret.merge(ind_amount, on=["date", "industry"], how="left")
            ind = ind.sort_values(["industry", "date"])
            df = df.drop(columns=["_ret_1d"], errors="ignore")
        else:
            # 方案 A: 用已有的 sw_ret_1d 按行业聚合 (取每行业每日第一条, 行业指数同行业个股共享)
            sw_cols = ["date", "industry", "sw_ret_1d"]
            extra = [c for c in ["sw_index_close", "sw_index_vol"] if c in df.columns]
            ind = df[sw_cols + extra].drop_duplicates(subset=["date", "industry"])
            ind = ind.sort_values(["industry", "date"])
            # 成交额代理: 行业内个股 amount 之和
            if "amount" in df.columns:
                ind_amt = (
                    df.groupby(["date", "industry"], observed=True)["amount"]
                    .sum()
                    .reset_index()
                )
                ind_amt = ind_amt.rename(columns={"amount": "_ind_amount"})
                ind = ind.merge(ind_amt, on=["date", "industry"], how="left")

        # per-industry rolling (统一路径)
        def per_ind(g):
            g = g.sort_values("date")
            g["sw_ret_5d"] = g["sw_ret_1d"].rolling(5, min_periods=3).sum()
            g["sw_ret_20d"] = g["sw_ret_1d"].rolling(20, min_periods=5).sum()
            g["sw_vol_20d"] = g["sw_ret_1d"].rolling(20, min_periods=10).std()
            # 成交额异常
            if "_ind_amount" in g.columns:
                amt_ma = g["_ind_amount"].rolling(20, min_periods=10).mean()
                g["sw_turnover_anomaly"] = (
                    g["_ind_amount"] / amt_ma.replace(0, np.nan) - 1
                )
            else:
                g["sw_turnover_anomaly"] = np.nan
            # 动量加速度
            g["sw_momentum_accel"] = g["sw_ret_5d"] - g["sw_ret_20d"]
            return g

        ind = per_ind(ind)
        ind = ind.drop(columns=["_ind_amount"], errors="ignore")

        # 相对强弱 (vs 全行业均值)
        ind["sw_relative_strength"] = ind.groupby("date")["sw_ret_20d"].transform(
            lambda s: s - s.mean()
        )
        # 轮动位置 + 截面排名
        ind["sw_rotation_position"] = ind.groupby("date")["sw_ret_5d"].rank(pct=True)
        for col in ["sw_ret_5d", "sw_ret_20d"]:
            ind[f"{col}_rank"] = ind.groupby("date")[col].rank(pct=True)

        # merge 回个股面板
        merge_cols = ["date", "industry"] + out_cols
        avail = [c for c in merge_cols if c in ind.columns]
        df = df.merge(ind[avail], on=["date", "industry"], how="left")

        return df

    # ---------------- ㉙ 股东增减持 (Alt-6) ----------------
    @staticmethod
    def dim29_holdertrade(df: pd.DataFrame) -> pd.DataFrame:
        """从股东增减持数据提取 7 个内部人交易行为因子.

        上游列: sh_net_change_sign, sh_change_amt_total (由 panel_builder enrich_alt_data
                从 Tushare stk_holdertrade / AKShare 股东增减持 聚合而来)

        数据特点: 不定期公告, 非日频 — 用 ffill + rolling 处理稀疏性.

        产出:
          1. sh_net_sign_20d      — 近 20 日净增减持方向累计 (正=净增持)
          2. sh_net_sign_60d      — 近 60 日净增减持方向累计
          3. sh_change_amt_20d    — 近 20 日增减持金额合计 (取最近公告值前向填充后 rolling sum)
          4. sh_change_amt_60d    — 近 60 日增减持金额合计
          5. sh_insider_signal    — 内部人信号: 20 日内净增持=+1, 净减持=-1, 无=0
          6. sh_change_frequency  — 近 60 日公告次数 (活跃度指标)
          7. sh_amt_vs_amount     — 增减持金额/日成交额 (影响规模)
        """
        has_ht = any(
            c in df.columns for c in ["sh_net_change_sign", "sh_change_amt_total"]
        )
        out_cols = [
            "sh_net_sign_20d",
            "sh_net_sign_60d",
            "sh_change_amt_20d",
            "sh_change_amt_60d",
            "sh_insider_signal",
            "sh_change_frequency",
            "sh_amt_vs_amount",
        ]
        if not has_ht:
            for c in out_cols:
                df[c] = np.nan
            return df

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")

            # 净方向: ffill 填补公告间隙, 未公告期视为 0 (无变动)
            sign = g.get("sh_net_change_sign", pd.Series(0.0, index=g.index)).fillna(0)
            g["sh_net_sign_20d"] = sign.rolling(20, min_periods=1).sum()
            g["sh_net_sign_60d"] = sign.rolling(60, min_periods=1).sum()

            # 内部人信号: 20 日净方向符号
            g["sh_insider_signal"] = np.sign(g["sh_net_sign_20d"])

            # 金额: ffill 最近公告值, rolling sum 累计净变动
            amt = g.get("sh_change_amt_total", pd.Series(np.nan, index=g.index))
            amt_ffill = amt.replace(0, np.nan).ffill()
            g["sh_change_amt_20d"] = amt_ffill.rolling(20, min_periods=1).sum()
            g["sh_change_amt_60d"] = amt_ffill.rolling(60, min_periods=1).sum()

            # 活跃度: 近 60 日有公告的天数
            has_event = (amt.notna() & (amt != 0)).astype(int)
            g["sh_change_frequency"] = has_event.rolling(60, min_periods=1).sum()

            # 影响规模: 增减持金额 / 日成交额 (取对数避免极端值)
            if "amount" in g.columns:
                g["sh_amt_vs_amount"] = (
                    amt_ffill.abs() / g["amount"].replace(0, np.nan)
                ).replace([np.inf, -np.inf], np.nan)
            else:
                g["sh_amt_vs_amount"] = np.nan

            return g

        return _apply_per_stock(df, per_stock)

    # ---------------- ㉚ K线几何特征 (缺口/实体/影线/连续) ----------------
    @staticmethod
    def dim30_kline_geometry(df: pd.DataFrame) -> pd.DataFrame:
        """K 线几何结构 — 纯价格形态, 与 MACD/RSI/MA 正交.

        四组特征:
          1. 缺口: up_gap / down_gap 强度 + 回补
          2. 实体: 实体占比, 实体方向一致性, 大阳大阴密度
          3. 影线: 上下影线比率, 多空攻击力
          4. 连续: 连阳/连阴天数, 新高/新低累计
        """

        def per_stock(g: pd.DataFrame) -> pd.DataFrame:
            o, h, l, c = g["open"], g["high"], g["low"], g["close"]  # noqa: E741
            pc = c.shift(1)

            # ── 1. 缺口 ──
            gap_pct = (o - pc) / pc.replace(0, np.nan)
            g["up_gap_pct"] = gap_pct.clip(lower=0)
            g["down_gap_pct"] = (-gap_pct).clip(lower=0)
            # 缺口强度 (5/20日累计)
            g["gap_strength_5d"] = gap_pct.rolling(5, min_periods=1).sum()
            g["gap_strength_20d"] = gap_pct.rolling(20, min_periods=1).sum()
            # 缺口未回补: 当日最低 > 前日最高 (上跳缺口未补)
            g["unfilled_up_gap"] = ((l > pc) & (gap_pct > 0.01)).astype(float)
            g["unfilled_down_gap"] = ((h < pc) & (gap_pct < -0.01)).astype(float)
            g["gap_unfilled_5d"] = (
                g["unfilled_up_gap"].rolling(5, min_periods=1).sum()
                - g["unfilled_down_gap"].rolling(5, min_periods=1).sum()
            )

            # ── 2. 实体 ──
            total_range = (h - l).replace(0, np.nan)
            body = (c - o).abs()
            body_pct = body / total_range  # 实体占全日振幅比 (0~1)
            g["body_pct"] = body_pct
            g["body_pct_ma5"] = body_pct.rolling(5, min_periods=1).mean()
            g["body_pct_ma20"] = body_pct.rolling(20, min_periods=1).mean()
            # 大阳线密度 (实体>50%全日振幅 且 收阳)
            big_white = ((body_pct > 0.5) & (c > o)).astype(float)
            big_black = ((body_pct > 0.5) & (c < o)).astype(float)
            g["big_white_density_5d"] = big_white.rolling(5, min_periods=1).sum()
            g["big_black_density_5d"] = big_black.rolling(5, min_periods=1).sum()
            g["big_white_density_20d"] = big_white.rolling(20, min_periods=1).sum()
            g["big_black_density_20d"] = big_black.rolling(20, min_periods=1).sum()
            # 实体方向一致性 (连续同向实体=趋势确定, 频繁切换=震荡)
            body_dir = np.sign(c - o)
            g["body_dir_consistency_5d"] = body_dir.rolling(5, min_periods=1).apply(
                lambda x: (x == x.iloc[-1]).mean()
            )
            g["body_dir_consistency_20d"] = body_dir.rolling(20, min_periods=1).apply(
                lambda x: (x == x.iloc[-1]).mean()
            )

            # ── 3. 影线 ──
            upper_shadow = h - np.maximum(o, c)
            lower_shadow = np.minimum(o, c) - l
            g["upper_shadow_pct"] = upper_shadow / total_range
            g["lower_shadow_pct"] = lower_shadow / total_range
            g["upper_shadow_ma5"] = (
                g["upper_shadow_pct"].rolling(5, min_periods=1).mean()
            )
            g["lower_shadow_ma5"] = (
                g["lower_shadow_pct"].rolling(5, min_periods=1).mean()
            )
            # 影线比率 (上/下影平衡 → 多空博弈强度)
            g["shadow_ratio"] = (upper_shadow / lower_shadow.replace(0, np.nan)).clip(
                0, 10
            )
            g["shadow_ratio_ma5"] = g["shadow_ratio"].rolling(5, min_periods=1).mean()
            # 攻击力: 上影增长=空方反击, 下影增长=多方支撑
            g["upper_shadow_chg_5d"] = g["upper_shadow_pct"] - g[
                "upper_shadow_pct"
            ].shift(5)
            g["lower_shadow_chg_5d"] = g["lower_shadow_pct"] - g[
                "lower_shadow_pct"
            ].shift(5)

            # ── 4. 连续 ──
            is_white = (c > o).astype(int)
            # 连阳/连阴天数
            g["white_streak"] = is_white.groupby((is_white == 0).cumsum()).cumsum()
            g["black_streak"] = (
                (1 - is_white).groupby((is_white == 1).cumsum()).cumsum()
            )
            # 连创新高/新低天数
            new_high = (h > h.shift(1)).astype(int)
            new_low = (l < l.shift(1)).astype(int)
            g["new_high_streak"] = new_high.groupby((new_high == 0).cumsum()).cumsum()
            g["new_low_streak"] = new_low.groupby((new_low == 0).cumsum()).cumsum()
            # 收盘位置 (日内 0=最低收, 1=最高收)
            close_pos = (c - l) / total_range
            g["close_position"] = close_pos
            g["close_position_ma5"] = close_pos.rolling(5, min_periods=1).mean()
            g["close_position_ma20"] = close_pos.rolling(20, min_periods=1).mean()

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

    # ---------------- ㉛ 公告事件特征 ----------------
    @staticmethod
    def dim31_announcement(df: pd.DataFrame) -> pd.DataFrame:
        """从 announce_date 派生公告事件特征 (PIT 安全).

        产出:
          1. days_since_last_ann — 距上次公告天数 (正=越久未公告, NaT → NaN)
          2. ann_count_60d       — 近 60 日公告次数
          3. is_ann_day          — 当日是否有公告 (二元)

        上游列: announce_date (datetime64[ns])
        NaN 率高时 (当前 ~99%) 特征填 NaN, LightGBM 自行处理.
        """
        if "announce_date" not in df.columns:
            for c in ("days_since_last_ann", "ann_count_60d", "is_ann_day"):
                df[c] = np.nan
            return df

        def _per_stock(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values("date")
            ann_dt = pd.to_datetime(g["announce_date"])
            # PIT: 仅用 <=当前日期 的公告
            last_ann = ann_dt.where(ann_dt <= g["date"]).ffill()
            g["days_since_last_ann"] = (g["date"] - last_ann).dt.days
            # 公告频率: 近 60 日有公告的交易日数
            g["ann_count_60d"] = (
                ann_dt.notna().astype(int).rolling(60, min_periods=1).sum()
            )
            g["is_ann_day"] = ann_dt.notna().astype(int)
            return g

        return _apply_per_stock(df, _per_stock)

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
            # dim09 前瞻信号: 吸筹峰含 REF(X,-1) 前瞻, 严禁入特征 (安全网 #4)
            "吸筹峰",
            "time",
            "红在蓝上",
        }
        return [
            c
            for c in df.columns
            if c not in id_cols and not c.startswith(exclude_prefix)
        ]
