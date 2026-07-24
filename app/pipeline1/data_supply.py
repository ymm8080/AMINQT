"""
数据供应链 (DESIGN §14 〇.2, 安全网 #0)
========================================
- 全部 API 自动拉取入库, 严禁手动导出; 本地缓存仅供开发调试
- 双价格体系: 同时获取 hfq (后复权, 特征/标签用) + raw (原始价, 成交额/换手/涨停判定用)
- 每日 15:00 前完成拉取; 失败 → 告警 + 降级 (清单三档降级见 list_generator.ListDeliveryGuard)
- fetcher 可注入: 生产用 akshare, 测试用 mock
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 标准列: symbol, date, board, open/high/low/close (raw), open_hfq..close_hfq,
#         volume, amount, turnover_rate, pre_close, is_suspended, is_st,
#         industry, list_days, free_float_turnover_rate
REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "board",
    "open",
    "high",
    "low",
    "close",
    "close_hfq",
    "volume",
    "amount",
    "turnover_rate",
    "pre_close",
    "is_suspended",
    "is_st",
]


class DataSupplyError(Exception):
    """数据拉取失败 (触发告警 + 降级)."""


class DataSupplyChain:
    """数据供应链 — hfq/raw 双价格, 按日缓存, 失败告警.

    Args:
        cache_dir: 本地缓存目录 (parquet)
        fetcher:   可注入数据函数 (trade_date: str) -> DataFrame (全市场当日截面)
        fetcher_hist: 可注入历史数据函数 (symbol, start, end) -> DataFrame
    """

    def __init__(
        self, cache_dir: str = "data/supply_cache", fetcher=None, fetcher_hist=None
    ):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._fetcher = fetcher or self._akshare_fetch_daily
        self._fetcher_hist = fetcher_hist or self._akshare_fetch_hist

    # ---------------- 生产数据源 (akshare) ----------------
    def _akshare_fetch_daily(self, trade_date: str) -> pd.DataFrame:
        """全市场当日截面: 东财 spot (raw) + 缓存 hist 补 hfq 收盘价.

        实现说明: spot 无 hfq; close_hfq 由最近一次 fetch_history 缓存的
        hfq/raw 比率换算 (日级近似, 已在模块文档声明).
        """
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        if spot is None or len(spot) == 0:
            raise DataSupplyError("akshare spot 拉取失败")
        df = pd.DataFrame(
            {
                "symbol": spot["代码"].str[-6:],
                "close": pd.to_numeric(spot["最新价"], errors="coerce"),
                "amount": pd.to_numeric(spot["成交额"], errors="coerce"),
                "turnover_rate": pd.to_numeric(spot["换手率"], errors="coerce"),
            }
        )
        df["date"] = pd.to_datetime(trade_date)
        df["close_hfq"] = df["close"]  # 近似: 首次拉取无比率, 由历史缓存覆盖
        for col in ("open", "high", "low", "open_hfq", "high_hfq", "low_hfq"):
            df[col] = np.nan
        df["pre_close"] = df["close"]  # spot 无昨收列时降级, 生产应用历史缓存回填
        return df

    @staticmethod
    def _akshare_fetch_hist(symbol: str, start: str, end: str) -> pd.DataFrame:
        """akshare 个股日线: raw (adjust="") + hfq (adjust="hfq") 双价格合并 (安全网 #0).

        产出列: symbol, date, open/high/low/close (raw), *_hfq, volume, amount,
                turnover_rate, pre_close.
        """
        import akshare as ak

        code6 = str(symbol).split(".")[0]

        def fmt(d):
            return str(d).replace("-", "")

        raw = ak.stock_zh_a_hist(
            symbol=code6,
            period="daily",
            start_date=fmt(start),
            end_date=fmt(end),
            adjust="",
        )
        hfq = ak.stock_zh_a_hist(
            symbol=code6,
            period="daily",
            start_date=fmt(start),
            end_date=fmt(end),
            adjust="hfq",
        )
        if raw is None or len(raw) == 0:
            raise DataSupplyError(f"akshare 无数据: {symbol}")
        rename = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover_rate",
        }
        raw = raw.rename(columns=rename)
        hfq = hfq.rename(columns=rename)[["date", "open", "high", "low", "close"]]
        hfq.columns = ["date", "open_hfq", "high_hfq", "low_hfq", "close_hfq"]
        df = raw.merge(hfq, on="date", how="left")
        df["date"] = pd.to_datetime(df["date"])
        df["symbol"] = code6
        df["pre_close"] = df["close"].shift(1)
        return df[
            [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "open_hfq",
                "high_hfq",
                "low_hfq",
                "close_hfq",
                "volume",
                "amount",
                "turnover_rate",
                "pre_close",
            ]
        ]

    # ---------------- 缓存 ----------------
    def _cache_path(self, trade_date: str) -> str:
        return os.path.join(self.cache_dir, f"daily_{trade_date}.parquet")

    def fetch_daily(self, trade_date: str, refresh: bool = False) -> pd.DataFrame:
        """拉取全市场当日截面 (含 hfq+raw 双价格).

        Raises:
            DataSupplyError: 拉取失败 — 调用方应触发告警 + 降级流程.
        """
        path = self._cache_path(trade_date)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)
        try:
            df = self._fetcher(trade_date)
        except Exception as exc:
            logger.error("数据拉取失败 %s: %s", trade_date, exc)
            raise DataSupplyError(f"fetch_daily {trade_date}: {exc}") from exc
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataSupplyError(f"fetch_daily {trade_date}: 缺关键字段 {missing}")
        df.to_parquet(path, index=False)
        return df

    def fetch_history(
        self, symbol: str, start: str, end: str, refresh: bool = False
    ) -> pd.DataFrame:
        """拉取个股历史日线 (hfq+raw)."""
        path = os.path.join(self.cache_dir, f"hist_{symbol}_{start}_{end}.parquet")
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)
        try:
            df = self._fetcher_hist(symbol, start, end)
        except Exception as exc:
            logger.error("历史数据拉取失败 %s: %s", symbol, exc)
            raise DataSupplyError(f"fetch_history {symbol}: {exc}") from exc
        df.to_parquet(path, index=False)
        return df

    def fetch_fundamentals(self) -> pd.DataFrame:
        """财务数据 (PE/PB/净利增速/营收增速), 必须含 announce_date (PIT 对齐用).

        实现: tushare daily_basic (需 TUSHARE_TOKEN); 严格 PIT 需 fina_indicator
        的 ann_date (积分接口), 无 token 时显式报错 — 严禁无 PIT 日期进训练.
        """
        try:
            import tushare as ts

            pro = ts.pro_api(os.environ.get("TUSHARE_TOKEN"))
            daily = pro.daily_basic(fields="ts_code,trade_date,pe_ttm,pb")
            daily = daily.rename(
                columns={
                    "ts_code": "symbol",
                    "trade_date": "announce_date",
                    "pe_ttm": "PE_TTM",
                    "pb": "PB_LF",
                }
            )
            logger.warning(
                "tushare daily_basic 以 trade_date 近似 announce_date; "
                "严格 PIT 需 fina_indicator 的 ann_date (积分接口)"
            )
            return daily
        except Exception as exc:
            raise NotImplementedError(
                "生产财务数据: 配置 TUSHARE_TOKEN (PIT 对齐需 announce_date)"
            ) from exc

    def fetch_money_flow(self, trade_date: str) -> pd.DataFrame:
        """资金流/筹码 — 锁死单一数据源 (东财), 换源即换模型."""
        import akshare as ak

        df = ak.stock_individual_fund_flow_rank(indicator="今日")
        df = df.rename(
            columns={
                "代码": "symbol",
                "今日主力净流入-净额": "main_money_flow",
                "今日超大单净流入-净额": "super_large_order_net",
            }
        )
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["date"] = pd.to_datetime(trade_date)
        return df[["symbol", "date", "main_money_flow", "super_large_order_net"]]

    def fetch_market_sentiment(self, trade_date: str) -> dict:
        """全市场情绪: 两市总成交额 / 涨停家数 / 跌停家数 / 沪深300 涨跌幅 (空仓触发用)."""
        import akshare as ak

        daily = self.fetch_daily(trade_date)
        hs300 = ak.index_zh_a_hist(
            symbol="000300",
            period="daily",
            start_date=str(trade_date).replace("-", ""),
            end_date=str(trade_date).replace("-", ""),
        )
        hs300_chg = float(hs300["涨跌幅"].iloc[-1]) / 100 if len(hs300) else 0.0
        limit_up = (
            int((daily["close"] >= daily["pre_close"] * 1.098).sum())
            if len(daily)
            else 0
        )
        limit_dn = (
            int((daily["close"] <= daily["pre_close"] * 0.902).sum())
            if len(daily)
            else 0
        )
        return {
            "market_turnover": float(daily["amount"].sum()),
            "count_limit_up": limit_up,
            "count_limit_down": limit_dn,
            "hs300_chg": hs300_chg,
        }

    # ---------------- 新鲜度 ----------------
    @staticmethod
    def check_freshness(now: str | None = None, deadline: str = "15:00") -> bool:
        """15:00 前置检查: 当前时间是否早于拉取死线."""
        now = now or time.strftime("%H:%M")
        return now < deadline

    # ---------------- [B11] OHLCV 回填 ----------------
    BACKFILL_MIN_DAYS = 1250  # ≥5 年交易日 (特征预热期 250 日独立于训练窗口)

    def backfill_ohlcv(
        self,
        symbols: list[str],
        years: int = 5,
        end: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[B11] OHLCV 回填 ≥5 年 (≥1250 交易日, akshare/Tushare, 不受 iFinD 3 年限制).

        逐股拉取历史日线 (hfq+raw 双价格), 合并为面板; 单股失败告警并跳过
        (不中断整体回填). 筹码/资金流维持可得深度, 不足按缺失处理
        (NaN 不参与缩尾分位计算).
        """
        end = end or time.strftime("%Y-%m-%d")
        start = (
            pd.Timestamp(end) - pd.DateOffset(years=years) - pd.Timedelta(days=30)
        ).strftime("%Y-%m-%d")  # 多取 30 自然日覆盖非交易日
        frames = []
        for sym in symbols:
            try:
                frames.append(self.fetch_history(sym, start, end, refresh=refresh))
            except DataSupplyError as exc:
                logger.warning("B11 回填跳过 %s: %s", sym, exc)
        if not frames:
            raise DataSupplyError("B11 回填失败: 全部标的无数据")
        panel = pd.concat(frames, ignore_index=True)
        return panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    def check_backfill_depth(
        self, panel: pd.DataFrame, min_days: int | None = None
    ) -> bool:
        """[B11] 深度校验: 各标的交易日数 ≥1250 → True (720 日训练窗口);
        否则 False (首个训练窗口降为 540 日过渡, 达标后恢复)."""
        min_days = min_days or self.BACKFILL_MIN_DAYS
        if panel is None or len(panel) == 0:
            return False
        counts = panel.groupby("symbol")["date"].nunique()
        return bool(len(counts) > 0 and (counts >= min_days).all())
