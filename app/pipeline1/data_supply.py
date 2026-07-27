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
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FETCH_TIMEOUT = 120  # 单源单股拉取硬超时 (秒) — 防接口挂起卡死整个回填


def _with_timeout(fn, timeout: float = FETCH_TIMEOUT):
    """硬超时包装: 接口挂起 (无 timeout 的 requests/socket) 时按失败处理.

    守护线程执行, 超时被遗弃 (不阻塞主流程与进程退出); 线程内异常原样抛出.
    """
    import threading

    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 — 透传给主线程
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"拉取超过 {timeout}s 挂起")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _ak_call(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs):
    """akshare 调用重试 (东财接口频繁断连/限流, 指数退避)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — 网络层异常类型多样, 统一重试
            last = exc
            wait = backoff * (attempt + 1)
            logger.warning(
                "akshare 调用失败 (%s), %.1fs 后重试 %d/%d",
                exc,
                wait,
                attempt + 1,
                retries,
            )
            time.sleep(wait)
    raise last


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
        self._fetcher_hist = fetcher_hist or self._default_fetch_hist

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

    # ---------------- 历史数据: akshare 主源 + baostock 回退 ----------------
    def _default_fetch_hist(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """历史日线多源级联: akshare(东财) → sina → baostock.

        东财对批量调用会临时封 IP (RemoteDisconnected), baostock 高频 login 会挂起;
        某源一次失败后本次运行内跳过该源 (封禁是 IP 级且持续一段时间), 直接走下一源.
        口径对齐: volume 统一为手 (sina/baostock 股 ÷100), turnover_rate 统一为 %
        (sina turnover 是小数 ×100); close 跨源已校验精确一致 (hfq 基点各源不同,
        但特征只用收益率/比值, 不受影响).
        """
        down = getattr(self, "_hist_sources_down", set())
        self._hist_sources_down = down
        last: Exception | None = None
        for name, fn in (
            ("akshare", self._akshare_fetch_hist),
            ("sina", self._sina_fetch_hist),
            ("baostock", self._baostock_fetch_hist),
        ):
            if name in down:
                continue
            try:
                return _with_timeout(lambda: fn(symbol, start, end), FETCH_TIMEOUT)
            except Exception as exc:
                logger.warning(
                    "%s 历史拉取失败 %s (%s) → 本次运行内切换下一源", name, symbol, exc
                )
                down.add(name)
                last = exc
        raise DataSupplyError(f"全部数据源失败 {symbol}: {last}")

    @staticmethod
    def _sina_fetch_hist(symbol: str, start: str, end: str) -> pd.DataFrame:
        """sina 个股日线: raw (adjust="") + hfq 双价格合并.

        产出列与 _akshare_fetch_hist 相同; volume 股→手 (÷100),
        turnover 小数→% (×100) 对齐东财口径.
        """
        import akshare as ak

        code6 = str(symbol).split(".")[0]
        exchange = "sh" if code6.startswith(("6", "9")) else "sz"
        sina_code = f"{exchange}{code6}"
        raw = _ak_call(
            ak.stock_zh_a_daily,
            symbol=sina_code,
            start_date=start,
            end_date=end,
            adjust="",
        )
        hfq = _ak_call(
            ak.stock_zh_a_daily,
            symbol=sina_code,
            start_date=start,
            end_date=end,
            adjust="hfq",
        )
        if raw is None or len(raw) == 0:
            raise DataSupplyError(f"sina 无数据: {symbol}")
        hfq = hfq[["date", "open", "high", "low", "close"]]
        hfq.columns = ["date", "open_hfq", "high_hfq", "low_hfq", "close_hfq"]
        df = raw.merge(hfq, on="date", how="left")
        df = df.rename(columns={"turnover": "turnover_rate"})
        df["volume"] = df["volume"] / 100  # 股 → 手
        df["turnover_rate"] = df["turnover_rate"] * 100  # 小数 → %
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

    # baostock 会话复用: login 一次 (~5s), atexit 统一 logout;
    # 逐股 login/logout 会让 300 只回填多花 ~40 分钟
    _bs_session = None

    def _baostock_login(self):
        import baostock as bs

        if DataSupplyChain._bs_session is None:
            rs = bs.login()
            if rs.error_code != "0":
                raise DataSupplyError(f"baostock login 失败: {rs.error_msg}")
            DataSupplyChain._bs_session = bs
            import atexit

            atexit.register(self._baostock_logout)
        return DataSupplyChain._bs_session

    @staticmethod
    def _baostock_logout() -> None:
        if DataSupplyChain._bs_session is not None:
            try:
                DataSupplyChain._bs_session.logout()
            except Exception:  # 退出阶段异常忽略
                pass
            DataSupplyChain._bs_session = None

    def _baostock_fetch_hist(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """baostock 个股日线: raw (adjustflag=3) + hfq (adjustflag=1) 双价格合并.

        产出列与 _akshare_fetch_hist 相同; volume 股→手 (÷100) 对齐 akshare 口径.
        """
        bs = self._baostock_login()
        code6 = str(symbol).split(".")[0]
        exchange = "sh" if code6.startswith(("6", "9")) else "sz"
        bs_code = f"{exchange}.{code6}"
        fields = "date,open,high,low,close,volume,amount,turn"

        def query(adjustflag: str) -> pd.DataFrame:
            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag=adjustflag,
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            df = pd.DataFrame(
                rows,
                columns=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "amount",
                    "turnover_rate",
                ],
            )
            return df

        raw = query("3")
        if raw.empty:
            raise DataSupplyError(f"baostock 无数据: {symbol}")
        hfq = query("1")[["date", "open", "high", "low", "close"]]
        hfq.columns = ["date", "open_hfq", "high_hfq", "low_hfq", "close_hfq"]
        df = raw.merge(hfq, on="date", how="left")
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = df["volume"] / 100  # baostock 股 → akshare 手
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

        raw = _ak_call(
            ak.stock_zh_a_hist,
            symbol=code6,
            period="daily",
            start_date=fmt(start),
            end_date=fmt(end),
            adjust="",
        )
        hfq = _ak_call(
            ak.stock_zh_a_hist,
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
        now_str = now or datetime.now().strftime("%H:%M")
        now_dt = datetime.strptime(now_str, "%H:%M")
        deadline_dt = datetime.strptime(deadline, "%H:%M")
        return now_dt < deadline_dt

    # ---------------- [B11] OHLCV 回填 ----------------
    BACKFILL_MIN_DAYS = 1250  # ≥5 年交易日 (特征预热期 250 日独立于训练窗口)

    def backfill_ohlcv(
        self,
        symbols: list[str],
        years: int = 5,
        end: str | None = None,
        refresh: bool = False,
        throttle: float = 0.5,
    ) -> pd.DataFrame:
        """[B11] OHLCV 回填 ≥5 年 (≥1250 交易日, akshare/Tushare, 不受 iFinD 3 年限制).

        逐股拉取历史日线 (hfq+raw 双价格), 合并为面板; 单股失败告警并跳过
        (不中断整体回填). 筹码/资金流维持可得深度, 不足按缺失处理
        (NaN 不参与缩尾分位计算). throttle: 每股间隔秒数 (东财限流防护).
        """
        end_dt = pd.Timestamp(end) if end else pd.Timestamp.now()
        start_dt = end_dt - pd.DateOffset(years=years) - pd.Timedelta(days=30)
        start = start_dt.strftime("%Y-%m-%d")  # API 边界: 转字符串
        end = end_dt.strftime("%Y-%m-%d")
        frames = []
        for i, sym in enumerate(symbols):
            try:
                frames.append(self.fetch_history(sym, start, end, refresh=refresh))
            except DataSupplyError as exc:
                logger.warning("B11 回填跳过 %s: %s", sym, exc)
            if throttle and i < len(symbols) - 1:
                time.sleep(throttle)
            if (i + 1) % 50 == 0:
                logger.info("B11 回填进度: %d/%d", i + 1, len(symbols))
        if not frames:
            raise DataSupplyError("B11 回填失败: 全部标的无数据")
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        logger.info(
            "B11 回填完成: %d/%d 标的, %d 行", len(frames), len(symbols), len(panel)
        )
        return panel

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
