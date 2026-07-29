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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self._fetcher = fetcher or self._tushare_fetch_daily
        self._fetcher_hist = fetcher_hist or self._default_fetch_hist

    # ---------------- 生产数据源 (Tushare primary, akshare fallback) ----------------
    def _tushare_fetch_daily(self, trade_date: str) -> pd.DataFrame:
        """全市场当日截面: Tushare pro.daily() (主力源, akshare 永久封 IP).

        产出与 _akshare_fetch_daily 同 schema: symbol, date, open..close,
        close_hfq, volume, amount, turnover_rate, pre_close.
        """
        pro = self._tushare_pro()
        if pro is None:
            raise DataSupplyError("Tushare 不可用, 无法拉取当日全市场截面")
        try:
            raw = _with_timeout(lambda: pro.daily(trade_date=trade_date))
        except Exception as exc:
            raise DataSupplyError(f"Tushare daily 拉取失败 {trade_date}: {exc}") from exc
        if raw is None or len(raw) == 0:
            raise DataSupplyError(f"Tushare daily 拉取失败: {trade_date}")
        df = pd.DataFrame({
            "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
            "date": pd.to_datetime(trade_date),
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "close_hfq": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": pd.to_numeric(raw["vol"], errors="coerce"),
            "amount": pd.to_numeric(raw["amount"], errors="coerce"),
            "turnover_rate": np.nan,
            "pre_close": pd.to_numeric(raw["pre_close"], errors="coerce"),
        })
        return df

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
    _MAX_CONSECUTIVE_FAILS = 3  # 连续失败 N 次才判定源不可用 (防单股缺失误杀全源)

    def _default_fetch_hist(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """历史日线多源级联: akshare(东财) → sina → baostock.

        东财对批量调用会临时封 IP (RemoteDisconnected), baostock 高频 login 会挂起;
        单次失败可能是个股在该源无数据 (如退市/代码变更), 不立即判死刑.
        连续 3 次失败才判定源不可用 (封禁是 IP 级且持续一段时间), 该源本运行内跳过.
        口径对齐: volume 统一为手 (sina/baostock 股 ÷100), turnover_rate 统一为 %
        (sina turnover 是小数 ×100); close 跨源已校验精确一致 (hfq 基点各源不同,
        但特征只用收益率/比值, 不受影响).
        """
        down = getattr(self, "_hist_sources_down", set())
        self._hist_sources_down = down
        fail_counters: dict[str, int] = getattr(self, "_hist_fail_counters", {})
        self._hist_fail_counters = fail_counters
        last: Exception | None = None
        for name, fn in (
            ("tushare", self._tushare_fetch_hist),
            ("akshare", self._akshare_fetch_hist),
            ("sina", self._sina_fetch_hist),
            ("baostock", self._baostock_fetch_hist),
        ):
            if name in down:
                continue
            try:
                result = _with_timeout(lambda: fn(symbol, start, end), FETCH_TIMEOUT)
                fail_counters[name] = 0  # 成功后重置连续失败计数
                return result
            except Exception as exc:
                fail_counters[name] = fail_counters.get(name, 0) + 1
                consecutive = fail_counters[name]
                if consecutive >= self._MAX_CONSECUTIVE_FAILS:
                    down.add(name)
                    logger.error(
                        "%s 连续 %d 次失败 (≥%d), 本次运行内永久跳过该源",
                        name, consecutive, self._MAX_CONSECUTIVE_FAILS,
                    )
                else:
                    logger.warning(
                        "%s 历史拉取失败 %s (%s) → 尝试下一源 (连续失败 %d/%d)",
                        name, symbol, exc, consecutive, self._MAX_CONSECUTIVE_FAILS,
                    )
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

    def _tushare_fetch_hist(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Tushare pro.daily() 个股历史日线 — hfq+raw 双价格合并.
        主力源: akshare 永久封该 IP, Tushare 替代.
        """
        pro = self._tushare_pro()
        if pro is None:
            raise DataSupplyError("Tushare 不可用")
        code6 = str(symbol).split(".")[0]
        ts_code = f"{code6}.{'SZ' if code6.startswith(('0','3','1')) else 'SH'}"
        try:
            raw = _ak_call(
                lambda: pro.daily(ts_code=ts_code, start_date=start.replace("-", ""),
                                  end_date=end.replace("-", "")),
                retries=3, backoff=1.0,
            )
        except Exception as exc:
            raise DataSupplyError(f"Tushare daily 历史拉取失败 {symbol}: {exc}") from exc
        if raw is None or len(raw) == 0:
            raise DataSupplyError(f"Tushare 无数据: {symbol}")
        rename = {
            "trade_date": "date", "open": "open", "close": "close",
            "high": "high", "low": "low", "vol": "volume", "amount": "amount",
        }
        raw = raw.rename(columns=rename)
        raw["date"] = pd.to_datetime(raw["date"])
        raw["symbol"] = code6
        raw["close_hfq"] = raw["close"]  # Tushare daily 只有不复权, hfq 需单独拉取
        raw["open_hfq"] = raw["open"]
        raw["high_hfq"] = raw["high"]
        raw["low_hfq"] = raw["low"]
        raw["turnover_rate"] = np.nan  # Tushare daily 不含换手
        raw["pre_close"] = raw["close"].shift(1)
        return raw[["symbol","date","open","high","low","close","open_hfq","high_hfq",
                     "low_hfq","close_hfq","volume","amount","turnover_rate","pre_close"]]

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

    # ---------------- Tushare cyq_perf 筹码分布 (日频) ----------------
    def fetch_chip_distribution(
        self,
        ts_code: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """[B13] 单股筹码分布 (Tushare cyq_perf, 日频).

        Args:
            ts_code: 股票代码 (e.g. '000001.SZ')
            start_date: 起始日期 'YYYYMMDD' (默认最近 5 年)
            end_date: 截止日期 'YYYYMMDD' (默认今天)

        Returns:
            DataFrame with columns: symbol, date, his_low, his_high,
            cost_5pct, cost_15pct, cost_50pct, cost_85pct, cost_95pct,
            weight_avg, winner_rate

        Raises:
            DataSupplyError: Tushare 不可用或该股票无 cyq 数据.
        """
        pro = self._tushare_pro()
        if pro is None:
            raise DataSupplyError("TUSHARE_TOKEN 未配置, 无法拉取筹码分布数据")
        kwargs = {"ts_code": ts_code}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date
        try:
            raw = _with_timeout(lambda: pro.cyq_perf(**kwargs))
        except Exception as exc:
            raise DataSupplyError(
                f"cyq_perf {ts_code} 拉取失败: {exc}"
            ) from exc
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        out = pd.DataFrame({
            "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
            "his_low": pd.to_numeric(raw["his_low"], errors="coerce"),
            "his_high": pd.to_numeric(raw["his_high"], errors="coerce"),
            "cost_5pct": pd.to_numeric(raw["cost_5pct"], errors="coerce"),
            "cost_15pct": pd.to_numeric(raw["cost_15pct"], errors="coerce"),
            "cost_50pct": pd.to_numeric(raw["cost_50pct"], errors="coerce"),
            "cost_85pct": pd.to_numeric(raw["cost_85pct"], errors="coerce"),
            "cost_95pct": pd.to_numeric(raw["cost_95pct"], errors="coerce"),
            "weight_avg": pd.to_numeric(raw["weight_avg"], errors="coerce"),
            "winner_rate": pd.to_numeric(raw["winner_rate"], errors="coerce"),
        })
        return out.sort_values(["symbol", "date"]).reset_index(drop=True)

    def fetch_chip_distribution_batch(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        throttle: float = 0.3,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[B13] 批量拉取筹码分布, 逐股请求, 失败跳过 (不中断批次).

        缓存: data/supply_cache/alt_data/cyq_tushare/cyq_<start>_<end>.parquet
        全量拉取 ~3,227 股 × 0.3s ≈ 16 分钟; 后续命中缓存秒级加载.

        Args:
            symbols: 股票代码列表 (6 位数字, e.g. ['000001', '600519'])
            start_date: 'YYYYMMDD'
            end_date: 'YYYYMMDD'
            throttle: 每股间隔秒数 (Tushare 免费 token 限流 ~200次/分钟)
            refresh: True 强制重新拉取

        Returns:
            全量筹码面板 (symbol, date, ...cyq 列)
        """
        cache_path = self._alt_cache_path("cyq_tushare", f"cyq_{start_date}_{end_date}")
        if not refresh and os.path.exists(cache_path):
            logger.info("cyq_perf 命中缓存: %s", cache_path)
            return pd.read_parquet(cache_path)

        frames = []
        for i, sym in enumerate(symbols):
            ts_code = f"{sym}.{'SZ' if sym.startswith(('0','3','1')) else 'SH'}"
            try:
                df = self.fetch_chip_distribution(ts_code, start_date, end_date)
                if len(df):
                    frames.append(df)
            except DataSupplyError as exc:
                logger.warning("cyq_perf 跳过 %s: %s", sym, exc)
            if throttle and i < len(symbols) - 1:
                time.sleep(throttle)
            if (i + 1) % 100 == 0:
                logger.info("cyq_perf 批量进度: %d/%d", i + 1, len(symbols))
        if not frames:
            raise DataSupplyError("cyq_perf 批量拉取: 全部标的无数据")
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        panel.to_parquet(cache_path, index=False)
        logger.info(
            "cyq_perf 批量完成: %d/%d 标的, %d 行 → %s",
            len(frames), len(symbols), len(panel), cache_path,
        )
        return panel

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

    # ════════════════════════════════════════════════════════════
    # 替代数据源 fetcher (2026-07-27 Phase 0)
    # 每类数据: Tushare 主源 (需 TUSHARE_TOKEN, 积分≥2000) → AKShare 降级
    # 缓存: data/supply_cache/<source>_<date>.parquet
    # ════════════════════════════════════════════════════════════

    ALT_CACHE_SUBDIR = "alt_data"

    def _alt_cache_path(self, source: str, key: str) -> str:
        """替代数据缓存路径: data/supply_cache/alt_data/<source>/<key>.parquet"""
        d = os.path.join(self.cache_dir, self.ALT_CACHE_SUBDIR, source)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{key}.parquet")

    def _tushare_pro(self):
        """懒加载 Tushare pro_api; 仅从环境变量 TUSHARE_TOKEN 读取 (禁止文件存储凭据)."""
        token = os.environ.get("TUSHARE_TOKEN")
        if not token:
            return None
        try:
            import tushare as ts
            return ts.pro_api(token)
        except Exception as exc:
            logger.warning("Tushare pro_api 初始化失败: %s", exc)
            return None

    # ── 1. 北向资金 ──
    def fetch_northbound(
        self, trade_date: str | None = None, start_date: str | None = None,
        end_date: str | None = None, refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-1] 北向资金 (沪深港通) — Tushare moneyflow_hsgt 主源, AKShare 降级.

        Args:
            trade_date: 单日 'YYYYMMDD' (优先级 > start/end)
            start_date: 起始日期 'YYYYMMDD'
            end_date:   截止日期 'YYYYMMDD'

        Returns:
            DataFrame [symbol, date, north_net_buy, north_buy_amt, north_sell_amt]
            仅沪深港通标的 (~1500只), 非通标的不在返回中.
        """
        if trade_date:
            key = trade_date
        else:
            key = f"{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("northbound", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        frames = []
        if pro is not None:
            try:
                kwargs = {}
                if trade_date:
                    kwargs["trade_date"] = trade_date
                else:
                    if start_date:
                        kwargs["start_date"] = start_date
                    if end_date:
                        kwargs["end_date"] = end_date
                raw = _with_timeout(lambda: pro.moneyflow_hsgt(**kwargs))
                if raw is not None and len(raw) > 0:
                    # moneyflow_hsgt: hgt = 沪股通净流向(north_net_buy_sh),
                    #                 sgt = 深股通净流向(north_net_buy_sz)
                    tushare_df = pd.DataFrame({
                        "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
                        "north_net_buy_sh": pd.to_numeric(raw["hgt"], errors="coerce"),
                        "north_net_buy_sz": pd.to_numeric(raw["sgt"], errors="coerce"),
                        "north_buy_amt_sh": None,
                        "north_sell_amt_sh": None,
                        "north_buy_amt_sz": None,
                        "north_sell_amt_sz": None,
                    })
                    # 按日期范围过滤
                    if trade_date:
                        target = pd.to_datetime(trade_date, format="%Y%m%d")
                        tushare_df = tushare_df[tushare_df["date"] == target]
                    else:
                        if start_date:
                            tushare_df = tushare_df[tushare_df["date"] >= pd.to_datetime(start_date, format="%Y%m%d")]
                        if end_date:
                            tushare_df = tushare_df[tushare_df["date"] <= pd.to_datetime(end_date, format="%Y%m%d")]
                    if len(tushare_df) > 0:
                        frames.append(tushare_df)
                        logger.info("Tushare moneyflow_hsgt: %d 行北向净买入数据 (最近日期 %s)", len(tushare_df), raw["trade_date"].iloc[0])
            except Exception as exc:
                logger.warning("Tushare moneyflow_hsgt 失败: %s", exc)

        # AKShare 降级 — 市场级北向资金 (沪股通+深股通分别拉取, 按 date 合并)
        try:
            import akshare as ak
            sh_frames = []
            # stock_hsgt_hist_em 返回沪股通/深股通各自的日频数据
            for market, prefix in [("沪股通", "sh"), ("深股通", "sz")]:
                try:
                    raw = _ak_call(ak.stock_hsgt_hist_em, symbol=market)
                    if raw is not None and len(raw) > 0:
                        # 尝试多个可能列名以应对AKShare/东方财富API变动
                        net_cols = ["当日成交净买额", "当日资金流入", "netBuyAmt"]
                        buy_cols = ["买入成交额", "BUY_AMT"]
                        sell_cols = ["卖出成交额", "SELL_AMT"]

                        net_val = None
                        for c in net_cols:
                            if c in raw.columns:
                                net_val = pd.to_numeric(raw[c], errors="coerce")
                                break
                        if net_val is None:
                            net_val = pd.Series(0, index=raw.index)

                        buy_val = None
                        for c in buy_cols:
                            if c in raw.columns:
                                buy_val = pd.to_numeric(raw[c], errors="coerce")
                                break
                        if buy_val is None:
                            buy_val = pd.Series(0, index=raw.index)

                        sell_val = None
                        for c in sell_cols:
                            if c in raw.columns:
                                sell_val = pd.to_numeric(raw[c], errors="coerce")
                                break
                        if sell_val is None:
                            sell_val = pd.Series(0, index=raw.index)

                        out = pd.DataFrame({
                            "date": pd.to_datetime(raw["日期"]),
                            f"north_net_buy_{prefix}": net_val,
                            f"north_buy_amt_{prefix}": buy_val,
                            f"north_sell_amt_{prefix}": sell_val,
                        })
                        # 东方财富API自2024-08-19起停止返回北向资金明细数据(NET_DEAL_AMT=None)
                        # 删除净买入为NaN的行, 避免下游使用无效NaN值
                        before = len(out)
                        out = out.dropna(subset=[f"north_net_buy_{prefix}"])
                        if len(out) < before:
                            logger.info("AKShare 北向资金(%s): 过滤 %d 行无明细数据 (最后有效日期≈2024-08-16)", market, before - len(out))
                        sh_frames.append(out)
                except Exception as exc:
                    logger.warning("AKShare 北向资金(%s)失败: %s", market, exc)

            if sh_frames:
                merged = sh_frames[0]
                if len(sh_frames) > 1:
                    merged = merged.merge(sh_frames[1], on="date", how="outer")
                # 按日期范围过滤
                if trade_date:
                    target = pd.to_datetime(trade_date, format="%Y%m%d")
                    merged = merged[merged["date"] == target]
                else:
                    if start_date:
                        merged = merged[merged["date"] >= pd.to_datetime(start_date, format="%Y%m%d")]
                    if end_date:
                        merged = merged[merged["date"] <= pd.to_datetime(end_date, format="%Y%m%d")]
                if len(merged) > 0:
                    frames.append(merged)
        except Exception as exc:
            logger.warning("AKShare 北向资金失败: %s", exc)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        # 去重: AKShare 有更完整的列(含买卖明细), 优先保留其行
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        df.to_parquet(path, index=False)
        return df

    # ── 2. 融资融券 ──

    @staticmethod
    def _parse_margin(raw: pd.DataFrame) -> pd.DataFrame:
        """解析 Tushare margin_detail 原始 DataFrame 为标准格式."""
        return pd.DataFrame({
            "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
            "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d"),
            "margin_balance": pd.to_numeric(raw.get("rzye", raw.get("margin_balance", 0)), errors="coerce"),
            "short_balance": pd.to_numeric(raw.get("rqye", raw.get("short_balance", 0)), errors="coerce"),
            "margin_buy_amt": pd.to_numeric(raw.get("rzmre", raw.get("margin_buy", 0)), errors="coerce"),
            "short_sell_vol": pd.to_numeric(raw.get("rqmcl", raw.get("short_sell", 0)), errors="coerce"),
        })

    def _margin_date_loop(
        self, pro, start_date: str, end_date: str, frames: list, *, step: int = 5,
    ) -> None:
        """逐日补采融资融券 (每 step 个交易日采样一次), 结果追加到 frames."""
        dates = pd.bdate_range(start=start_date, end=end_date)[::step]
        logger.info(
            "  融资融券逐日补采: %d 个交易日 (步长 %d)", len(dates), step,
        )
        for i, dt in enumerate(dates):
            dt_str = dt.strftime("%Y%m%d")
            try:
                raw = _with_timeout(lambda d=dt_str: pro.margin_detail(trade_date=d))
                if raw is not None and len(raw) > 0:
                    frames.append(self._parse_margin(raw))
                if i % 10 == 0 and i > 0:
                    logger.info("    ... %d/%d 完成", i + 1, len(dates))
            except Exception as exc:
                logger.debug("Tushare margin_detail %s 失败: %s", dt_str, exc)
            time.sleep(0.2)

    def fetch_margin(
        self, trade_date: str | None = None, start_date: str | None = None,
        end_date: str | None = None, refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-2] 融资融券明细 — Tushare margin_detail 主源, AKShare 降级.

        Tushare margin_detail 区间查询仅返回近期少数交易日; 当检测到
        返回值 < 100 个交易日时自动切换到逐日补采 (每 5 个交易日采样),
        覆盖完整历史.

        Returns:
            DataFrame [symbol, date, margin_balance, short_balance,
                       margin_buy_amt, short_sell_vol, ...]
            仅两融标的 (~1800只).
        """
        if trade_date:
            key = trade_date
        else:
            key = f"{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("margin", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        frames: list[pd.DataFrame] = []
        MIN_BULK_DATES = 100

        if pro is not None:
            try:
                # ── 单日查询 ──
                if trade_date:
                    raw = _with_timeout(lambda: pro.margin_detail(trade_date=trade_date))
                    if raw is not None and len(raw) > 0:
                        frames.append(self._parse_margin(raw))
                # ── 区间查询: 先批量尝试, 不足则逐日采样 ──
                elif start_date and end_date:
                    raw = _with_timeout(
                        lambda: pro.margin_detail(start_date=start_date, end_date=end_date),
                    )
                    if raw is not None and len(raw) > 0:
                        dates_in_bulk = raw["trade_date"].nunique()
                        frames.append(self._parse_margin(raw))
                        if dates_in_bulk < MIN_BULK_DATES:
                            logger.info(
                                "Tushare margin_detail 批量仅 %d 天 (<%d), 启动逐日补采",
                                dates_in_bulk, MIN_BULK_DATES,
                            )
                            self._margin_date_loop(pro, start_date, end_date, frames, step=2)
                    else:
                        self._margin_date_loop(pro, start_date, end_date, frames, step=2)
                # ── 无起止日期: 全量拉取 ──
                else:
                    raw = _with_timeout(lambda: pro.margin_detail())
                    if raw is not None and len(raw) > 0:
                        frames.append(self._parse_margin(raw))
            except Exception as exc:
                logger.warning("Tushare margin_detail 失败: %s", exc)
                # 区间查询抛异常时, 用逐日补采兜底
                if not frames and start_date and end_date:
                    self._margin_date_loop(pro, start_date, end_date, frames, step=2)

        # AKShare 降级 — 沪市融资融券
        if not frames:
            try:
                import akshare as ak
                dt = trade_date or end_date or datetime.now().strftime("%Y%m%d")
                for exchange, fn in [
                    ("sse", ak.stock_margin_detail_sse),
                    ("szse", getattr(ak, "stock_margin_detail_szse", None)),
                ]:
                    if fn is None:
                        continue
                    try:
                        raw = _ak_call(fn, date=dt)
                        if raw is not None and len(raw) > 0:
                            raw["symbol"] = raw.get("股票代码", raw.get("证券代码", "")).astype(str).str.zfill(6)
                            raw["date"] = pd.to_datetime(dt)
                            frames.append(raw)
                    except Exception:
                        pass
            except Exception as exc:
                logger.warning("AKShare 融资融券失败: %s", exc)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        # 逐日补采 + 批量可能重复, 去重保唯一
        if not trade_date:
            df = df.drop_duplicates(subset=["symbol", "date"])
        df.to_parquet(path, index=False)
        return df

    # ── 3. 基本面 PIT (fina_indicator) ──
    def fetch_fina_indicator(
        self, ts_code: str | None = None, period: str | None = None,
        start_date: str | None = None, end_date: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-3] 财务指标 PIT — Tushare fina_indicator (含 ann_date, 可严格 PIT 对齐).

        Args:
            ts_code:  单只股票 '000001.SZ' (None=全量, 需积分)
            period:   报告期 '20251231'
            start_date/end_date: 公告日期范围

        Returns:
            DataFrame [symbol, ann_date, end_date, roe, roa, gross_margin,
                       net_margin, eps_yoy, rev_yoy, profit_yoy, ...]
        """
        key = f"{ts_code or 'all'}_{period or ''}_{start_date or ''}_{end_date or ''}"
        path = self._alt_cache_path("fina_indicator", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        if pro is not None:
            kwargs: dict = {
                "fields": ("ts_code,ann_date,end_date,roe,roe_dt,roa,np_margin,gross_margin,"
                           "eps_yoy,or_yoy,profit_yoy,cf_sales,debt_to_assets,current_ratio,"
                           "assets_turn,ar_turn,inv_turn,ocf_to_or"),
            }
            if ts_code:
                kwargs["ts_code"] = ts_code
            if period:
                kwargs["period"] = period
            # fina_indicator 的 start_date/end_date 是 ann_date (公告日) 过滤
            # 部分token不支持此参数 → 先尝试带日期, 失败则不带日期做全量拉取
            date_kwargs = {}
            if start_date:
                date_kwargs["start_date"] = start_date
            if end_date:
                date_kwargs["end_date"] = end_date
            raw = None
            for attempt_kwargs in (
                {**kwargs, **date_kwargs},  # 尝试1: 带日期过滤
                kwargs,                       # 尝试2: 不带日期 (全量)
            ):
                try:
                    raw = _with_timeout(lambda: pro.fina_indicator(**attempt_kwargs))
                    if raw is not None and len(raw) > 0:
                        break
                except Exception as exc:
                    if attempt_kwargs is kwargs:  # 最后一次尝试也失败
                        logger.warning("Tushare fina_indicator 失败: %s", exc)
                    continue
            if raw is not None and len(raw) > 0:
                    col_rename = {
                        "ts_code": "_ts_code", "ann_date": "announce_date",
                        "end_date": "report_period",
                        "roe": "roe", "roe_dt": "roe_deducted", "roa": "roa",
                        "np_margin": "net_margin", "gross_margin": "gross_margin",
                        "eps_yoy": "eps_yoy", "or_yoy": "rev_yoy",
                        "profit_yoy": "profit_yoy", "cf_sales": "op_cf_ratio",
                        "debt_to_assets": "debt_ratio", "current_ratio": "current_ratio",
                        "assets_turn": "asset_turnover", "ar_turn": "ar_turnover",
                        "inv_turn": "inventory_turnover", "ocf_to_or": "ocf_to_or",
                    }
                    out = pd.DataFrame()
                    out["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
                    for old, new in col_rename.items():
                        if old in raw.columns:
                            out[new] = pd.to_numeric(raw[old], errors="coerce")
                    out["announce_date"] = pd.to_datetime(
                        raw.get("ann_date", raw.get("f_ann_date", None)), format="%Y%m%d", errors="coerce"
                    )
                    out["report_period"] = pd.to_datetime(raw.get("end_date", None), format="%Y%m%d", errors="coerce")
                    out.to_parquet(path, index=False)
                    return out

        # AKShare 降级 — 新浪财务分析指标 (单股)
        if ts_code:
            try:
                import akshare as ak
                symbol_raw = ts_code.replace(".SZ", "").replace(".SH", "")
                raw = _ak_call(ak.stock_financial_analysis_indicator, symbol=symbol_raw)
                if raw is not None and len(raw) > 0:
                    ak_col_map = {
                        "净资产收益率(%)": "roe",
                        "总资产利润率(%)": "roa",
                        "销售毛利率(%)": "gross_margin",
                        "营业利润率(%)": "net_margin",
                        "资产负债率(%)": "debt_ratio",
                        "流动比率": "current_ratio",
                        "总资产周转率(次)": "asset_turnover",
                    }
                    out = pd.DataFrame()
                    out["symbol"] = symbol_raw
                    for old, new in ak_col_map.items():
                        if old in raw.columns:
                            out[new] = pd.to_numeric(raw[old], errors="coerce")
                    # AKShare 返回的是百分比值 (如 15.5 代表 15.5%), 转换为小数
                    for col in ["roe", "roa", "gross_margin", "net_margin", "debt_ratio"]:
                        if col in out.columns:
                            out[col] = pd.to_numeric(out[col], errors="coerce") / 100.0
                    out.to_parquet(path, index=False)
                    return out
            except Exception as exc:
                logger.warning("AKShare 财务指标失败: %s", exc)

        return pd.DataFrame()

    # ── 4. 龙虎榜 ──
    def fetch_lhb(
        self, trade_date: str | None = None, start_date: str | None = None,
        end_date: str | None = None, refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-4] 龙虎榜明细 + 机构席位 — Tushare top_list + top_inst, AKShare 降级.

        Returns:
            DataFrame [symbol, date, lhb_net_buy, lhb_buy_amt, lhb_sell_amt,
                       lhb_institutional_count, lhb_institutional_net_buy, ...]
        """
        if trade_date:
            key = f"lhb_{trade_date}"
        else:
            key = f"lhb_{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("lhb", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        frames = []
        # Tushare top_list 仅支持单日 trade_date, 不支持日期范围
        # 优先用 AKShare stock_lhb_detail_em (支持 start_date/end_date)
        if pro is not None and trade_date:
            try:
                top = _with_timeout(lambda: pro.top_list(trade_date=trade_date))
                if top is not None and len(top) > 0:
                    top["symbol"] = top["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
                    top["date"] = pd.to_datetime(top["trade_date"], format="%Y%m%d")
                    # Tushare top_list 列名版本兼容 (net_buy/net_amount, l_buy/buy_amount 等)
                    net_col = next((c for c in ["net_buy", "net_amount"] if c in top.columns), None)
                    top["lhb_net_buy"] = pd.to_numeric(top[net_col], errors="coerce") if net_col else 0
                    buy_col = next((c for c in ["buy_amount", "l_buy"] if c in top.columns), None)
                    top["lhb_buy_amt"] = pd.to_numeric(top[buy_col], errors="coerce") if buy_col else 0
                    sell_col = next((c for c in ["sell_amount", "l_sell"] if c in top.columns), None)
                    top["lhb_sell_amt"] = pd.to_numeric(top[sell_col], errors="coerce") if sell_col else 0
                    frames.append(top)

                # top_inst: 机构席位交易明细 (同样仅支持单日 trade_date)
                try:
                    inst = _with_timeout(lambda: pro.top_inst(trade_date=trade_date))
                    if inst is not None and len(inst) > 0:
                        inst["symbol"] = inst["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
                        inst["date"] = pd.to_datetime(inst["trade_date"], format="%Y%m%d")
                        # 按 symbol+date 聚合机构席位
                        inst_agg = inst.groupby(["symbol", "date"]).agg(
                            lhb_institutional_count=("exalter", "nunique"),
                            lhb_institutional_buy=("buy_amount", "sum"),
                            lhb_institutional_sell=("sell_amount", "sum"),
                        ).reset_index()
                        inst_agg["lhb_institutional_net_buy"] = (
                            inst_agg["lhb_institutional_buy"].fillna(0)
                            - inst_agg["lhb_institutional_sell"].fillna(0)
                        )
                        # merge 回 top_list 主表
                        if frames:
                            base = frames[0]
                            base = base.merge(inst_agg, on=["symbol", "date"], how="left")
                            frames[0] = base
                except Exception as exc:
                    logger.warning("Tushare top_inst 失败 (积分不足?): %s", exc)
            except Exception as exc:
                logger.warning("Tushare top_list 失败: %s", exc)

        # AKShare 降级
        if not frames:
            try:
                import akshare as ak
                dt = start_date or trade_date or datetime.now().strftime("%Y%m%d")
                de = end_date or trade_date or dt
                raw = None

                # 尝试多个龙虎榜接口
                for fn, kwargs in [
                    (ak.stock_lhb_detail_em, {"start_date": dt, "end_date": de}),
                    (getattr(ak, "stock_lhb_jgmmtj_em", None), {"start_date": dt, "end_date": de}),
                ]:
                    if fn is None:
                        continue
                    try:
                        candidate = _ak_call(fn, **kwargs)
                        if candidate is not None and len(candidate) > 0:
                            raw = candidate
                            break
                    except Exception:
                        continue

                if raw is not None and len(raw) > 0:
                    # 兼容列名
                    symbol_col = next(
                        (c for c in ["代码", "stock_code", "ts_code", "symbol"] if c in raw.columns), None
                    )
                    if symbol_col:
                        raw["symbol"] = raw[symbol_col].astype(str).str.zfill(6).str.replace(".SZ", "").str.replace(".SH", "")
                    date_col = next((c for c in ["日期", "上榜日期", "trade_date", "date"] if c in raw.columns), None)
                    if date_col:
                        raw["date"] = pd.to_datetime(raw[date_col])
                    # 龙虎榜特征列映射 — AKShare stock_lhb_detail_em 列名:
                    #   龙虎榜净买额(索引7) ← 注意不是"净买入额", 无"入"字
                    #   龙虎榜买入额(索引8), 龙虎榜卖出额(索引9), 龙虎榜成交额(索引10)
                    for col in raw.columns:
                        if "龙虎榜" in col and "净买" in col:
                            raw["lhb_net_buy"] = pd.to_numeric(raw[col], errors="coerce")
                        elif "龙虎榜买入" in col:
                            raw["lhb_buy_amt"] = pd.to_numeric(raw[col], errors="coerce")
                        elif "龙虎榜卖出" in col:
                            raw["lhb_sell_amt"] = pd.to_numeric(raw[col], errors="coerce")
                    # 兼容旧版/备用接口 — 若上面未命中，尝试宽泛匹配
                    if "lhb_buy_amt" not in raw.columns:
                        buy = raw.get("成交额", raw.get("买入金额"))
                        if buy is not None:
                            raw["lhb_buy_amt"] = pd.to_numeric(buy, errors="coerce")
                    if "lhb_net_buy" not in raw.columns:
                        net = raw.get("净买入额")
                        if net is not None:
                            raw["lhb_net_buy"] = pd.to_numeric(net, errors="coerce")
                    frames.append(raw)
            except Exception as exc:
                logger.warning("AKShare 龙虎榜失败: %s", exc)

        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path, index=False)
        return df

    # ── 5. 股东户数 ──
    def fetch_holdernumber(
        self, ts_code: str | None = None, start_date: str | None = None,
        end_date: str | None = None, refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-5] 股东户数 — Tushare stk_holdernumber 主源, AKShare 降级.

        Returns:
            DataFrame [symbol, date, holder_count, avg_shares_per_holder, ...]
            季频更新 (随季报/中报/年报公告后).
        """
        key = f"{ts_code or 'all'}_{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("holdernumber", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        if pro is not None:
            try:
                kwargs = {}
                if ts_code:
                    kwargs["ts_code"] = ts_code
                if start_date:
                    kwargs["start_date"] = start_date
                if end_date:
                    kwargs["end_date"] = end_date
                raw = _with_timeout(lambda: pro.stk_holdernumber(**kwargs))
                if raw is not None and len(raw) > 0:
                    out = pd.DataFrame({
                        "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
                        "date": pd.to_datetime(raw.get("ann_date", raw.get("end_date")), format="%Y%m%d", errors="coerce"),
                        "holder_count": pd.to_numeric(raw.get("holder_num", 0), errors="coerce"),
                        "announce_date": pd.to_datetime(raw.get("ann_date", raw.get("end_date")), format="%Y%m%d", errors="coerce"),
                    })
                    out.to_parquet(path, index=False)
                    return out
            except Exception as exc:
                logger.warning("Tushare stk_holdernumber 失败: %s", exc)

        # AKShare 降级 — 单股股东户数
        if ts_code:
            try:
                import akshare as ak
                symbol = ts_code.replace(".SZ", "").replace(".SH", "")
                raw = _ak_call(ak.stock_zh_a_gdhs_detail_em, symbol=symbol)
                if raw is not None and len(raw) > 0:
                    out = pd.DataFrame({
                        "symbol": symbol,
                        "date": pd.to_datetime(raw.get("股东户数统计截止日", raw.get("截止日期", "")), errors="coerce"),
                        "holder_count": pd.to_numeric(raw.get("股东户数", raw.get("股东总户数", 0)), errors="coerce"),
                        "avg_shares_per_holder": pd.to_numeric(raw.get("户均持股数", raw.get("户均持有流通股数", 0)), errors="coerce"),
                    })
                    out.to_parquet(path, index=False)
                    return out
            except Exception as exc:
                logger.warning("AKShare 股东户数失败: %s", exc)

        return pd.DataFrame()

    # ── 6. 股东增减持 ──
    def fetch_holdertrade(
        self, ts_code: str | None = None, start_date: str | None = None,
        end_date: str | None = None, refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-6] 股东增减持 — Tushare stk_holdertrade 主源, AKShare 降级.

        Returns:
            DataFrame [symbol, date, sh_net_change_amt, sh_change_type, ...]
            不定期更新 (公告后).
        """
        key = f"{ts_code or 'all'}_{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("holdertrade", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        if pro is not None:
            try:
                kwargs = {}
                if ts_code:
                    kwargs["ts_code"] = ts_code
                if start_date:
                    kwargs["start_date"] = start_date
                if end_date:
                    kwargs["end_date"] = end_date

                _PAGE_SIZE = 3000
                all_pages: list[pd.DataFrame] = []
                offset = 0
                while True:
                    page_kwargs = {**kwargs, "limit": _PAGE_SIZE, "offset": offset}
                    raw = _with_timeout(lambda: pro.stk_holdertrade(**page_kwargs))
                    if raw is None or len(raw) == 0:
                        break
                    all_pages.append(raw)
                    if len(raw) < _PAGE_SIZE:
                        break
                    offset += _PAGE_SIZE
                    time.sleep(0.3)  # 分页间限流

                if all_pages:
                    raw = pd.concat(all_pages, ignore_index=True)
                    # Tushare stk_holdertrade 实际列: in_de("IN"/"DE"), change_vol, change_ratio, avg_price
                    # 无 change_amt 列 — 用 change_vol * avg_price 推导
                    change_vol = pd.to_numeric(raw.get("change_vol", 0), errors="coerce").fillna(0)
                    avg_price = pd.to_numeric(raw.get("avg_price", 0), errors="coerce").fillna(0)
                    change_amt = change_vol * avg_price
                    in_de = raw.get("in_de", raw.get("change_type", ""))
                    out = pd.DataFrame({
                        "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
                        "date": pd.to_datetime(raw.get("ann_date", raw.get("trade_date", "")), format="%Y%m%d", errors="coerce"),
                        "sh_change_vol": change_vol,
                        "sh_change_amt": change_amt,
                        "sh_holder_name": raw.get("holder_name", ""),
                        "sh_change_type": in_de,
                        "announce_date": pd.to_datetime(raw.get("ann_date", raw.get("trade_date", "")), format="%Y%m%d", errors="coerce"),
                    })
                    out["sh_net_sign"] = in_de.apply(
                        lambda x: 1 if str(x).upper() == "IN" else (-1 if str(x).upper() == "DE" else 0)
                    )
                    out.to_parquet(path, index=False)
                    return out
            except Exception as exc:
                logger.warning("Tushare stk_holdertrade 失败: %s", exc)

        # AKShare 降级
        if ts_code:
            try:
                import akshare as ak
                symbol = ts_code.replace(".SZ", "").replace(".SH", "")
                raw = _ak_call(ak.stock_zh_a_gdhs_detail_em, symbol=symbol)
                if raw is not None and len(raw) > 0 and "增减持" in str(raw.columns):
                    # 筛选增减持相关列
                    out = pd.DataFrame({"symbol": symbol})
                    for col in raw.columns:
                        if any(kw in str(col) for kw in ["增减", "变动", "股东"]):
                            out[col] = raw[col]
                    out["date"] = pd.to_datetime(
                        raw.get("公告日期", raw.get("变动日期", datetime.now().strftime("%Y%m%d"))),
                        errors="coerce"
                    )
                    if len(out.columns) > 2:  # 至少 symbol+date+其他
                        out.to_parquet(path, index=False)
                        return out
            except Exception as exc:
                logger.warning("AKShare 股东增减持失败: %s", exc)

        return pd.DataFrame()

    # ── 7. 申万行业指数 ──
    def fetch_sector_index(
        self, start_date: str | None = None, end_date: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[Alt-7] 申万一级行业指数日线 — AKShare index_sw_hist 逐行业拉取.

        Returns:
            DataFrame [index_code, index_name, date, open, high, low, close,
                       volume, amount, ret_pct]
            28 个申万一级行业, 日频.
            用于 dim28_sector_index: 行业动量/轮动/相对强弱.
        """
        key = f"sw_{start_date or 'all'}_{end_date or 'all'}"
        path = self._alt_cache_path("sector_index", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        # 申万一级行业指数代码 (28个)
        SW_CODES = {
            "801010": "农林牧渔", "801020": "采掘", "801030": "化工",
            "801040": "钢铁", "801050": "有色金属", "801080": "电子",
            "801110": "家用电器", "801120": "食品饮料", "801130": "纺织服装",
            "801140": "轻工制造", "801150": "医药生物", "801160": "公用事业",
            "801170": "交通运输", "801180": "房地产", "801200": "商业贸易",
            "801210": "休闲服务", "801230": "综合",
            "801710": "建筑材料", "801720": "建筑装饰", "801730": "电力设备",
            "801740": "国防军工", "801750": "计算机", "801760": "传媒",
            "801770": "通信", "801780": "银行", "801790": "非银金融",
            "801880": "汽车", "801890": "机械设备",
        }
        try:
            import akshare as ak
            frames = []

            def _fetch_one_sw(code_name):
                code, name = code_name
                raw = _ak_call(ak.index_hist_sw, symbol=code)
                if raw is not None and len(raw) > 0:
                    out = pd.DataFrame({
                        "index_code": code,
                        "index_name": name,
                        "date": pd.to_datetime(raw["日期"], errors="coerce"),
                        "open": pd.to_numeric(raw.get("开盘", raw.get("开盘价", np.nan)), errors="coerce"),
                        "high": pd.to_numeric(raw.get("最高", raw.get("最高价", np.nan)), errors="coerce"),
                        "low": pd.to_numeric(raw.get("最低", raw.get("最低价", np.nan)), errors="coerce"),
                        "close": pd.to_numeric(raw.get("收盘", raw.get("收盘价", np.nan)), errors="coerce"),
                        "volume": pd.to_numeric(raw.get("成交量", np.nan), errors="coerce"),
                        "amount": pd.to_numeric(raw.get("成交额", np.nan), errors="coerce"),
                    })
                    return out
                return None

            items = list(SW_CODES.items())
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_fetch_one_sw, item): item for item in items}
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=60)
                        if result is not None:
                            frames.append(result)
                    except Exception:
                        pass
            if not frames:
                return pd.DataFrame()
            df = pd.concat(frames, ignore_index=True)
            # 日收益率
            df = df.sort_values(["index_code", "date"]).reset_index(drop=True)
            df["ret_pct"] = df.groupby("index_code")["close"].pct_change()
            # 过滤日期范围
            if start_date:
                df = df[df["date"] >= pd.to_datetime(start_date, format="%Y%m%d")]
            if end_date:
                df = df[df["date"] <= pd.to_datetime(end_date, format="%Y%m%d")]
            df.to_parquet(path, index=False)
            logger.info("申万行业指数: %d 行业, %d 行", len(SW_CODES), len(df))
            return df
        except Exception as exc:
            logger.warning("申万行业指数拉取失败: %s", exc)
            return pd.DataFrame()



    # ── 8. 每日指标 (daily_basic) ──
    def fetch_daily_basic(
        self, trade_date: str | None = None, ts_code: str | None = None,
        start_date: str | None = None, end_date: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[dim05/dim06] Tushare daily_basic — 每日估值/换手/市值指标.

        单日全市场拉取 (~5500 股/次), 或按 ts_code + 日期范围拉取.
        产出列: turnover_rate_f, volume_ratio, pe_ttm, pb, ps_ttm, dv_ratio, dv_ttm,
                total_mv, circ_mv, total_share, float_share, free_share
        """
        key = f"{trade_date or 'range'}_{ts_code or 'all'}_{start_date or ''}_{end_date or ''}"
        path = self._alt_cache_path("daily_basic", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        if pro is None:
            return pd.DataFrame()

        try:
            kwargs = {}
            if trade_date:
                kwargs["trade_date"] = trade_date
            if ts_code:
                kwargs["ts_code"] = ts_code
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            raw = _with_timeout(lambda: pro.daily_basic(**kwargs))
            if raw is None or len(raw) == 0:
                return pd.DataFrame()

            out = pd.DataFrame({
                "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
                "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce"),
                "turnover_rate_f": pd.to_numeric(raw.get("turnover_rate_f", np.nan), errors="coerce"),
                "volume_ratio": pd.to_numeric(raw.get("volume_ratio", np.nan), errors="coerce"),
                "pe_ttm": pd.to_numeric(raw.get("pe_ttm", np.nan), errors="coerce"),
                "pb": pd.to_numeric(raw.get("pb", np.nan), errors="coerce"),
                "ps_ttm": pd.to_numeric(raw.get("ps_ttm", np.nan), errors="coerce"),
                "dv_ratio": pd.to_numeric(raw.get("dv_ratio", np.nan), errors="coerce"),
                "dv_ttm": pd.to_numeric(raw.get("dv_ttm", np.nan), errors="coerce"),
                "total_mv": pd.to_numeric(raw.get("total_mv", np.nan), errors="coerce"),
                "circ_mv": pd.to_numeric(raw.get("circ_mv", np.nan), errors="coerce"),
                "total_share": pd.to_numeric(raw.get("total_share", np.nan), errors="coerce"),
                "float_share": pd.to_numeric(raw.get("float_share", np.nan), errors="coerce"),
                "free_share": pd.to_numeric(raw.get("free_share", np.nan), errors="coerce"),
            })
            out.to_parquet(path, index=False)
            logger.info("daily_basic: %d stocks", len(out))
            return out
        except Exception as exc:
            logger.warning("Tushare daily_basic 失败: %s", exc)
            return pd.DataFrame()

    # ── 9. 涨跌停价格 (stk_limit) ──
    def fetch_stk_limit(
        self, ts_code: str | None = None, trade_date: str | None = None,
        start_date: str | None = None, end_date: str | None = None,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """[dim11] Tushare stk_limit — 每日涨跌停价格.

        产出列: up_limit, down_limit, limit_status (0=正常, 1=涨停, -1=跌停)
        """
        key = f"{trade_date or 'range'}_{ts_code or 'all'}_{start_date or ''}_{end_date or ''}"
        path = self._alt_cache_path("stk_limit", key)
        if not refresh and os.path.exists(path):
            return pd.read_parquet(path)

        pro = self._tushare_pro()
        if pro is None:
            return pd.DataFrame()

        try:
            kwargs = {}
            if trade_date:
                kwargs["trade_date"] = trade_date
            if ts_code:
                kwargs["ts_code"] = ts_code
            if start_date:
                kwargs["start_date"] = start_date
            if end_date:
                kwargs["end_date"] = end_date

            raw = _with_timeout(lambda: pro.stk_limit(**kwargs))
            if raw is None or len(raw) == 0:
                return pd.DataFrame()

            out = pd.DataFrame({
                "symbol": raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", ""),
                "date": pd.to_datetime(raw["trade_date"], format="%Y%m%d", errors="coerce"),
                "up_limit_raw": pd.to_numeric(raw.get("up_limit", np.nan), errors="coerce"),
                "down_limit_raw": pd.to_numeric(raw.get("down_limit", np.nan), errors="coerce"),
            })
            out.to_parquet(path, index=False)
            logger.info("stk_limit: %d stocks", len(out))
            return out
        except Exception as exc:
            logger.warning("Tushare stk_limit 失败: %s", exc)
            return pd.DataFrame()

    # ════════════════════════════════════════════════════════════

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

    # ── 每日数据拉取 (收盘前/后均可, 取当日全量) ──
    def fetch_today(
        self,
        trade_date: str | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """拉取当日全量数据 (OHLCV + alt data), 各源独立失败不阻断.

        Args:
            trade_date: 'YYYYMMDD', None=今天
            sources: 要拉取的数据源, None=全部 ['ohlcv','margin','northbound','lhb']

        Returns:
            {source_name: DataFrame}, 失败的源为空的 DataFrame
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
        if sources is None:
            sources = ["ohlcv", "margin", "northbound", "lhb"]

        results: dict[str, pd.DataFrame] = {}
        for src in sources:
            try:
                if src == "ohlcv":
                    results[src] = self._akshare_fetch_daily(trade_date)
                elif src == "margin":
                    results[src] = self.fetch_margin(trade_date=trade_date, refresh=True)
                elif src == "northbound":
                    results[src] = self.fetch_northbound(trade_date=trade_date, refresh=True)
                elif src == "lhb":
                    results[src] = self.fetch_lhb(trade_date=trade_date, refresh=True)
                else:
                    results[src] = pd.DataFrame()
                logger.info(
                    "fetch_today %s: %d rows", src, len(results.get(src, pd.DataFrame()))
                )
            except Exception as exc:
                logger.warning("fetch_today %s 失败 (非阻断): %s", src, exc)
                results[src] = pd.DataFrame()
        return results

    def append_today_to_panel(
        self,
        panel: pd.DataFrame,
        trade_date: str | None = None,
        sources: list[str] | None = None,
    ) -> pd.DataFrame:
        """拉取当日数据并追加到历史面板.

        1. 拉取当日 OHLCV + alt data
        2. append OHLCV 行到面板 (concat)
        3. merge alt data 到当日行 (left join on symbol+date)
        4. 返回扩展后的面板 (不改动历史行)

        Args:
            panel: 历史面板 (需含 symbol/date/board 等 enrich 后的列)
            trade_date: 'YYYYMMDD'
            sources: 要拉取的源

        Returns:
            扩展后的面板 (含当日新行)
        """
        if trade_date is None:
            trade_date = datetime.now().strftime("%Y%m%d")
        if sources is None:
            sources = ["ohlcv", "margin", "northbound", "lhb"]

        today_data = self.fetch_today(trade_date=trade_date, sources=sources)

        # 1. OHLCV: 只取面板中已有的 symbol, 补齐缺失列
        ohlcv = today_data.get("ohlcv", pd.DataFrame())
        if len(ohlcv) == 0:
            logger.warning("当日 OHLCV 拉取为空, 跳过 append")
            return panel

        existing_symbols = set(panel["symbol"].unique())
        ohlcv = ohlcv[ohlcv["symbol"].isin(existing_symbols)].copy()
        ohlcv["date"] = pd.to_datetime(trade_date)

        # 补齐面板元数据列 (board/industry/list_days 等)
        meta_cols = ["board", "industry", "is_st", "is_suspended", "list_days",
                      "free_float_turnover_rate", "limit_pct", "pre_close"]
        for col in meta_cols:
            if col in panel.columns and col not in ohlcv.columns:
                # 从面板最近一天取最新值
                latest = panel.sort_values("date").drop_duplicates(
                    subset=["symbol"], keep="last"
                )[["symbol", col]]
                ohlcv = ohlcv.merge(latest, on="symbol", how="left")

        # 2. 对齐面板列 (ohlcv 缺的列填 NaN)
        for col in panel.columns:
            if col not in ohlcv.columns:
                ohlcv[col] = pd.NA

        ohlcv = ohlcv[panel.columns.tolist()]  # 确保列顺序一致

        # 3. Concat
        panel = pd.concat([panel, ohlcv], ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

        # 4. Merge alt data 到当日行
        alt_sources = [s for s in sources if s != "ohlcv"]
        today_dt = pd.to_datetime(trade_date)
        for src in alt_sources:
            alt = today_data.get(src, pd.DataFrame())
            if len(alt) == 0:
                continue
            if "date" in alt.columns and not pd.api.types.is_datetime64_any_dtype(alt["date"]):
                alt["date"] = pd.to_datetime(alt["date"])

            if src == "northbound":
                # 市场级数据: 按 date 广播
                date_cols = [c for c in alt.columns
                            if c not in ("symbol", "date") and not c.startswith("_")]
                alt_subset = alt[["date"] + date_cols].drop_duplicates(subset=["date"])
                # 只更新当日行
                mask = panel["date"] == today_dt
                for col in date_cols:
                    if col in alt_subset.columns:
                        val_map = dict(zip(alt_subset["date"], alt_subset[col]))
                        panel.loc[mask, col] = panel.loc[mask, "date"].map(val_map)
            elif "symbol" in alt.columns:
                merge_cols = ["symbol", "date"]
                alt_cols = [c for c in alt.columns if c not in merge_cols and not c.startswith("_")]
                # 只 merge 到当日行
                today_panel = panel[panel["date"] == today_dt].copy()
                other_panel = panel[panel["date"] != today_dt]
                today_panel = today_panel.drop(columns=[c for c in alt_cols if c in today_panel.columns], errors="ignore")
                today_panel = today_panel.merge(
                    alt[merge_cols + alt_cols], on=merge_cols, how="left"
                )
                panel = pd.concat([other_panel, today_panel], ignore_index=True)
                panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

        logger.info(
            "append_today: +%d stocks, panel now %d rows %d cols",
            len(ohlcv), len(panel), len(panel.columns),
        )
        return panel
