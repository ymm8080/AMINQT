"""
看板数据服务层 (P10) — 纯函数, 可测试
==========================================
数据源优先级: 真实产出 (data/lists/*.parquet, data/tuning_report.json) >
本地缓存 > 合成演示数据 (DEMO_MODE 标记, 页面顶部显著提示).
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
import pandas as pd

from config.settings import data_others_path

LIST_DIR = "data/lists"
TUNING_REPORT = str(data_others_path("data/tuning_report.json"))
WATCHLIST_PATH = str(data_others_path("data/watchlist.json"))
PRIORITY_PATH = str(data_others_path("data/priority.json"))

DEMO_SYMBOLS = [
    "600519",
    "300750",
    "601318",
    "600000",
    "000001",
    "002594",
    "688981",
    "600036",
    "000858",
    "601899",
]
DEMO_NAMES = {
    "600519": "贵州茅台",
    "300750": "宁德时代",
    "601318": "中国平安",
    "600000": "浦发银行",
    "000001": "平安银行",
    "002594": "比亚迪",
    "688981": "中芯国际",
    "600036": "招商银行",
    "000858": "五粮液",
    "601899": "紫金矿业",
}
DEMO_INDUSTRIES = {
    "600519": "白酒",
    "300750": "电池",
    "601318": "保险",
    "600000": "银行",
    "000001": "银行",
    "002594": "汽车",
    "688981": "半导体",
    "600036": "银行",
    "000858": "白酒",
    "601899": "有色",
}
DEMO_SECTOR_BASE_INDEX: dict[str, float] = {
    "白酒": 1.0,
    "电池": 1.0,
    "保险": 1.0,
    "银行": 1.0,
    "汽车": 1.0,
    "半导体": 1.0,
    "有色": 1.0,
}


# ============================================================
# 重点股 (次日日内操作候选池)
# ============================================================
def load_priority_symbols(path: str = PRIORITY_PATH) -> set[str]:
    """读取已保存的重点股代码集合."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return set()
    if isinstance(data, list):
        return set(str(s).strip() for s in data if s)
    if isinstance(data, dict):
        return set(str(s).strip() for s in data.get("symbols", []) if s)
    return set()


def save_priority_symbols(symbols: set[str], path: str = PRIORITY_PATH) -> None:
    """保存重点股代码集合."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"symbols": sorted(symbols)}, fh, ensure_ascii=False, indent=1)


def toggle_priority(symbol: str, path: str = PRIORITY_PATH) -> bool:
    """切换重点股状态. 返回新状态 (True=已标记)."""
    symbols = load_priority_symbols(path)
    symbol = str(symbol).strip()
    if symbol in symbols:
        symbols.discard(symbol)
        save_priority_symbols(symbols, path)
        return False
    symbols.add(symbol)
    save_priority_symbols(symbols, path)
    return True


def pipeline_buy_candidates(df: pd.DataFrame) -> set[str]:
    """根据 Pipeline-1 清单规则选出推荐买入候选.

    规则优先级:
    1. 存在 pred_ret_1d 列时, 取预测次日收益 > 0 的股票;
    2. 否则存在 score 列时, 取 score 前 30%;
    3. 否则取前 5 名。
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return set()
    if "pred_ret_1d" in df.columns:
        return set(df.loc[df["pred_ret_1d"] > 0, "symbol"])
    if "score" in df.columns:
        threshold = df["score"].quantile(0.7)
        return set(df.loc[df["score"] >= threshold, "symbol"])
    return set(df["symbol"].head(5))


def apply_priority_tags(
    df: pd.DataFrame, priority_path: str = PRIORITY_PATH
) -> pd.DataFrame:
    """为清单打日内交易标签: 只读 priority.json.

    Pipeline1 程序离线运行时写入 priority.json; 用户手工 toggle 也写 priority.json.
    页面不做自动计算, 手工更改不会被覆盖.

    Args:
        df: 清单 DataFrame.
        priority_path: priority.json 路径 (测试可注入临时路径).
    """
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    df = df.copy()
    saved = load_priority_symbols(priority_path)
    df["priority"] = df["symbol"].isin(saved)
    return df


# ============================================================
# 清单 (Pipeline-1 产出)
# ============================================================
def list_available_dates(list_dir: str = LIST_DIR) -> list[str]:
    """返回已有清单的日期列表 (降序)."""
    dates = [
        os.path.basename(p).replace("list_", "").replace(".parquet", "")
        for p in glob.glob(os.path.join(list_dir, "list_*.parquet"))
    ]
    return sorted(dates, reverse=True)


def load_list(trade_date: str, list_dir: str = LIST_DIR) -> pd.DataFrame | None:
    """加载某日清单; 不存在返回 None."""
    path = os.path.join(list_dir, f"list_{trade_date}.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else None


def load_latest_list(
    list_dir: str = LIST_DIR,
    priority_path: str = PRIORITY_PATH,
) -> tuple[pd.DataFrame | None, str | None]:
    """加载最新清单 → (df, date); 无清单返回 (None, None).

    Args:
        list_dir: 清单目录.
        priority_path: priority.json 路径 (测试可注入临时路径).
    """
    dates = list_available_dates(list_dir)
    if not dates:
        return None, None
    df = load_list(dates[0], list_dir)
    if df is not None:
        df = apply_priority_tags(df, priority_path)
    return df, dates[0]


def demo_list(seed: int = 42) -> pd.DataFrame:
    """合成演示清单 (schema V1.0 同构)."""
    rng = np.random.default_rng(seed)
    n = len(DEMO_SYMBOLS)
    df = pd.DataFrame(
        {
            "symbol": DEMO_SYMBOLS,
            "board": [
                "main",
                "GEM",
                "main",
                "main",
                "main",
                "main",
                "STAR",
                "main",
                "main",
                "main",
            ],
            "day_change": rng.uniform(-0.03, 0.06, n),
            "pred_ret_1d": rng.uniform(-0.02, 0.05, n),
            "pred_ret_2d": rng.uniform(-0.03, 0.08, n),
            "pred_ret_3d": rng.uniform(-0.03, 0.09, n),
            "pred_ret_5d": rng.uniform(-0.04, 0.12, n),
            "prob_up": np.round(rng.uniform(0.42, 0.62, n), 3),
            "prob_up_2d": np.round(rng.uniform(0.40, 0.66, n), 3),
            "prob_up_3d": np.round(rng.uniform(0.38, 0.68, n), 3),
            "prob_up_5d": np.round(rng.uniform(0.36, 0.70, n), 3),
            "momentum": rng.choice(["high", "medium", "low"], n, p=[0.3, 0.5, 0.2]),
            "consensus_score": rng.uniform(1, n, n),
            "signal_conflict": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "is_limit_up_close": 0,
            "is_one_word_limit": 0,
            "market_state": "range",
            "score": rng.uniform(0, 0.05, n),
            # V1.2 新增列 (E1/E2/公告/分布权重)
            "pred_q10": rng.uniform(-0.04, 0.01, n),
            "pred_q50": rng.uniform(-0.01, 0.04, n),
            "pred_q90": rng.uniform(0.01, 0.10, n),
            "uncertainty_width": rng.uniform(0.02, 0.12, n),
            "pain_prob": np.round(rng.uniform(0.0, 0.5, n), 3),
            "announce_score": rng.uniform(-1.0, 1.0, n),
            "weight": np.round(rng.uniform(0.02, 0.10, n), 4),
            # V1.4 新增列 (多视界加权收益/概率)
            "compound_ret": np.round(rng.uniform(0.0, 0.06, n), 6),
            "compound_prob": np.round(rng.uniform(0.42, 0.62, n), 6),
            "schema_version": "1.4",
        }
    )
    df["name"] = df["symbol"].map(DEMO_NAMES)
    df["industry"] = df["symbol"].map(DEMO_INDUSTRIES)
    df = apply_priority_tags(df)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ============================================================
# 行情面板 (K线用)
# ============================================================
def demo_ohlc(symbol: str, days: int = 120, seed: int | None = None) -> pd.DataFrame:
    """合成个股日线 (详情弹窗 K线/副图用)."""
    rng = np.random.default_rng(seed if seed is not None else hash(symbol) % 10000)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    close = 50 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
    open_ = close * (1 + rng.normal(0, 0.005, days))
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.006, days))),
            "low": np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.006, days))),
            "close": close,
            "volume": rng.integers(int(1e6), int(5e7), days).astype(float),
        }
    )


def demo_intraday(symbol: str, seed: int = 7) -> pd.DataFrame:
    """合成当日 2 分钟分时 (120 根)."""
    rng = np.random.default_rng(seed + hash(symbol) % 100)
    n = 120
    times = pd.date_range("09:30", periods=n, freq="2min").strftime("%H:%M")
    price = 100 * np.cumprod(1 + rng.normal(0, 0.0015, n))
    return pd.DataFrame(
        {
            "time": times,
            "price": price,
            "volume": rng.integers(1000, 50000, n).astype(float),
        }
    )


# ============================================================
# 板块行情 (演示)
# ============================================================
def demo_sector_changes(seed: int = 11) -> pd.DataFrame:
    """合成板块当日涨跌幅排行 (演示用)."""
    rng = np.random.default_rng(seed)
    sectors = list(DEMO_SECTOR_BASE_INDEX.keys())
    pct = rng.uniform(-0.025, 0.035, len(sectors))
    # 让银行和白酒相对稳定, 电池/半导体波动大一点 (演示差异)
    multipliers = {
        "白酒": 0.8,
        "银行": 0.5,
        "保险": 0.7,
        "电池": 1.4,
        "半导体": 1.5,
        "汽车": 1.2,
        "有色": 1.3,
    }
    pct = np.array(
        [p * multipliers.get(s, 1.0) for s, p in zip(sectors, pct)], dtype=float
    )
    df = pd.DataFrame(
        {
            "板块": sectors,
            "涨跌幅": pct,
            "上涨家数": rng.integers(5, 80, len(sectors)),
            "下跌家数": rng.integers(5, 80, len(sectors)),
        }
    )
    df["上涨家数"] = df["上涨家数"].clip(lower=df["下跌家数"] * (1 + df["涨跌幅"] * 20))
    df["上涨家数"] = df["上涨家数"].astype(int)
    return df.sort_values("涨跌幅", ascending=False).reset_index(drop=True)


def demo_sector_intraday(sector: str, seed: int = 13) -> pd.DataFrame:
    """合成某板块当日 2 分钟分时曲线 (用于板块 sparkline)."""
    rng = np.random.default_rng(seed + hash(sector) % 100)
    n = 120
    times = pd.date_range("09:30", periods=n, freq="2min").strftime("%H:%M")
    base_index = DEMO_SECTOR_BASE_INDEX.get(sector, 1.0)
    pct = rng.normal(0, 0.0008, n).cumsum()
    price = base_index * (1 + pct)
    return pd.DataFrame({"time": times, "price": price})


# ============================================================
# 关注股
# ============================================================
def load_watchlist(path: str = WATCHLIST_PATH) -> list[dict]:
    """读取关注股 JSON [{symbol, note, tags, marked_at}]."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return [
            {"symbol": k, **(v if isinstance(v, dict) else {"note": str(v)})}
            for k, v in data.items()
        ]
    return data


def save_watchlist(items: list[dict], path: str = WATCHLIST_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)


def toggle_watchlist(symbol: str, name: str = "", path: str = WATCHLIST_PATH) -> bool:
    """切换关注状态. 返回新状态 (True=已关注)."""
    items = load_watchlist(path)
    symbols = [i["symbol"] for i in items]
    if symbol in symbols:
        items = [i for i in items if i["symbol"] != symbol]
        save_watchlist(items, path)
        return False
    items.append(
        {
            "symbol": symbol,
            "name": name,
            "note": "",
            "tags": [],
            "marked_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        }
    )
    save_watchlist(items, path)
    return True


# ============================================================
# 真实行情数据 (akshare fallback → demo)
# ============================================================

_data_logger = logging.getLogger(__name__)
_OHLC_CACHE: dict[str, tuple[str, pd.DataFrame]] = {}  # key → (date, df)


def fetch_real_ohlc(symbol: str, days: int = 120) -> pd.DataFrame | None:
    """从 akshare 获取真实个股日线 (前复权).

    Returns:
        DataFrame with date/open/high/low/close/volume columns, or None on failure.
    """
    cache_key = f"{symbol}:{days}"
    today = pd.Timestamp.today().strftime("%Y%m%d")
    if cache_key in _OHLC_CACHE and _OHLC_CACHE[cache_key][0] == today:
        return _OHLC_CACHE[cache_key][1].copy()

    try:
        import akshare as ak

        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date="20200101",
            end_date=today,
            adjust="qfq",
        )
        if df is None or len(df) == 0:
            return None
        df = df.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).sort_values("date").tail(days)
        _OHLC_CACHE[cache_key] = (today, df.copy())
        _data_logger.info("akshare OHLC %s: %d bars", symbol, len(df))
        return df
    except Exception:
        _data_logger.warning("akshare OHLC %s 失败, 回退 demo", symbol, exc_info=True)
        return None


def fetch_real_intraday(symbol: str) -> pd.DataFrame | None:
    """从 akshare 获取今日 5 分钟分时 (实时).

    Returns:
        DataFrame with time/price/volume columns, or None on failure.
    """
    try:
        import akshare as ak

        df = ak.stock_zh_a_hist_min_em(symbol=symbol, period="5", adjust="qfq")
        if df is None or len(df) == 0:
            return None
        df = df.rename(
            columns={
                "时间": "time",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
            }
        )
        # Keep only today's bars
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
            today = pd.Timestamp.today().normalize()
            df = df[df["time"] >= today]
            df["time"] = df["time"].dt.strftime("%H:%M")
        result = (
            df[["time", "close", "volume"]].rename(columns={"close": "price"}).copy()
        )
        result["volume"] = pd.to_numeric(result["volume"], errors="coerce").fillna(0)
        result["price"] = pd.to_numeric(result["price"], errors="coerce").ffill()
        _data_logger.info("akshare intraday %s: %d bars", symbol, len(result))
        return result if len(result) else None
    except Exception:
        _data_logger.warning(
            "akshare intraday %s 失败, 回退 demo", symbol, exc_info=True
        )
        return None


# ============================================================
# 调参报告 / 配置
# ============================================================
def load_tuning_report(path: str = TUNING_REPORT) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_yaml(path: str) -> dict:
    import yaml

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_yaml(data: dict, path: str) -> None:
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


# ============================================================
# 回测真实数据适配器 (V5.2 BacktestEngine 格式)
# ============================================================
V3_PANEL_PATH = os.getenv(
    "PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
)


def load_backtest_panel(days: int = 750) -> pd.DataFrame | None:
    """加载 v3 面板 parquet 并截取最近 N 个交易日.

    Returns:
        面板 DataFrame 或 None (文件不存在时).
    """
    if not os.path.exists(V3_PANEL_PATH):
        return None
    try:
        panel = pd.read_parquet(V3_PANEL_PATH)
        if "date" in panel.columns:
            panel["date"] = pd.to_datetime(panel["date"])
            all_dates = sorted(panel["date"].unique())
            if len(all_dates) > days:
                keep_dates = set(all_dates[-days:])
                panel = panel[panel["date"].isin(keep_dates)]
        return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    except Exception as exc:
        _data_logger.error("加载 v3 面板失败: %s", exc)
        return None


def load_predictions_from_db(
    dates: list[pd.Timestamp],
) -> pd.DataFrame | None:
    """从 PredictionDB (SQLite) 加载 pipeline 预测.

    预测来自 predict_runner.run_prediction() → PredictionDB.insert_run().
    包含: symbol, date, score, prob_up, pred_ret_3d 等.

    Args:
        dates: 面板中的日期列表.

    Returns:
        DataFrame (date, symbol, score, prob_up, pred_ret_3d) 或 None.
    """
    try:
        from app.pipeline1.prediction_db import PredictionDB

        db = PredictionDB()
        date_strs = [d.strftime("%Y%m%d") for d in dates]
        import sqlite3

        with sqlite3.connect(db.path) as conn:
            placeholders = ",".join("?" * len(date_strs))
            df = pd.read_sql(
                f"""SELECT date, symbol, score, prob_up,
                       pred_ret_1d, pred_ret_3d, pred_ret_5d
                    FROM prediction_stocks
                    WHERE date IN ({placeholders})""",
                conn,
                params=date_strs,
            )
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
        _data_logger.info(
            "从 PredictionDB 加载 %d 条预测 (%d 日期)", len(df), df["date"].nunique()
        )
        return df
    except Exception as exc:
        _data_logger.warning("加载预测失败, 回退 pctChg proxy: %s", exc)
        return None


def panel_to_v52_format(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list, str] | None:
    """将 v3 面板转换为 V5.2 BacktestEngine 所需的 pred_df + price_df.

    V5.2 引擎需要:
        pred_df:  date, stock, score_h2, prob_up_h2, pred_ret_h2
        price_df: date, stock, open, high, low, close, volume,
                  pre_close, up_limit, down_limit, is_halt, is_st,
                  avg_amount_20d, circ_mv, amount

    Returns:
        (pred_df, price_df, trade_dates, data_version_hash) 或 None.
    """
    if panel is None or panel.empty:
        return None
    try:
        df = panel.copy()
        # 重命名 symbol → stock
        df = df.rename(columns={"symbol": "stock"})
        # 确保 date 为 Timestamp
        df["date"] = pd.to_datetime(df["date"])

        # ---- 构建 price_df ----
        price_cols = [
            "date",
            "stock",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "pre_close",
            "amount",
        ]
        available = [c for c in price_cols if c in df.columns]
        price_df = df[available].copy()
        # 补充缺失列
        if "pre_close" not in price_df.columns:
            price_df["pre_close"] = (
                price_df.groupby("stock")["close"].shift(1).fillna(price_df["close"])
            )
        if "amount" not in price_df.columns:
            price_df["amount"] = price_df.get("volume", 0) * price_df["close"]
        # avg_amount_20d
        price_df["avg_amount_20d"] = price_df.groupby("stock")["amount"].transform(
            lambda x: x.rolling(20, min_periods=1).mean()
        )
        # circ_mv (流通市值, 面板可能没有 → 用 amount 近似)
        if "circ_mv" in df.columns:
            price_df["circ_mv"] = df["circ_mv"]
        else:
            price_df["circ_mv"] = price_df["avg_amount_20d"] * 20
        # 涨跌停: 优先使用面板的 up_limit_raw/down_limit_raw, 回退到 pre_close*1.1
        if "up_limit_raw" in df.columns:
            price_df["up_limit"] = df["up_limit_raw"]
        else:
            price_df["up_limit"] = (price_df["pre_close"] * 1.1).round(2)
        if "down_limit_raw" in df.columns:
            price_df["down_limit"] = df["down_limit_raw"]
        else:
            price_df["down_limit"] = (price_df["pre_close"] * 0.9).round(2)
        # 状态标记: 优先使用面板的 is_suspended/is_st, 回退到 0
        price_df["is_halt"] = (
            df["is_suspended"].astype(int) if "is_suspended" in df.columns else 0
        )
        price_df["is_st"] = df["is_st"].astype(int) if "is_st" in df.columns else 0
        # circ_mv NaN 填 0, 防止 NaN < 2e9 返回 False 绕过 BacktestEngine 流动性过滤
        price_df["circ_mv"] = price_df["circ_mv"].fillna(0)

        # ---- 构建 pred_df ----
        # 优先从 PredictionDB 加载 pipeline 预测; 缺失时用 pctChg 近似
        pred_cols_map = {
            "score": "score_h2",
            "prob_up": "prob_up_h2",
            "pred_ret_3d": "pred_ret_h2",
        }
        pred_df = df[["date", "stock"]].copy()
        has_prediction = any(src in df.columns for src in pred_cols_map)
        # 尝试从 PredictionDB 加载 (pipeline 真实预测)
        if not has_prediction:
            pred_db = load_predictions_from_db(sorted(df["date"].unique()))
            if pred_db is not None and not pred_db.empty:
                pred_db = pred_db.rename(columns={"symbol": "stock"})
                df = df.merge(
                    pred_db[["date", "stock", "score", "prob_up", "pred_ret_3d"]],
                    on=["date", "stock"],
                    how="left",
                    suffixes=("", "_pred"),
                )
                for src in pred_cols_map:
                    if src not in df.columns and f"{src}_pred" in df.columns:
                        df[src] = df[f"{src}_pred"]
                has_prediction = any(src in df.columns for src in pred_cols_map)
                _data_logger.info("已 merge pipeline 预测: %d 条", len(pred_db))
        # DB 预测只覆盖部分日期/股票, 对 NaN 补 pctChg 近似
        if has_prediction and "pctChg" in df.columns:
            pctchg = df["pctChg"].astype(float)
            if "score" in df.columns:
                df["score"] = df["score"].fillna(pctchg / 100.0)
            if "prob_up" in df.columns:
                df["prob_up"] = df["prob_up"].fillna(
                    (pctchg > 0).astype(float) * 0.6 + 0.4
                )
            if "pred_ret_3d" in df.columns:
                df["pred_ret_3d"] = df["pred_ret_3d"].fillna(pctchg / 100.0)
        for src, dst in pred_cols_map.items():
            if src in df.columns:
                pred_df[dst] = df[src].astype(float)
            elif not has_prediction and "pctChg" in df.columns:
                if dst == "pred_ret_h2":
                    pred_df[dst] = df["pctChg"].astype(float) / 100.0
                elif dst == "prob_up_h2":
                    pred_df[dst] = (df["pctChg"].astype(float) > 0).astype(
                        float
                    ) * 0.6 + 0.4
                elif dst == "score_h2":
                    pred_df[dst] = df["pctChg"].astype(float) / 100.0
            else:
                pred_df[dst] = 0.5 if "prob" in dst else 0.0
        # 如果没有 score 列, 用 prob_up 近似
        if "score_h2" not in pred_df.columns or pred_df["score_h2"].sum() == 0:
            if "prob_up_h2" in pred_df.columns:
                pred_df["score_h2"] = pred_df["prob_up_h2"]

        # 交易日历
        trade_dates = sorted(price_df["date"].unique())
        import hashlib

        data_version_hash = (
            "sha256:"
            + hashlib.sha256(
                str(trade_dates[0]).encode() + str(trade_dates[-1]).encode()
            ).hexdigest()[:16]
        )

        return pred_df, price_df, trade_dates, data_version_hash
    except Exception as exc:
        _data_logger.error("面板转 V5.2 格式失败: %s", exc)
        return None


def panel_to_v35_lists(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict] | None:
    """将 v3 面板转换为 V35 BacktestEngineV35 所需的 (panel, daily_lists).

    Returns:
        (panel, daily_lists) 或 None — daily_lists: {date: DataFrame(symbol, score, ...)}.
    """
    if panel is None or panel.empty:
        return None
    try:
        df = panel.copy()
        df["date"] = pd.to_datetime(df["date"])
        # 构建每日清单: 每个交易日取 score 前 30 只
        daily_lists = {}
        score_col = "score" if "score" in df.columns else "prob_up"
        has_score = score_col in df.columns
        # 无预测列时, 优先从 PredictionDB 加载 pipeline 预测
        if not has_score:
            pred_db = load_predictions_from_db(sorted(df["date"].unique()))
            if pred_db is not None and not pred_db.empty:
                df = df.merge(
                    pred_db[["date", "symbol", "score", "prob_up", "pred_ret_3d"]],
                    on=["date", "symbol"],
                    how="left",
                    suffixes=("", "_pred"),
                )
                if "score_pred" in df.columns:
                    df["score"] = df["score"].fillna(df["score_pred"])
                if "prob_up_pred" in df.columns:
                    df["prob_up"] = df["prob_up"].fillna(df["prob_up_pred"])
                score_col = "score" if "score" in df.columns else "prob_up"
                has_score = score_col in df.columns
                _data_logger.info("V35: 已 merge pipeline 预测: %d 条", len(pred_db))
        # DB 预测覆盖部分股票, 对 NaN 补 pctChg
        if has_score and "pctChg" in df.columns:
            if "score" in df.columns:
                df["score"] = df["score"].fillna(df["pctChg"].astype(float))
            if "prob_up" in df.columns:
                df["prob_up"] = df["prob_up"].fillna(
                    (df["pctChg"].astype(float) > 0).astype(float) * 0.6 + 0.4
                )
        # 仍无预测列时, 用 pctChg 作为 score 近似
        if not has_score and "pctChg" in df.columns:
            df["_proxy_score"] = df["pctChg"].astype(float)
            score_col = "_proxy_score"
            has_score = True
        for d, g in df.groupby("date"):
            top = g.nlargest(30, score_col) if score_col in g.columns else g.head(30)
            lst = top[["symbol"]].copy()
            if score_col in top.columns:
                lst["score"] = top[score_col].values
            else:
                lst["score"] = 0.5
            if "prob_up" in top.columns:
                lst["prob_up"] = top["prob_up"].values
            if "industry" in top.columns:
                lst["industry"] = top["industry"].values
            else:
                lst["industry"] = "UNKNOWN"
            daily_lists[d] = lst
        return df, daily_lists
    except Exception as exc:
        _data_logger.error("面板转 V35 清单失败: %s", exc)
        return None


# ============================================================
# 真实交易信号 / 持仓加载 (C1/C2 Gap)
# ============================================================
AUDIT_LOG_PATH = str(data_others_path("data/audit_log.json"))


def load_real_signals() -> list[dict]:
    """从最新清单加载真实交易信号 (替代硬编码演示数据).

    优先读取 priority.json 中的标记股, 结合最新清单的预测数据.
    """
    try:
        priority_syms = load_priority_symbols()
        if not priority_syms:
            return []
        lst, _ = load_latest_list()
        if lst is None:
            # 没有清单, 用 priority 标记构造最小信号
            return [
                {
                    "time": "14:50",
                    "symbol": s,
                    "side": "buy",
                    "priority": "L4-形态",
                    "reason": "日内买入标记",
                    "price": 0.0,
                    "qty": 100,
                }
                for s in sorted(priority_syms)
            ]
        # 从清单中匹配 priority 股票
        matched = lst[lst["symbol"].isin(priority_syms)]
        signals = []
        for _, row in matched.iterrows():
            sym = row["symbol"]
            pred_ret = float(row.get("pred_ret_1d", 0))
            prob = float(row.get("prob_up", 0))
            side = "buy" if pred_ret > 0 else "sell"
            reason = f"prob={prob:.2f} pred_ret_1d={pred_ret:+.2%}"
            signals.append(
                {
                    "time": "14:50",
                    "symbol": sym,
                    "side": side,
                    "priority": "Pipeline-1",
                    "reason": reason,
                    "price": float(row.get("close", 0) or 0),
                    "qty": 100,
                }
            )
        return signals
    except Exception as exc:
        _data_logger.warning("加载真实信号失败: %s", exc)
        return []


def load_real_positions() -> list[dict]:
    """从 priority.json + 最新行情构造持仓列表 (近似).

    Returns:
        持仓列表 [{symbol, qty, available_qty, cost, current_price}].
    """
    try:
        priority_syms = load_priority_symbols()
        if not priority_syms:
            return []
        lst, _ = load_latest_list()
        positions = []
        for sym in sorted(priority_syms):
            if lst is not None and sym in lst["symbol"].values:
                row = lst[lst["symbol"] == sym].iloc[0]
                close = float(row.get("close", 100) or 100)
                cost = close * 0.98  # 近似成本
            else:
                close = 100.0
                cost = 98.0
            positions.append(
                {
                    "symbol": sym,
                    "qty": 100,
                    "available_qty": 0,  # T+1: 当日买入不可卖
                    "cost": round(cost, 2),
                    "current_price": round(close, 2),
                }
            )
        return positions
    except Exception as exc:
        _data_logger.warning("加载真实持仓失败: %s", exc)
        return []


def load_real_account() -> dict:
    """构造账户快照 (演示, 生产接入 Executor.get_account).

    Returns:
        {total_asset, available_cash, frozen}.
    """
    return {"total_asset": 1018000.0, "available_cash": 817400.0, "frozen": 0.0}


def load_audit_log(limit: int = 50) -> list[dict]:
    """读取审计日志 (append-only JSON).

    Returns:
        日志记录列表 (最新的 limit 条).
    """
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    try:
        with open(AUDIT_LOG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return data[-limit:]
        if isinstance(data, dict):
            return data.get("entries", [])[-limit:]
        return []
    except Exception:
        return []


def append_audit_log(entry: dict) -> None:
    """追加审计日志条目 (append-only, 铁律10).

    Args:
        entry: 日志条目 {时间, 操作, 代码, 方向, 价格, 数量, 结果, 备注}.
    """
    os.makedirs(os.path.dirname(AUDIT_LOG_PATH), exist_ok=True)
    log = load_audit_log()
    log.append(entry)
    try:
        with open(AUDIT_LOG_PATH, "w", encoding="utf-8") as fh:
            json.dump(log, fh, ensure_ascii=False, indent=1)
    except Exception as exc:
        _data_logger.error("写入审计日志失败: %s", exc)
