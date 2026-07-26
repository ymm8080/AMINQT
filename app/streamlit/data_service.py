"""
看板数据服务层 (P10) — 纯函数, 可测试
==========================================
数据源优先级: 真实产出 (data/lists/*.parquet, data/tuning_report.json) >
本地缓存 > 合成演示数据 (DEMO_MODE 标记, 页面顶部显著提示).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

LIST_DIR = "data/lists"
TUNING_REPORT = "data/tuning_report.json"
WATCHLIST_PATH = "data/watchlist.json"
PRIORITY_PATH = "data/priority.json"

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
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
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


def apply_priority_tags(df: pd.DataFrame) -> pd.DataFrame:
    """为清单打重点股标签: 已保存 + Pipeline1 推荐买入."""
    if df is None or df.empty or "symbol" not in df.columns:
        return df
    df = df.copy()
    saved = load_priority_symbols()
    auto = pipeline_buy_candidates(df)
    df["priority"] = df["symbol"].isin(saved | auto)
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
) -> tuple[pd.DataFrame | None, str | None]:
    """加载最新清单 → (df, date); 无清单返回 (None, None)."""
    dates = list_available_dates(list_dir)
    if not dates:
        return None, None
    df = load_list(dates[0], list_dir)
    if df is not None:
        df = apply_priority_tags(df)
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
            "pred_ret_1d": rng.uniform(-0.02, 0.05, n),
            "pred_ret_3d": rng.uniform(-0.03, 0.09, n),
            "pred_ret_5d": rng.uniform(-0.04, 0.12, n),
            "prob_up": np.round(rng.uniform(0.42, 0.62, n), 3),
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
            "schema_version": "1.2",
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
