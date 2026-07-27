"""
Frontier 前端数据 API (React SPA 后端)
============================================
只读数据端点: V3.5 清单 / 关注股 / 回测 / 调参报告 / 规则参数 / K线.
数据源: app/streamlit/data_service.py (真实数据优先, 演示兜底).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol
from app.pipeline1.param_tuner import ParamTuner
from app.rules.config import TUNABLE_BOUNDS, Config
from app.streamlit import data_service as ds

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/frontier", tags=["frontier"])


# ============================================================
# 清单
# ============================================================
@router.get("/list/latest")
def get_latest_list() -> dict:
    """最新 V3.8 清单 (真实优先, 演示兜底 + demo 标记)."""
    lst, date = ds.load_latest_list()
    demo = lst is None
    if demo:
        lst, date = ds.demo_list(), "DEMO"
    return {
        "date": date,
        "demo": demo,
        "schema_version": "1.2",
        "items": lst.to_dict("records"),
    }


@router.get("/list/dates")
def get_list_dates() -> dict:
    return {"dates": ds.list_available_dates()}


@router.get("/list/{trade_date}")
def get_list(trade_date: str) -> dict:
    lst = ds.load_list(trade_date)
    if lst is None:
        raise HTTPException(404, f"清单不存在: {trade_date}")
    return {"date": trade_date, "demo": False, "items": lst.to_dict("records")}


# ============================================================
# 板块行情
# ============================================================
@router.get("/sectors")
def get_sectors() -> dict:
    """板块当日涨跌幅 + 日内走势 (演示数据)."""
    df = ds.demo_sector_changes()
    records = df.to_dict("records")
    for rec in records:
        sector = rec["板块"]
        intra = ds.demo_sector_intraday(sector)
        p0 = intra["price"].iloc[0]
        rec["intraday"] = ((intra["price"] / p0) - 1).tolist()
    return {"demo": True, "items": records}


@router.get("/signals/{symbol}")
def get_signals(symbol: str) -> dict:
    """演示买卖信号 (时间、价格、量、方向、原因); 价格取该标的日内最近时刻."""

    def _minutes(t: str) -> int:
        h, m = map(int, t.split(":"))
        return h * 60 + m

    def _price_at(sym: str, target: str) -> float:
        df = ds.demo_intraday(sym)
        tm = _minutes(target)
        df["_tm"] = df["time"].apply(_minutes)
        idx = (df["_tm"] - tm).abs().idxmin()
        return float(df.loc[idx, "price"])

    raw = [
        {
            "time": "09:44",
            "symbol": "600519",
            "side": "buy",
            "priority": "L4-形态",
            "reason": "下探后低峰确认回升",
            "qty": 100,
            "executed": True,
        },
        {
            "time": "10:12",
            "symbol": "300750",
            "side": "sell",
            "priority": "P7",
            "reason": "涨7%+高换手减半",
            "qty": 200,
            "executed": False,
        },
        {
            "time": "13:05",
            "symbol": "601318",
            "side": "sell",
            "priority": "P10",
            "reason": "浮盈≥20%人工复核",
            "qty": 500,
            "executed": True,
        },
    ]
    items = [
        {**s, "price": round(_price_at(s["symbol"], s["time"]), 2)}
        for s in raw
        if s["symbol"] == symbol
    ]
    return {"demo": True, "symbol": symbol, "items": items}


# ============================================================
# 日内买入候选 (priority)
# ============================================================
@router.get("/priority")
def get_priority() -> dict:
    return {"symbols": sorted(ds.load_priority_symbols())}


@router.post("/priority/toggle")
def toggle_priority(item: WatchItem) -> dict:
    return {
        "symbol": item.symbol,
        "priority": ds.toggle_priority(item.symbol),
    }


# ============================================================
# K线 / 分时
# ============================================================
@router.get("/ohlc/{symbol}")
def get_ohlc(symbol: str, days: int = 120) -> dict:
    """K线数据 (生产: 历史库; 当前演示合成)."""
    df = ds.demo_ohlc(symbol, days=min(days, 500))
    df["date"] = df["date"].astype(str)
    return {"symbol": symbol, "demo": True, "items": df.to_dict("records")}


@router.get("/intraday/{symbol}")
def get_intraday(symbol: str) -> dict:
    df = ds.demo_intraday(symbol)
    return {"symbol": symbol, "demo": True, "items": df.to_dict("records")}


# ============================================================
# 关注股
# ============================================================
class WatchItem(BaseModel):
    symbol: str
    name: str = ""


@router.get("/watchlist")
def get_watchlist() -> dict:
    return {"items": ds.load_watchlist()}


@router.post("/watchlist/toggle")
def toggle_watch(item: WatchItem) -> dict:
    return {
        "symbol": item.symbol,
        "watched": ds.toggle_watchlist(item.symbol, item.name),
    }


# ============================================================
# 回测
# ============================================================
class BacktestRequest(BaseModel):
    top_n: int = 15
    max_hold_days: int = 3
    hard_stop: float = -0.04
    trailing_drawdown: float = 0.04
    prob_exit: float = 0.50
    initial_capital: float = 1_000_000
    window_days: int = 180
    start_date: str | None = None   # 'YYYY-MM-DD', 优先级高于 window_days
    end_date: str | None = None     # 'YYYY-MM-DD'
    objective: str = "net_excess_annual"
    max_dd_limit: float | None = None


def _demo_panel_and_lists(
    window_days: int = 180,
    seed: int = 9,
    start_date: str | None = None,
    end_date: str | None = None,
):
    rng = np.random.default_rng(seed)
    if start_date and end_date:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        dates = pd.bdate_range(start=start, end=end)
        n_days = len(dates)
    else:
        end = pd.Timestamp.today().normalize()
        dates = pd.bdate_range(end=end, periods=window_days)
        n_days = window_days
    frames = []
    for sym, ind in (("600519", "白酒"), ("601318", "保险"), ("600000", "银行")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, n_days))
        open_ = close * (1 + rng.normal(0, 0.003, n_days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.01,
                    "low": np.minimum(open_, close) * 0.99,
                    "close": close,
                    "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                    "board": "main",
                    "industry": ind,
                    "amount": 1e9,
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    rng2 = np.random.default_rng(3)
    lists = {
        d: pd.DataFrame(
            {
                "symbol": g["symbol"].values,
                "score": rng2.uniform(0, 1, len(g)),
                "prob_up": 0.60,
                "industry": g["industry"].values,
            }
        )
        for d, g in panel.groupby("date")
    }
    return panel, lists


@router.post("/backtest/run")
def run_backtest(req: BacktestRequest) -> dict:
    """V3.5 协议回测 (演示面板)."""
    panel, lists = _demo_panel_and_lists(
        req.window_days, start_date=req.start_date, end_date=req.end_date
    )
    proto = BacktestProtocol(
        top_n=req.top_n,
        max_hold_days=req.max_hold_days,
        hard_stop=req.hard_stop,
        trailing_drawdown=req.trailing_drawdown,
        prob_exit=req.prob_exit,
    )
    result = BacktestEngineV35(panel, proto).run(
        lists, initial_capital=req.initial_capital
    )
    nav = result["nav_curve"].copy()
    nav["date"] = nav["date"].astype(str)
    trades = result["trades"].copy()
    if len(trades):
        trades["date"] = trades["date"].astype(str)
    return {
        "demo": True,
        "metrics": result["metrics"],
        "nav_curve": nav.to_dict("records"),
        "trades": trades.to_dict("records"),
    }


class TuneRequest(BaseModel):
    params: list[str] = ["max_hold_days", "prob_exit"]
    top_k: int = 3
    objective: str = "net_excess_annual"
    max_dd_limit: float | None = None
    ranges: dict[str, tuple[float, float, float]] | None = None


@router.post("/backtest/tune")
def run_tune(req: TuneRequest) -> dict:
    """参数调优: 网格搜索 + OOS 复验."""
    invalid = [p for p in req.params if p not in TUNABLE_BOUNDS]
    if invalid:
        raise HTTPException(400, f"非法参数: {invalid}")
    if len(req.params) > 4:
        raise HTTPException(400, "建议 ≤4 维 (控制组合数)")
    panel, lists = _demo_panel_and_lists()

    original_bounds = dict(TUNABLE_BOUNDS)
    if req.ranges:
        for name, (lo, hi, step) in req.ranges.items():
            if name in TUNABLE_BOUNDS:
                TUNABLE_BOUNDS[name] = (lo, hi, step)
    try:
        report = ParamTuner(panel, lists).grid_search(
            req.params,
            top_k=req.top_k,
            objective=req.objective,
            max_dd_limit=req.max_dd_limit,
        )
    finally:
        TUNABLE_BOUNDS.clear()
        TUNABLE_BOUNDS.update(original_bounds)

    report["leaderboard"] = [(str(p), s) for p, s in report["leaderboard"]]
    return report


# ============================================================
# 规则参数 / 调参报告
# ============================================================
@router.get("/config/rules")
def get_rule_config() -> dict:
    """规则引擎 Config 当前值 + [TUNABLE] 边界."""
    cfg = Config()
    return {
        "tunable": {
            name: {"value": getattr(cfg, name), "bounds": list(b)}
            for name, b in sorted(TUNABLE_BOUNDS.items())
        }
    }


@router.get("/tuning/report")
def get_tuning_report() -> dict:
    report = ds.load_tuning_report()
    if report is None:
        return {"exists": False}
    return {"exists": True, **report}


# ============================================================
# 预测质量 (P25)
# ============================================================
@router.get("/forecast/quality")
def get_forecast_quality() -> dict:
    """返回最新预测质量报告 (MAE/BIAS/方向准确率/分桶/红灯).

    优先读 data/forecast_accuracy/ 真实数据; 无数据时返回 demo 占位.
    """
    import json
    import os

    acc_dir = os.path.join("data", "forecast_accuracy")
    if os.path.isdir(acc_dir):
        files = sorted(
            [
                f
                for f in os.listdir(acc_dir)
                if f.startswith("accuracy_") and f.endswith(".json")
            ],
            reverse=True,
        )
        if files:
            with open(os.path.join(acc_dir, files[0]), "r", encoding="utf-8") as fh:
                report = json.load(fh)
            latest_1d = report.get("horizons", {}).get("1", {})
            return {
                "exists": True,
                "demo": False,
                "date": report.get("forecast_date", ""),
                "mae_1d": latest_1d.get("mae_1d"),
                "bias_1d": latest_1d.get("bias_1d"),
                "direction_accuracy": latest_1d.get("direction_accuracy"),
                "n_samples": latest_1d.get("n_samples"),
                "bias_big_up": latest_1d.get("bias_big_up"),
                "bias_small_up": latest_1d.get("bias_small_up"),
                "bias_small_down": latest_1d.get("bias_small_down"),
                "bias_big_down": latest_1d.get("bias_big_down"),
            }

    # 无真实数据 — 返回 demo 占位
    return {
        "exists": False,
        "demo": True,
        "mae_1d": None,
        "bias_1d": None,
        "direction_accuracy": None,
        "n_samples": 0,
        "bias_big_up": None,
        "bias_small_up": None,
        "bias_small_down": None,
        "bias_big_down": None,
    }
