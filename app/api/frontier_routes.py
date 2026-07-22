# -*- coding: utf-8 -*-
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
    """最新 V3.5 清单 (真实优先, 演示兜底 + demo 标记)."""
    lst, date = ds.load_latest_list()
    demo = lst is None
    if demo:
        lst, date = ds.demo_list(), "DEMO"
    return {"date": date, "demo": demo, "schema_version": "1.0",
            "items": lst.to_dict("records")}


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
    return {"symbol": item.symbol, "watched": ds.toggle_watchlist(item.symbol, item.name)}


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


def _demo_panel_and_lists():
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2025-06-01", periods=180)
    frames = []
    for sym, ind in (("600519", "白酒"), ("601318", "保险"), ("600000", "银行")):
        close = 100 * np.cumprod(1 + rng.normal(0.001, 0.015, 180))
        open_ = close * (1 + rng.normal(0, 0.003, 180))
        frames.append(pd.DataFrame({
            "symbol": sym, "date": dates, "open": open_,
            "high": np.maximum(open_, close) * 1.01,
            "low": np.minimum(open_, close) * 0.99,
            "close": close, "pre_close": pd.Series(close).shift(1).fillna(close[0]),
            "board": "main", "industry": ind, "amount": 1e9}))
    panel = pd.concat(frames, ignore_index=True)
    rng2 = np.random.default_rng(3)
    lists = {d: pd.DataFrame({"symbol": g["symbol"].values,
                              "score": rng2.uniform(0, 1, len(g)),
                              "prob_up": 0.60,
                              "industry": g["industry"].values})
             for d, g in panel.groupby("date")}
    return panel, lists


@router.post("/backtest/run")
def run_backtest(req: BacktestRequest) -> dict:
    """V3.5 协议回测 (演示面板)."""
    panel, lists = _demo_panel_and_lists()
    proto = BacktestProtocol(top_n=req.top_n, max_hold_days=req.max_hold_days,
                             hard_stop=req.hard_stop,
                             trailing_drawdown=req.trailing_drawdown,
                             prob_exit=req.prob_exit)
    result = BacktestEngineV35(panel, proto).run(lists,
                                                 initial_capital=req.initial_capital)
    nav = result["nav_curve"].copy()
    nav["date"] = nav["date"].astype(str)
    trades = result["trades"].copy()
    if len(trades):
        trades["date"] = trades["date"].astype(str)
    return {"demo": True, "metrics": result["metrics"],
            "nav_curve": nav.to_dict("records"),
            "trades": trades.to_dict("records")}


class TuneRequest(BaseModel):
    params: list[str] = ["max_hold_days", "prob_exit"]
    top_k: int = 3


@router.post("/backtest/tune")
def run_tune(req: TuneRequest) -> dict:
    """参数调优: 网格搜索 + OOS 复验."""
    invalid = [p for p in req.params if p not in TUNABLE_BOUNDS]
    if invalid:
        raise HTTPException(400, f"非法参数: {invalid}")
    if len(req.params) > 4:
        raise HTTPException(400, "建议 ≤4 维 (控制组合数)")
    panel, lists = _demo_panel_and_lists()
    report = ParamTuner(panel, lists).grid_search(req.params, top_k=req.top_k)
    report["leaderboard"] = [(str(p), s) for p, s in report["leaderboard"]]
    return report


# ============================================================
# 规则参数 / 调参报告
# ============================================================
@router.get("/config/rules")
def get_rule_config() -> dict:
    """规则引擎 Config 当前值 + [TUNABLE] 边界."""
    cfg = Config()
    return {"tunable": {name: {"value": getattr(cfg, name), "bounds": list(b)}
                        for name, b in sorted(TUNABLE_BOUNDS.items())}}


@router.get("/tuning/report")
def get_tuning_report() -> dict:
    report = ds.load_tuning_report()
    if report is None:
        return {"exists": False}
    return {"exists": True, **report}
