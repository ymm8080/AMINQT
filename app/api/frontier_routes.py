"""
Frontier 前端数据 API (React SPA 后端)
============================================
只读数据端点: V3.5 清单 / 关注股 / 回测 / 调参报告 / 规则参数 / K线.
数据源: app/streamlit/data_service.py (真实数据优先, 演示兜底).
"""

from __future__ import annotations

import logging
from datetime import datetime

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
    """买卖信号: Pipeline2 V5.1 引擎生成 (买/卖纯函数 trigger).

    Pipeline1 提供候选池 + stop_price/pred_q50 → Pipeline2 V5.1 决策.
    """
    items = []

    # 1. 检查是否在 Pipeline1 清单中
    lst_df, _ = ds.load_latest_list()
    in_pool = False
    stock_info = {}
    if lst_df is not None and len(lst_df):
        row = lst_df[lst_df["symbol"] == symbol]
        if len(row):
            in_pool = True
            r = row.iloc[0]
            stock_info = {
                "pred_q50": float(r.get("pred_q50", 0.005)),
                "prob_up": float(r.get("prob_up", 0.5)),
                "score": float(r.get("score", 0)),
                "weight": float(r.get("weight", 0.1)),
                "board": str(r.get("board", "main")),
                "momentum": str(r.get("momentum", "medium")),
            }

    if in_pool:
        # 2. Pipeline2 V5.1 买入引擎
        try:
            ohlc = ds.fetch_real_ohlc(symbol, days=30)
            if ohlc is None:
                ohlc = ds.demo_ohlc(symbol, days=30)
            last_bar = ohlc.iloc[-1]
            prev_bar = ohlc.iloc[-2] if len(ohlc) > 1 else last_bar
            atr_pct = 0.02
            if len(ohlc) >= 15:
                tr = np.maximum(
                    ohlc["high"].values - ohlc["low"].values,
                    np.abs(ohlc["high"].values - np.roll(ohlc["close"].values, 1)),
                )
                atr_pct = float(np.mean(tr[-14:]) / last_bar["close"])
            stop_price = last_bar["close"] * (1 + max(-0.04, -1.5 * atr_pct))

            # Pipeline2 买入信号
            from app.intraday.v51.buy_engine import BuyContext, trigger as buy_trigger
            from app.intraday.v51.buy_engine import Bar as BuyBar

            ctx = BuyContext(
                symbol=symbol,
                t="09:35",
                price=float(last_bar["close"]),
                pre_close=float(prev_bar["close"]),
                pred_q50=stock_info["pred_q50"],
                atr_pct=atr_pct,
                stop_price=stop_price,
                adv_20d=float(ohlc["volume"].tail(20).mean() * last_bar["close"]),
                order_value=100000,
                bar_amount=float(last_bar["volume"] * last_bar["close"])
                if "volume" in last_bar.index
                else 2e6,
            )
            bars = tuple(
                BuyBar(
                    t=str(row_date)[:16]
                    if hasattr(row_date, "__str__")
                    else str(row_date),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 1e6)),
                    amount=float(row["close"] * row.get("volume", 1e6)),
                )
                for row_date, row in ohlc.tail(5).iterrows()
            )
            buy_signal = buy_trigger(ctx, bars)
            if buy_signal["pass"]:
                items.append(
                    {
                        "time": "09:35",
                        "symbol": symbol,
                        "side": "buy",
                        "price": round(float(last_bar["close"]), 2),
                        "priority": f"P2-{buy_signal.get('positive', 'B1')}",
                        "reason": f"V5.1买入·{buy_signal.get('positive', 'B1')} pred_q50={stock_info['pred_q50']:+.3f}",
                        "qty": 100,
                        "executed": False,
                    }
                )
            else:
                veto_str = ", ".join(buy_signal.get("vetoes", ["未知"])) or "无否决"
                items.append(
                    {
                        "time": "09:35",
                        "symbol": symbol,
                        "side": "buy",
                        "price": 0,
                        "priority": "P2-BLOCKED",
                        "reason": f"V5.1否决: {veto_str}",
                        "qty": 0,
                        "executed": False,
                    }
                )
        except Exception:
            logger.warning("Pipeline2 buy_engine 失败", exc_info=True)

        # 3. Pipeline2 V5.1 卖出信号
        try:
            from app.intraday.v51.sell_engine import (
                SellContext,
                trigger as sell_trigger,
            )
            from app.intraday.v51.position import Position

            intraday = ds.fetch_real_intraday(symbol)
            if intraday is None:
                intraday = ds.demo_intraday(symbol)
            latest_px = (
                float(intraday["price"].iloc[-1])
                if len(intraday)
                else float(last_bar["close"])
            )
            ld_price = round(float(prev_bar["close"]) * 0.9, 2)

            sctx = SellContext(
                t="14:50",
                price=latest_px,
                limit_down_price=ld_price,
                limit_up_price=round(float(prev_bar["close"]) * 1.1, 2),
                atr_pct=atr_pct,
            )
            pos = Position(
                symbol=symbol,
                total_qty=100,
                sellable_qty=100,
                entry_price=float(last_bar["close"]),
                entry_date=str(ohlc.index[-1])[:10]
                if hasattr(ohlc.index[-1], "__str__")
                else str(ohlc.iloc[-1]["date"])[:10],
                hold_days=1,
                max_price_since_entry=float(last_bar["high"]),
                stop_price=stop_price,
            )
            sell_signal = sell_trigger(sctx, pos)
            if sell_signal["action"] != "HOLD":
                items.append(
                    {
                        "time": sell_signal.get("time", "14:50"),
                        "symbol": symbol,
                        "side": "sell",
                        "price": round(latest_px, 2),
                        "priority": sell_signal.get("rule", "S0"),
                        "reason": sell_signal.get("reason", sell_signal["action"]),
                        "qty": sell_signal.get("qty", 0),
                        "executed": False,
                    }
                )
        except Exception:
            logger.warning("Pipeline2 sell_engine 失败", exc_info=True)

    # 4. Fallback: 不在清单中 → demo
    if not in_pool:
        items = [
            {
                "time": "09:44",
                "symbol": symbol,
                "side": "buy",
                "priority": "L4-DEMO",
                "reason": "不在Pipeline1清单中, 演示数据",
                "qty": 0,
                "price": 0,
                "executed": False,
            },
        ]

    return {"demo": not in_pool, "symbol": symbol, "items": items, "pipeline2": True}


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
    """K线数据: 真实优先 (akshare), 失败回退 demo."""
    try:
        df = ds.fetch_real_ohlc(symbol, days=min(days, 500))
    except Exception:
        logger.warning("fetch_real_ohlc 网络异常: %s", symbol, exc_info=True)
        df = None
    demo = df is None
    if demo:
        df = ds.demo_ohlc(symbol, days=min(days, 500))
    df["date"] = df["date"].astype(str)
    return {"symbol": symbol, "demo": demo, "items": df.to_dict("records")}


@router.get("/intraday/{symbol}")
def get_intraday(symbol: str) -> dict:
    """分时数据: 真实优先 (akshare 5min), 失败回退 demo."""
    try:
        df = ds.fetch_real_intraday(symbol)
    except Exception:
        logger.warning("fetch_real_intraday 网络异常: %s", symbol, exc_info=True)
        df = None
    demo = df is None
    if demo:
        df = ds.demo_intraday(symbol)
    return {"symbol": symbol, "demo": demo, "items": df.to_dict("records")}


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
    start_date: str | None = None  # 'YYYY-MM-DD', 优先级高于 window_days
    end_date: str | None = None  # 'YYYY-MM-DD'
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


# ============================================================
# 预测池数据库 (P25.5)
# ============================================================
@router.get("/prediction/runs")
def get_prediction_runs(limit: int = 60) -> dict:
    """预测运行日期列表."""
    try:
        from app.pipeline1.prediction_db import PredictionDB

        return {"runs": PredictionDB().list_runs(limit)}
    except Exception:
        return {"runs": []}


@router.get("/prediction/run/{date}")
def get_prediction_run(date: str) -> dict:
    """单个预测日期完整记录 (stocks + outcomes)."""
    try:
        from app.pipeline1.prediction_db import PredictionDB

        run = PredictionDB().get_run(date)
    except Exception:
        run = None
    if run is None:
        raise HTTPException(404, f"预测记录不存在: {date}")
    return run


@router.get("/prediction/quality")
def get_prediction_quality(limit: int = 60) -> dict:
    """所有日期预测质量汇总 (趋势)."""
    try:
        from app.pipeline1.prediction_db import PredictionDB

        return {"items": PredictionDB().all_quality(limit)}
    except Exception:
        return {"items": []}


# ============================================================
# Pipeline 触发 (每日数据追加 + 预测)
# ============================================================
PANEL_PATHS = [
    "data/panel_full_enriched_v3.parquet",
    "data/panel_full_enriched_v2.parquet",
]
MODEL_DIR = "models/pipeline1"


class AppendDailyRequest(BaseModel):
    trade_date: str | None = None  # YYYYMMDD, None=今天
    market_state: str = "range"  # bull / bear / range
    save_panel: bool = True  # 追加后保存面板 (WORM 备份)


@router.get("/pipeline/status")
def get_pipeline_status() -> dict:
    """面板 + 模型 + 清单状态快照."""
    import os

    # 面板状态
    panel_info = {"exists": False}
    for path in PANEL_PATHS:
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path, columns=["symbol", "date"])
                panel_info = {
                    "exists": True,
                    "path": path,
                    "n_stocks": int(df["symbol"].nunique()),
                    "n_rows": len(df),
                    "last_date": str(df["date"].max().strftime("%Y-%m-%d")),
                    "first_date": str(df["date"].min().strftime("%Y-%m-%d")),
                    "size_mb": round(os.path.getsize(path) / 1e6, 1),
                }
                break
            except Exception as exc:
                panel_info = {"exists": False, "error": str(exc)}
                break

    # 模型状态
    model_info: dict[str, dict] = {}
    if os.path.isdir(MODEL_DIR):
        for board in ("main", "dual"):
            pkls = sorted(
                f
                for f in os.listdir(MODEL_DIR)
                if f.startswith(f"{board}_") and f.endswith(".pkl")
            )
            if pkls:
                latest = pkls[-1]
                model_info[board] = {
                    "file": latest,
                    "path": os.path.join(MODEL_DIR, latest),
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(MODEL_DIR, latest))
                    ).strftime("%Y-%m-%d %H:%M"),
                }

    # 最新清单
    list_dates = ds.list_available_dates()

    return {
        "panel": panel_info,
        "models": model_info,
        "latest_list_date": list_dates[0] if list_dates else None,
        "list_count": len(list_dates),
    }


@router.post("/pipeline/append-daily")
def append_daily_and_predict(req: AppendDailyRequest) -> dict:
    """触发每日数据追加到 V3 面板 + 推理预测.

    流程:
      1. 加载 panel_full_enriched_v3.parquet (历史面板)
      2. 追加当日 OHLCV + margin/northbound/lhb
      3. (可选) 保存更新后的面板 (WORM: 先备份)
      4. 加载最新模型包 → 推理 → 清单生成 → 持久化
      5. 返回结果摘要
    """
    import os
    import shutil

    from app.pipeline1.daily_pipeline import DailySelectionPipeline
    from app.pipeline1.data_supply import DataSupplyChain, DataSupplyError
    from app.pipeline1.panel_builder import enrich_cyq
    from app.pipeline1.predict_runner import find_bundles

    trade_date = req.trade_date or datetime.now().strftime("%Y%m%d")
    log_lines: list[str] = []

    def _log(msg: str):
        logger.info("[pipeline/append-daily] %s", msg)
        log_lines.append(f"{datetime.now():%H:%M:%S} {msg}")

    # 1. 定位历史面板
    panel_path = None
    for p in PANEL_PATHS:
        if os.path.exists(p):
            panel_path = p
            break
    if panel_path is None:
        raise HTTPException(500, "无可用历史面板 (panel_full_enriched_v3.parquet)")

    _log(f"加载面板: {panel_path}")
    panel = pd.read_parquet(panel_path)
    _log(
        f"面板: {panel['symbol'].nunique()} stocks, {len(panel)} rows, "
        f"{panel['date'].min()} ~ {panel['date'].max()}"
    )

    # 去除当日数据 (避免重复)
    panel = panel[panel["date"] < pd.to_datetime(trade_date)]

    # 2. CYQ enrich
    try:
        panel = enrich_cyq(panel, cyq_cache="data/cyq_panel.parquet")
        _log("CYQ enrich 完成")
    except Exception as exc:
        _log(f"CYQ enrich 跳过: {exc}")

    # 3. 追加当日数据
    supply = DataSupplyChain()
    try:
        panel = supply.append_today_to_panel(
            panel,
            trade_date=trade_date,
            sources=["ohlcv", "margin", "northbound", "lhb"],
        )
        _log(f"当日数据追加完成: {panel['symbol'].nunique()} stocks, {len(panel)} rows")
    except DataSupplyError as exc:
        _log(f"数据追加失败: {exc}")
        return {
            "success": False,
            "trade_date": trade_date,
            "error": str(exc),
            "logs": log_lines,
        }

    # 4. 保存面板 (WORM: 先备份)
    if req.save_panel:
        backup_path = panel_path.replace(
            ".parquet", f"_backup_{datetime.now():%Y%m%d_%H%M%S}.parquet"
        )
        shutil.copy2(panel_path, backup_path)
        _log(f"面板备份: {backup_path}")
        panel.to_parquet(panel_path, index=False)
        _log(f"面板已保存: {panel_path}")

    # 5. 推理预测
    bundles = find_bundles(model_dir=MODEL_DIR)
    if not bundles:
        raise HTTPException(500, "无可用模型包, 请先训练 (scripts/train_pipeline1.py)")

    _log(f"模型包: { {k: os.path.basename(v) for k, v in bundles.items()} }")

    pipe = DailySelectionPipeline(
        supply=supply,
        bundle_paths=bundles,
    )
    try:
        result = pipe.run(trade_date, panel=panel, market_state=req.market_state)
    except Exception as exc:
        _log(f"推理失败: {exc}")
        return {
            "success": False,
            "trade_date": trade_date,
            "error": str(exc),
            "logs": log_lines,
        }

    lst = result.get("list")
    n = 0 if lst is None or result.get("empty") else len(lst)
    _log(f"清单生成: mode={result.get('mode')}, {n} stocks")

    # 返回摘要
    list_summary = []
    if lst is not None and n > 0:
        for _, row in lst.head(20).iterrows():
            list_summary.append(
                {
                    "symbol": str(row.get("symbol", "")),
                    "board": str(row.get("board", "")),
                    "prob_up": round(float(row.get("prob_up", 0)), 4),
                    "pred_ret_1d": round(float(row.get("pred_ret_1d", 0)), 4),
                    "pred_ret_3d": round(float(row.get("pred_ret_3d", 0)), 4),
                    "pred_ret_5d": round(float(row.get("pred_ret_5d", 0)), 4),
                    "score": round(float(row.get("score", 0)), 4),
                    "weight": round(float(row.get("weight", 0)), 4),
                }
            )

    return {
        "success": True,
        "trade_date": trade_date,
        "mode": result.get("mode"),
        "n_stocks": n,
        "empty": result.get("empty", False),
        "cap_position": result.get("cap_position", 0.0),
        "valve_state": result.get("valve_state"),
        "list_preview": list_summary,
        "logs": log_lines,
    }
