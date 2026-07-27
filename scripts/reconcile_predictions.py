# -*- coding: utf-8 -*-
"""
预测池实际收益回填 (每日收盘后运行)
========================================
从 data/lists/ 读历史清单 → 计算 T+1/3/5 实际收益 → 回填 prediction DB.

用法:
  python scripts/reconcile_predictions.py           # 回填所有未完成的日期
  python scripts/reconcile_predictions.py 20260724  # 仅回填指定日期
  python scripts/reconcile_predictions.py --all     # 从 OHLCV 数据库回填全部
"""

from __future__ import annotations

import logging
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.prediction_db import PredictionDB

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_ohlcv_panel(list_dir: str = "data/lists") -> pd.DataFrame | None:
    """从清单目录反推 symbol 列表, 尝试从历史库加载 OHLCV 数据.

    Returns:
        DataFrame with symbol, date, close, pre_close columns, or None if no data source.
    """
    # 尝试从 akshare 获取
    try:
        import akshare as ak

        symbols = set()
        for fname in os.listdir(list_dir):
            if fname.startswith("list_") and fname.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(list_dir, fname))
                symbols.update(df["symbol"].tolist())

        frames = []
        for sym in sorted(symbols):
            try:
                df = ak.stock_zh_a_hist(symbol=sym, period="daily", adjust="qfq")
                if len(df):
                    df["symbol"] = sym
                    df = df.rename(
                        columns={"日期": "date", "收盘": "close", "前收盘": "pre_close"}
                    )
                    df["date"] = pd.to_datetime(df["date"])
                    frames.append(df[["symbol", "date", "close", "pre_close"]])
            except Exception:
                logger.warning("无法获取 %s 历史数据", sym)
        if frames:
            return pd.concat(frames, ignore_index=True)
    except ImportError:
        pass
    return None


def reconcile_date(date_str: str, panel: pd.DataFrame, db: PredictionDB) -> int:
    """回填单个日期的实际收益."""
    stocks = db.get_run(date_str)
    if not stocks or not stocks["stocks"]:
        logger.info("%s: 无清单数据", date_str)
        return 0

    # 检查是否已回填
    outcomes = db.quality_summary(date_str)
    if outcomes.get("n", 0) > 0:
        logger.info("%s: 已回填 (%d 条)", date_str, outcomes["n"])
        return 0

    if panel is None:
        logger.warning("%s: 无 OHLCV 数据源, 跳过", date_str)
        return 0

    rows = []
    pred_date = pd.to_datetime(date_str)
    for s in stocks["stocks"]:
        sym = s["symbol"]
        sym_data = panel[panel["symbol"] == sym].sort_values("date")
        if len(sym_data) < 6:
            continue

        # 找到预测日之后 1/3/5 个交易日的收盘价
        future = sym_data[sym_data["date"] > pred_date]
        if len(future) < 5:
            continue

        # T+0 收盘 (预测日收盘, 作为买入价)
        pred_close = sym_data[sym_data["date"] == pred_date]
        if len(pred_close) == 0:
            # 取预测日最近一个交易日
            pred_close = sym_data[sym_data["date"] <= pred_date].iloc[-1:]
        if len(pred_close) == 0:
            continue
        entry = float(pred_close.iloc[0]["close"])

        for k, offset in ((1, 1), (3, 3), (5, 5)):
            if len(future) >= offset:
                exit_px = float(future.iloc[offset - 1]["close"])
                row = {"symbol": sym, f"actual_ret_{k}d": round(exit_px / entry - 1, 6)}
                rows.append(row)

    if not rows:
        return 0

    outcomes_df = pd.DataFrame(rows)
    # Merge same-symbol rows
    outcomes_df = outcomes_df.groupby("symbol").first().reset_index()
    return db.backfill_outcomes(date_str, outcomes_df)


def main():
    db = PredictionDB()
    list_dir = os.path.join("data", "lists")

    # Parse args
    target_date = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if re.match(r"^\d{8}$", arg):
            target_date = arg

    # Load panel once
    panel = load_ohlcv_panel(list_dir)

    if target_date:
        reconcile_date(target_date, panel, db)
        return

    # Scan all unmatured lists
    files = sorted(
        f
        for f in os.listdir(list_dir)
        if f.startswith("list_") and f.endswith(".parquet")
    )
    for fname in files:
        date_str = fname.replace("list_", "").replace(".parquet", "")
        reconcile_date(date_str, panel, db)

    # Print summary
    runs = db.list_runs(20)
    quality = db.all_quality(20)
    print(f"\n最近 {len(runs)} 次预测:")
    for r in runs:
        q = next((q for q in quality if q["date"] == r["date"]), {})
        dir_acc = q.get("direction_accuracy")
        dir_str = f"方向={dir_acc:.1%}" if dir_acc is not None else "未回填"
        print(f"  {r['date']}  {r['n_stocks']}只  {dir_str}")


if __name__ == "__main__":
    main()
