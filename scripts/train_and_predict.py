# -*- coding: utf-8 -*-
"""
全 A 股训练 + 预测 (端到端)
==============================
1. 拉取全 A 股 1.5 年日线 → panel.parquet
2. 训练 (清洗→特征→标签→IC筛选→双轨训练) → models/pipeline1/
3. 预测今日候选 → data/lists/list_YYYYMMDD.parquet

用法:
  python scripts/train_and_predict.py              # 完整流程
  python scripts/train_and_predict.py --fetch-only # 只拉数据
  python scripts/train_and_predict.py --train-only # 只训练(用已有panel)
  python scripts/train_and_predict.py --predict-only # 只预测
  python scripts/train_and_predict.py --stocks 500  # 只拉500只测试
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_predict")

MODEL_DIR = "models/pipeline1"
PANEL_PATH = "data/panel_18m.parquet"
LIST_DIR = "data/lists"
CACHE_DIR = "data/stock_cache"
YEARS = 1.5
MIN_DAYS = 180  # 最少交易日


# ============================================================
# 阶段 1: 全 A 数据拉取
# ============================================================
def fetch_all_stocks(limit: int = 0) -> pd.DataFrame | None:
    """拉取全 A 股 1.5 年日线 → 本地 parquet (支持断点续拉).

    Args:
        limit: 0=全量, N=只拉前 N 只.

    Returns:
        panel DataFrame or None on failure.
    """
    try:
        import akshare as ak
    except ImportError:
        logger.error("akshare not installed")
        return None

    os.makedirs(CACHE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=int(365 * YEARS + 60))).strftime("%Y%m%d")

    # 全 A 列表
    logger.info("Fetching stock list...")
    all_symbols = []
    try:
        stock_list = ak.stock_zh_a_spot_em()
        all_symbols = stock_list["代码"].tolist()
        logger.info("Got %d stocks from akshare", len(all_symbols))
    except Exception:
        logger.warning("akshare stock list failed, trying backup...")

    # Backup: 中证1000 + 沪深300 成分股
    if not all_symbols:
        try:
            for index_name, index_code in [("中证1000", "000852"), ("沪深300", "000300")]:
                try:
                    df = ak.index_stock_cons(symbol=index_code)
                    symbols = df["品种代码"].tolist() if "品种代码" in df.columns else df["成分券代码"].tolist()
                    all_symbols.extend(symbols)
                    logger.info("Got %d stocks from %s", len(symbols), index_name)
                except Exception:
                    pass
        except Exception:
            pass

    # Last resort: local CSI 1000 proxy list
    if not all_symbols:
        import json
        proxy_path = os.path.join("data", "csi1000_stocks.json")
        if os.path.exists(proxy_path):
            with open(proxy_path) as f:
                all_symbols = json.load(f).get("symbols", [])
            logger.warning("Using proxy stock list: %d symbols", len(all_symbols))
        else:
            # Bare minimum: top 500 liquid stocks
            all_symbols = [f"60{i:04d}" for i in range(0, 500)] + [f"00{i:04d}" for i in range(1, 501)]
            logger.warning("Using generated stock list: %d symbols", len(all_symbols))

    if not all_symbols:
        logger.error("Cannot get stock list from any source")
        return None

    if limit and limit > 0:
        all_symbols = all_symbols[:limit]
    total = len(all_symbols)
    logger.info("Total stocks: %d (limit=%d)", total, limit or total)

    frames = []
    fetched = set()
    errors = 0

    for i, sym in enumerate(all_symbols):
        # 缓存命中
        cache_file = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if len(df) >= MIN_DAYS:
                    frames.append(df)
                    fetched.add(sym)
                    continue
            except Exception:
                pass

        # 拉取 (带重试)
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=sym, period="daily", start_date=start_date,
                    end_date=today, adjust="qfq", timeout=30,
                )
                if df is not None and len(df) > 0:
                    break
            except Exception:
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                continue
        if df is None or len(df) == 0:
            errors += 1
            if errors <= 5:
                logger.warning("Fetch failed (3 retries): %s", sym)
            time.sleep(0.05)
            continue

        if df is None or len(df) < MIN_DAYS:
            time.sleep(0.05)
            continue

        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume",
            "成交额": "amount", "换手率": "turnover_rate",
        })
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "turnover_rate" in df.columns:
            df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce") / 100
        else:
            df["turnover_rate"] = 0.02

        df["symbol"] = sym
        df = df.dropna(subset=["close"]).sort_values("date")

        # 补全必需列
        df["pre_close"] = df["close"].shift(1).fillna(df["close"])
        df["close_hfq"] = df["close"]
        df["board"] = (
            "STAR" if sym.startswith("688") else
            "GEM" if sym.startswith(("300", "301")) else "main"
        )
        df["industry"] = "ALL"
        df["is_st"] = False
        df["is_suspended"] = False
        df["is_one_word_limit"] = False
        df["name"] = sym
        df["list_days"] = range(1, len(df) + 1)
        if "amount" not in df.columns or df["amount"].isna().all():
            df["amount"] = df["volume"] * df["close"]
        df["amount"] = df["amount"].fillna(df["volume"] * df["close"])

        # 缓存
        try:
            df.to_parquet(cache_file, index=False)
        except Exception:
            pass

        frames.append(df)
        fetched.add(sym)

        # 进度
        if (i + 1) % 100 == 0:
            logger.info("Fetch: %d/%d (cached=%d errors=%d)", i + 1, total,
                        sum(1 for s in all_symbols[:i+1] if os.path.exists(os.path.join(CACHE_DIR, f"{s}.parquet"))), errors)
        time.sleep(0.15)  # 反限流

    logger.info("Fetch done: %d/%d stocks, %d errors", len(frames), total, errors)

    if not frames:
        return None

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet(PANEL_PATH, index=False)
    logger.info("Panel saved: %s (%d stocks, %d rows)", PANEL_PATH,
                panel["symbol"].nunique(), len(panel))
    return panel


# ============================================================
# 阶段 2: 训练
# ============================================================
def run_training(panel: pd.DataFrame, tag: str | None = None) -> dict:
    """训练双板块模型."""
    from app.pipeline1.train_runner import run_training as train

    if tag is None:
        from datetime import date
        iso = date.today().isocalendar()
        tag = f"{iso[0]}W{iso[1]:02d}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info("Starting training: tag=%s stocks=%d rows=%d",
                tag, panel["symbol"].nunique(), len(panel))

    t0 = time.time()
    results = train(
        panel, tag=tag, model_dir=MODEL_DIR,
        use_ic_screen=True,
    )
    elapsed = time.time() - t0

    for board, res in results.items():
        oos_1d = res["oos"]["ics"].get("1d_reg", 0)
        logger.info("[%s] model=%s OOS_IC(1d)=%.4f switched=%s time=%.0fs",
                    board, os.path.basename(res["path"]),
                    oos_1d, res["switched"], elapsed)
    return results


# ============================================================
# 阶段 3: 预测
# ============================================================
def run_prediction(trade_date: str | None = None) -> pd.DataFrame | None:
    """用最新模型预测当日候选."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.list_generator import ListGenerator, MarketEnv
    from app.pipeline1.predict_runner import find_bundles
    from app.pipeline1.predictor import V35Predictor

    if trade_date is None:
        trade_date = datetime.now().strftime("%Y%m%d")

    # 加载面板
    if not os.path.exists(PANEL_PATH):
        logger.error("Panel not found: %s (run --fetch first)", PANEL_PATH)
        return None
    panel = pd.read_parquet(PANEL_PATH)
    logger.info("Panel: %d stocks, %d rows", panel["symbol"].nunique(), len(panel))

    # 模型
    bundles = find_bundles(model_dir=MODEL_DIR)
    if "main" not in bundles:
        logger.error("No model found. Run --train first.")
        return None
    logger.info("Model: %s", {k: os.path.basename(v) for k, v in bundles.items()})

    # 预测范围: 中证1000成分股 (用户指定)
    logger.info("Filtering to CSI 1000 constituents for prediction...")
    try:
        import akshare as ak
        csi1000 = ak.index_stock_cons(symbol="000852")
        csi1000_syms = set(
            csi1000["品种代码"].tolist() if "品种代码" in csi1000.columns
            else csi1000["成分券代码"].tolist()
        )
        panel = panel[panel["symbol"].isin(csi1000_syms)]
        logger.info("CSI 1000 filter: %d stocks remaining", panel["symbol"].nunique())
    except Exception:
        logger.warning("CSI 1000 filter failed, using all stocks in panel")

    # 推理端清洗
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info("Cleaning: main=%d dual=%d valve=%s", len(main_df), len(dual_df), valve)
    if valve == "empty":
        logger.error("Safety valve triggered: empty list")
        return pd.DataFrame()

    # 特征 + 推理
    features = FeatureEngineV35()
    predictor = V35Predictor(bundles)
    frames = []
    for board, df in (("main", main_df), ("dual", dual_df)):
        if len(df) == 0:
            continue
        feats = features.build(df)
        latest = df["date"].max()
        today_feat = feats[feats["symbol"].isin(
            set(df[df["date"] == latest]["symbol"])
        )]
        if len(today_feat) == 0:
            continue
        pred = predictor.predict(today_feat, board)
        pred["board"] = board
        frames.append(pred)

    if not frames:
        logger.warning("No candidates after prediction")
        return pd.DataFrame()

    candidates = pd.concat(frames, ignore_index=True)
    logger.info("Candidates: %d", len(candidates))

    # 清单
    lister = ListGenerator()
    result = lister.emit(candidates, env=MarketEnv(), market_state="range")
    logger.info("List: %d stocks (empty=%s)", len(result["list"]), result["empty"])

    if not result["empty"] and len(result["list"]):
        os.makedirs(LIST_DIR, exist_ok=True)
        path = os.path.join(LIST_DIR, f"list_{trade_date}.parquet")
        result["list"].to_parquet(path, index=False)
        logger.info("Saved: %s", path)

        # sync priority
        import json
        pq_path = os.path.join("data", "priority.json")
        existing = set()
        if os.path.exists(pq_path):
            with open(pq_path, encoding="utf-8") as f:
                existing = set(json.load(f).get("symbols", []))
        merged = sorted(existing | set(result["list"]["symbol"].tolist()))
        with open(pq_path, "w", encoding="utf-8") as f:
            json.dump({"symbols": merged}, f, ensure_ascii=False, indent=2)

        # prediction DB
        try:
            from app.pipeline1.prediction_db import PredictionDB
            PredictionDB().insert_run(trade_date, result["list"])
        except Exception:
            pass

    return result["list"]


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Full A-share train + predict")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--stocks", type=int, default=0, help="Limit stocks (0=all)")
    parser.add_argument("--tag", type=str, default=None, help="Model tag")
    args = parser.parse_args()

    trade_date = datetime.now().strftime("%Y%m%d")
    tag = args.tag
    if tag is None:
        iso = datetime.now().isocalendar()
        tag = f"{iso[0]}W{iso[1]:02d}"

    do_all = not (args.fetch_only or args.train_only or args.predict_only)

    # 1. Fetch
    if args.fetch_only or do_all:
        print(f"\n{'='*60}")
        print(f"  STAGE 1: Fetch {args.stocks or 'ALL'} A-stocks, {YEARS}yr data")
        print(f"{'='*60}")
        panel = fetch_all_stocks(limit=args.stocks)
        if panel is None:
            print("ERROR: Data fetch failed")
            sys.exit(1)
        print(f"Fetched: {panel.symbol.nunique()} stocks, {len(panel)} rows")

    # 2. Train
    if args.train_only or do_all:
        print(f"\n{'='*60}")
        print(f"  STAGE 2: Train models (tag={tag})")
        print(f"{'='*60}")
        if not os.path.exists(PANEL_PATH):
            print("ERROR: Panel not found. Run --fetch first.")
            sys.exit(1)
        panel = pd.read_parquet(PANEL_PATH)
        results = run_training(panel, tag=tag)
        if not results:
            print("ERROR: Training failed")
            sys.exit(1)

    # 3. Predict
    if args.predict_only or do_all:
        print(f"\n{'='*60}")
        print(f"  STAGE 3: Predict {trade_date}")
        print(f"{'='*60}")
        lst = run_prediction(trade_date)
        if lst is not None and len(lst):
            print(f"\n{'Symbol':<8s} {'Board':<6s} {'Prob':<6s} {'Score':<10s} {'Pred1d':<9s}")
            print("-" * 50)
            for _, r in lst.head(20).iterrows():
                print(f"{r['symbol']:<8s} {r['board']:<6s} {float(r['prob_up']):.0%}     {float(r['score']):.4f}     {float(r['pred_ret_1d']):+.4f}")
        else:
            print("No stocks in list (safety valve or no candidates)")

    print(f"\nDone: {trade_date} tag={tag}")


if __name__ == "__main__":
    main()
