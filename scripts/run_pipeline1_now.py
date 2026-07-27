# -*- coding: utf-8 -*-
"""
实盘前 Pipeline1 预测 (收盘前跑)
=================================
用法:
  python scripts/run_pipeline1_now.py              # 全 A 股扫描
  python scripts/run_pipeline1_now.py --top 20     # 只看前 20 只
  python scripts/run_pipeline1_now.py --symbols "002353,600958,000001"  # 指定股票

流程: 拉取最新日线 (akshare) → 清洗 → 特征 → 推理 → 校准 → 清单生成 → 输出

输出:
  data/lists/list_YYYYMMDD.parquet  — 完整清单
  stdout — Top-N 表格
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline1_now")

# ---- 模型路径 ----
MODEL_DIR = "models/pipeline1"
LIST_DIR = "data/lists"


def fetch_today_data(symbols: list[str] | None = None) -> pd.DataFrame | None:
    """拉取全 A 股最新日线 (今日 + 历史 720 天).

    Args:
        symbols: 指定股票代码列表; None = 全市场.

    Returns:
        panel DataFrame (symbol/date/open/high/low/close/volume/amount/board),
        或 None (数据源不可用).
    """
    try:
        import akshare as ak

        if symbols is None:
            # 全 A 股列表
            logger.info("拉取全 A 股列表 ...")
            stock_info = ak.stock_zh_a_spot_em()
            symbols = stock_info["代码"].tolist()
            logger.info("共 %d 只", len(symbols))
            # 控制总量避免 API 限流
            symbols = symbols[:200]  # 先取前 200 只测试, 全量跑再加

        frames = []
        today = datetime.now().strftime("%Y%m%d")
        start = "20240101"

        for i, sym in enumerate(symbols):
            if i % 50 == 0:
                logger.info("拉取进度: %d/%d", i, len(symbols))
            try:
                df = ak.stock_zh_a_hist(
                    symbol=sym, period="daily", start_date=start,
                    end_date=today, adjust="qfq",
                )
                if df is None or len(df) == 0:
                    continue
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "amount",
                })
                df["symbol"] = sym
                df["date"] = pd.to_datetime(df["date"])
                for col in ["open", "high", "low", "close", "volume", "amount"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["pre_close"] = df["close"].shift(1)
                # board 判断: 300/688/301 → dual, 其他 → main
                df["board"] = "dual" if sym.startswith(("300", "688", "301")) else "main"
                df["industry"] = "UNKNOWN"
                frames.append(df)
                time.sleep(0.1)  # 反限流
            except Exception as e:
                logger.debug("跳过 %s: %s", sym, e)

        if not frames:
            logger.error("未拉到任何数据")
            return None

        panel = pd.concat(frames, ignore_index=True)
        panel = panel.dropna(subset=["close"]).sort_values(["symbol", "date"])
        logger.info("面板: %d 只, %d 行, %s ~ %s",
                     panel["symbol"].nunique(), len(panel),
                     panel["date"].min().strftime("%Y-%m-%d"),
                     panel["date"].max().strftime("%Y-%m-%d"))
        return panel

    except Exception:
        logger.exception("数据拉取失败 (akshare 不可用?)")
        return None


def run_pipeline(panel: pd.DataFrame, trade_date: str) -> dict:
    """运行 Pipeline1 完整推理链路."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.list_generator import ListGenerator, MarketEnv
    from app.pipeline1.predictor import V35Predictor

    # 1. 定位最新模型包
    from app.pipeline1.predict_runner import find_bundles
    bundles = find_bundles(model_dir=MODEL_DIR)
    if "main" not in bundles:
        raise RuntimeError(f"主板块模型包缺失: {MODEL_DIR}")
    logger.info("模型: %s", {k: os.path.basename(v) for k, v in bundles.items()})

    # 2. 清洗
    cleaner = CleaningPipeline()
    main_df, dual_df, valve_state = cleaner.run_inference(panel)
    logger.info("清洗: main=%d dual=%d valve=%s", len(main_df), len(dual_df), valve_state)
    if valve_state == "empty":
        logger.error("流动性安全阀触发: 空清单")
        return {"mode": "valve_empty", "list": pd.DataFrame(), "empty": True}

    # 3. 特征
    features = FeatureEngineV35()
    feat_main = features.build(main_df) if len(main_df) else pd.DataFrame()
    feat_dual = features.build(dual_df) if len(dual_df) else pd.DataFrame()

    # 4. 推理
    predictor = V35Predictor(bundles)
    frames = []
    for board, feat, survivors in [
        ("main", feat_main, main_df),
        ("dual", feat_dual, dual_df),
    ]:
        if len(feat) == 0:
            continue
        latest_date = survivors["date"].max()
        today_feat = feat[feat["symbol"].isin(
            set(survivors[survivors["date"] == latest_date]["symbol"])
        )]
        if len(today_feat) == 0:
            continue
        pred = predictor.predict(today_feat, board)
        # 合并 board/industry/limit 信息
        pred["board"] = board
        frames.append(pred)

    if not frames:
        return {"mode": "no_candidates", "list": pd.DataFrame(), "empty": True}

    candidates = pd.concat(frames, ignore_index=True)
    logger.info("候选: %d 只 (main=%d dual=%d)",
                 len(candidates),
                 len(frames[0]) if len(frames) > 0 else 0,
                 len(frames[1]) if len(frames) > 1 else 0)

    # 5. 清单生成
    env = MarketEnv()
    lister = ListGenerator()
    result = lister.emit(candidates, env=env, market_state="range")
    logger.info("清单: %d 只 empty=%s cap=%.2f",
                 len(result["list"]), result["empty"], result["cap_position"])

    # 6. 持久化
    os.makedirs(LIST_DIR, exist_ok=True)
    if not result["empty"] and len(result["list"]):
        path = os.path.join(LIST_DIR, f"list_{trade_date}.parquet")
        result["list"].to_parquet(path, index=False)
        logger.info("已保存: %s", path)

        # 同步 priority.json
        _sync_priority(result["list"])

        # 入库 prediction DB
        try:
            from app.pipeline1.prediction_db import PredictionDB
            PredictionDB().insert_run(trade_date, result["list"])
        except Exception:
            pass

    return result


def _sync_priority(list_df: pd.DataFrame) -> None:
    """将清单股票合并到 priority.json."""
    import json
    pq_path = os.path.join("data", "priority.json")
    existing = set()
    if os.path.exists(pq_path):
        with open(pq_path, "r", encoding="utf-8") as f:
            existing = set(json.load(f).get("symbols", []))
    new_syms = set(list_df["symbol"].tolist())
    merged = sorted(existing | new_syms)
    with open(pq_path, "w", encoding="utf-8") as f:
        json.dump({"symbols": merged}, f, ensure_ascii=False, indent=2)
    logger.info("priority.json: %d → %d (新增 %d)", len(existing), len(merged), len(new_syms - existing))


def print_list(df: pd.DataFrame, top: int = 20) -> None:
    """打印清单 (表格)."""
    if df is None or len(df) == 0:
        print("\n  WARNING:️ 空清单 — 当日无候选\n")
        return

    cols = ["symbol", "board", "prob_up", "score", "momentum",
            "pred_ret_1d", "pred_ret_3d", "weight"]
    available = [c for c in cols if c in df.columns]
    show = df[available].head(top).copy()

    # 格式化
    for c in ["pred_ret_1d", "pred_ret_3d"]:
        if c in show.columns:
            show[c] = show[c].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "—")
    if "prob_up" in show.columns:
        show["prob_up"] = show["prob_up"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    if "score" in show.columns:
        show["score"] = show["score"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    if "weight" in show.columns:
        show["weight"] = show["weight"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "—")

    print(f"\n{'='*80}")
    print(f"  Pipeline1 推荐股票池 · {date.today().isoformat()} · Top {min(top, len(df))}/{len(df)}")
    print(f"{'='*80}")
    print(show.to_string(index=False))
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Pipeline1 实盘预测")
    parser.add_argument("--top", type=int, default=20, help="显示前 N 只")
    parser.add_argument("--symbols", type=str, default=None, help="指定股票, 逗号分隔")
    parser.add_argument("--no-fetch", action="store_true", help="跳过数据拉取, 用已有 panel")
    args = parser.parse_args()

    trade_date = datetime.now().strftime("%Y%m%d")
    symbols = args.symbols.split(",") if args.symbols else None

    print(f"\n{'='*60}")
    print(f"  Pipeline1 实盘预测 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  模型: {MODEL_DIR}")
    print(f"{'='*60}")

    # 1. 拉取数据
    panel = None
    if not args.no_fetch:
        print("\n[1/4] 拉取市场数据 ...")
        panel = fetch_today_data(symbols)
        if panel is None:
            print("\nERROR: 数据拉取失败 (网络不通或 akshare 限流)")
            print("   解决方案:")
            print("   1. 确认外网连通 (ping eastmoney.com)")
            print("   2. 用 --symbols 缩小范围 (如 --symbols '002353,600958')")
            print("   3. 改用 --no-fetch + 已有数据文件")
            sys.exit(1)

    # 2. 跑 Pipeline
    print("\n[2/4] 运行 Pipeline1 推理 ...")
    try:
        result = run_pipeline(panel, trade_date)
    except Exception as e:
        logger.exception("Pipeline1 运行失败")
        print(f"\nERROR: 运行失败: {e}")
        sys.exit(1)

    # 3. 输出
    print("\n[3/4] 生成结果 ...")
    print_list(result.get("list"), top=args.top)

    # 4. 质量提示
    print("[4/4] 收盘后运行 reconcile 回填实际收益:")
    print(f"  python scripts/reconcile_predictions.py {trade_date}")


if __name__ == "__main__":
    main()
