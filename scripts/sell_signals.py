# -*- coding: utf-8 -*-
"""持仓卖出信号: 预测驱动 (绿/黄/橙/红) + 价格硬止损 -6%.

用法:
  python scripts/sell_signals.py --date 20260803 --symbols 600519,000001
  python scripts/sell_signals.py --date 20260803 --symbols 600519 --entry data/positions.csv
  python scripts/sell_signals.py --date 20260803            # 缺省昨日清单 (无则需 --symbols)

entry csv: 列 symbol, cost (买入成本价, 元); 现价取当日面板 close.
输出: data/sell_signals_{date}.parquet + 终端表格.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.predict_runner import find_bundles

SIGNAL_LABEL = {
    "hold": "持有(绿)",
    "watch": "警戒(黄)",
    "sell": "卖出(橙)",
    "strong_sell": "强卖(红)",
}


def _yesterday_symbols(list_dir: str = "data/lists") -> list[str]:
    """上一交易日清单 = 今日实际持仓 (T 日清单 T+1 买入)."""
    if not os.path.isdir(list_dir):
        return []
    dates = sorted(
        f.replace("list_", "").replace(".parquet", "")
        for f in os.listdir(list_dir)
        if f.startswith("list_")
    )
    if not dates:
        return []
    last = os.path.join(list_dir, f"list_{dates[-1]}.parquet")
    df = pd.read_parquet(last)
    return df["symbol"].tolist()


def _read_entry(path: str) -> dict:
    cost = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = str(row["symbol"]).strip()
            cost[sym] = float(row["cost"])
    return cost


def main() -> int:
    ap = argparse.ArgumentParser(description="持仓卖出信号")
    ap.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--symbols", help="逗号分隔持仓股 (缺省取昨日清单)")
    ap.add_argument("--entry", help="持仓成本 csv (symbol,cost)")
    ap.add_argument(
        "--tag",
        default="2026W31_fix",
        help="模型包 tag (find_bundles 默认取最新, 可能挑中训练期 brute 包"
        "无法推理; 默认钉在可复现的 2026W31_fix)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols
        else _yesterday_symbols()
    )
    if not symbols:
        print("未指定 --symbols 且昨日清单为空, 无法生成卖出信号")
        return 1

    bundles = find_bundles(tag=args.tag) or find_bundles()
    if not bundles:
        print("无可用模型包 (models/pipeline1/*.pkl), 请先训练")
        return 1
    print("模型包:", {k: os.path.basename(v) for k, v in bundles.items()})

    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=bundles)
    entry_cost_map = _read_entry(args.entry) if args.entry else None

    print(
        f"日期 {args.date} | 持仓 {len(symbols)} 只 | "
        f"{'价格硬止损 -6% 已启用' if entry_cost_map else '仅预测信号 (无价格硬止损)'}"
    )
    try:
        out = pipe.predict_held(args.date, symbols, entry_cost_map=entry_cost_map)
    except Exception as exc:  # 数据供应链失败等
        print(f"预测失败: {exc!r}")
        return 1
    if len(out) == 0:
        print("无持仓股可预测 (可能不在面板/清洗后被过滤)")
        return 1

    out["signal_label"] = out["sell_signal"].map(SIGNAL_LABEL)
    rank = {"strong_sell": 3, "sell": 2, "watch": 1, "hold": 0}
    out["_lv"] = out["sell_signal"].map(rank)
    out = out.sort_values("_lv", ascending=False).drop(columns="_lv")

    cols = [
        c
        for c in (
            "symbol",
            "board",
            "close",
            "pnl",
            "signal_label",
            "sell_reason",
            "pred_ret_1d",
            "pred_ret_3d",
            "prob_up",
            "pain_prob",
        )
        if c in out.columns
    ]
    print("\n" + out[cols].to_string(index=False))

    out_path = args.out or f"data/sell_signals_{args.date}.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n已写出: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
