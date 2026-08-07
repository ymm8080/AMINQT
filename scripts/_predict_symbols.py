"""诊断: 对指定 symbol 用当前 legacy 模型 (dual/main) 做当日预测, 不重训.

与生产链路 (cleaner -> feature_engine -> V35Predictor) 完全一致, 但:
  - 只读末 300 交易日切片 (pyarrow row filter, 避免全量 2.67M 行读入)
  - 只构建目标 symbol 所在板块特征 (可只走 dual)
  - 只对目标 symbol 推理
结果逐字段与 list_{date}.parquet 对齐 (验证: 传一个已入选股对比).

用法:
  python scripts/_predict_symbols.py 20260806 301326 300911 [--boards dual]
  python scripts/_predict_symbols.py 20260806 002319        # 验证用 (对比 list_20260806)
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain
from config.settings import PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}


def main() -> int:
    args = sys.argv[1:]
    trade_date = args[0] if (args and len(args[0]) == 8 and args[0].isdigit()) else None
    symbols = [a for a in args if a != trade_date]
    boards = []
    for i, a in enumerate(args):
        if a == "--boards":
            boards = args[i + 1 :]
    t0 = time.time()
    if not symbols:
        print("[usage] python scripts/_predict_symbols.py YYYYMMDD SYMBOL...", flush=True)
        return 1

    # 1) 末 300 交易日切片 (pyarrow row filter, 列全读)
    dates = pq.read_table(str(PANEL_V3_PATH), columns=["date"]).to_pandas()["date"]
    uniq = np.unique(dates.values)
    cut = pd.Timestamp(uniq[-300])
    panel = pq.read_table(
        str(PANEL_V3_PATH), filters=[("date", ">=", cut)]
    ).to_pandas()
    print(
        f"[panel] {len(panel):,}r 切片 {cut.date()}..{pd.Timestamp(uniq[-1]).date()} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    main_df, dual_df, valve = pipe.cleaner.run_inference(panel)
    del panel
    print(
        f"[clean] main={len(main_df):,} dual={len(dual_df):,} valve={valve} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    want = set(symbols)
    out_frames = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        if boards and board not in boards:
            continue
        if len(dfb) == 0 or board not in pipe.predictor.bundles:
            continue
        present = set(dfb["symbol"].astype(str).str.zfill(6)) & want
        if not present:
            continue
        cols = pipe.predictor.bundles[board]["feature_cols"]
        t1 = time.time()
        feat = pipe.features.build(
            dfb,
            pipe.float_shares_map,
            inference_cols=cols,
            cross_sectional_rank=csr,
        )
        print(f"[feat {board}] {len(feat):,}r ({time.time() - t1:.0f}s)", flush=True)
        sub = feat[feat["symbol"].astype(str).str.zfill(6).isin(want)].copy()
        if len(sub) == 0:
            continue
        pred = pipe.predictor.predict(sub, board)
        pred["symbol"] = pred["symbol"].astype(str).str.zfill(6)
        # 附最新行情/波动列
        latest = (
            dfb[dfb["symbol"].astype(str).str.zfill(6).isin(want)]
            .sort_values("date")
            .groupby("symbol")
            .tail(1)
        )
        for col in ("close", "pre_close", "ATR_pct", "adv20", "amount", "turnover_rate"):
            if col in latest.columns:
                pred[col] = latest.set_index("symbol").reindex(pred["symbol"])[col].values
        out_frames.append(pred)

    if not out_frames:
        print(f"[miss] 目标股 {sorted(want)} 不在可预测板块 (被清洗剔除或缺模型)", flush=True)
        return 1
    res = pd.concat(out_frames, ignore_index=True)

    show = [
        "symbol", "board", "industry", "close", "day_change",
        "pred_ret_1d", "pred_ret_2d", "pred_ret_3d", "pred_ret_5d",
        "prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d",
        "pred_q10", "pred_q50", "pred_q90",
        "pain_prob", "rank_score", "composite_score",
        "ATR_pct", "adv20",
    ]
    show = [c for c in show if c in res.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    print("\n" + "=" * 120)
    print(f"当前 legacy 模型 ({trade_date}) 逐股预测:")
    print(res[show].to_string(index=False), flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)

    if trade_date:
        list_path = f"data/lists/list_{trade_date}.parquet"
        if os.path.exists(list_path):
            lst = pd.read_parquet(list_path)
            lst["symbol"] = lst["symbol"].astype(str).str.zfill(6)
            hit = lst[lst["symbol"].isin(want)]
            if len(hit):
                print("\n[对比] 当日已落盘清单 (生产 _gen_legacy_list 输出):", flush=True)
                cmp_cols = [c for c in ("symbol", "pred_ret_1d", "pred_ret_2d", "pred_ret_3d", "pred_ret_5d", "prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d", "score") if c in hit.columns]
                print(hit[cmp_cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
