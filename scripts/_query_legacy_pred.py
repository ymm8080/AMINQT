"""查询: 用当前 legacy 模块对指定 symbol 查当日预测.

默认路径 (instant): 读当日已持久化的全量候选预测 candidates_{date}.parquet
  (由 DailySelectionPipeline.run() 每日自动落盘 WORM, 无需任何重算).
仅当 candidates 缺失且显式传 --rebuild 才走慢路径 (全市场特征构建 + 预测).

用法:
  python scripts/_query_legacy_pred.py [YYYYMMDD] [symbol...]
  python scripts/_query_legacy_pred.py 20260806 301326 300911
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

LIST_DIR = "data/lists"

_COLS = [
    "symbol", "board", "pred_ret_1d", "pred_ret_2d", "pred_ret_3d", "pred_ret_5d",
    "prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d",
    "pred_q50", "pred_q50_2d", "pred_q50_3d", "pred_q50_5d",
    "pain_prob", "score", "weight", "compound_ret", "model_version",
]


def _read_frame(path):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    return df


def _show(df, want, label):
    print(f"[{label}] {len(df):,} 只", flush=True)
    hit = df[df["symbol"].isin(want)]
    cols = [c for c in _COLS if c in hit.columns]
    if len(hit):
        print(hit[cols].to_string(index=False), flush=True)
    return hit


def main():
    args = sys.argv[1:]
    rebuild = "--rebuild" in args
    if rebuild:
        args = [a for a in args if a != "--rebuild"]
    trade_date = args[0] if args else "20260804"
    want = set(a.zfill(6) for a in args[1:]) if len(args) > 1 else None

    cand_path = os.path.join(LIST_DIR, f"candidates_{trade_date}.parquet")
    list_path = os.path.join(LIST_DIR, f"list_{trade_date}.parquet")
    df = _read_frame(cand_path)
    if df is None:
        print(f"[warn] 无候选持久化 {cand_path} — 当日可能未跑完整推理", flush=True)
        df = _read_frame(list_path)
        if df is not None:
            print(f"[info] 回退读当日入选清单 {list_path} (仅含入选股)", flush=True)

    if df is not None and want is not None:
        hit = _show(df, want, "candidates")
        missing = sorted(want - set(hit["symbol"])) if len(hit) else sorted(want)
        if missing:
            print(f"[miss] 未在候选池: {missing}", flush=True)
            print(
                "[hint] 该股当日未被预测 (清洗/特征过滤剔除), 或 candidates 未落盘. "
                "如需强制重算: python scripts/_query_legacy_pred.py --rebuild "
                f"{trade_date} {' '.join(missing)}",
                flush=True,
            )
        return

    if not rebuild:
        print(
            "[err] 无任何当日持久化帧. 若必须现场重算, 加 --rebuild "
            "(全市场特征构建 ~17min)",
            flush=True,
        )
        sys.exit(1)

    # ── 慢路径: 全市场特征构建 + 预测 (显式 --rebuild 才走) ──
    from app.pipeline1.daily_pipeline import DailySelectionPipeline
    from app.pipeline1.data_supply import DataSupplyChain
    from config.settings import PANEL_V3_PATH

    BUNDLES = {
        "main": "models/pipeline1/main_current.pkl",
        "dual": "models/pipeline1/dual_current.pkl",
    }
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-300]]
    print(f"[panel] {len(panel):,}r slice300 ({time.time()-t0:.0f}s)", flush=True)
    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    main_df, dual_df, valve = pipe.cleaner.run_inference(panel)
    print(f"[clean] main={len(main_df):,} dual={len(dual_df):,} valve={valve}", flush=True)
    frames = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        if not len(dfb):
            continue
        f = pipe.features.build(
            dfb, pipe.float_shares_map,
            inference_cols=pipe.predictor.bundles[board]["feature_cols"],
            cross_sectional_rank=csr,
        )
        latest = dfb[dfb["date"] == dfb["date"].max()]["symbol"]
        frames.append(pipe.predictor.predict(f[f["symbol"].isin(set(latest))], board))
    cand = pd.concat(frames, ignore_index=True)
    cand["symbol"] = cand["symbol"].astype(str).str.zfill(6)
    _show(cand, want or set(), "candidates(rebuild)")
    if want:
        missing = sorted(want - set(cand["symbol"]))
        if missing:
            print(f"[miss] 重算后仍不在候选池: {missing}", flush=True)


if __name__ == "__main__":
    main()
