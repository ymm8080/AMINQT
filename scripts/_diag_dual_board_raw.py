"""诊断: 双板原始预测截面 (最新交易日, 生产同路径推理).

镜像 _gen_legacy_list 的数据通路 (300 交易日切片 → CleaningPipeline.run_train →
FeatureEngineV35.build(inference_cols, dual 横截面) → V35Predictor.predict),
直接输出每股原始 pred_ret_{1,2,3,5}d / prob_up / prob_up_{2,3,5}d /
pred_q50_{1,2,3,5}d, 不经过任何排序/门槛, 定位 prob_up_3d 塌缩是否真实.

用法: python scripts/_diag_dual_board_raw.py
输出 (WORM): data/_diag_dual_board_raw_{ts}.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH

MODEL_DIR = "models/pipeline1"
SLICE_DAYS = 300
BOARDS = ("main", "dual")


def _latest_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("date").groupby("symbol").tail(1).copy()


def main() -> int:
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-SLICE_DAYS]]
    print(f"[slice] {SLICE_DAYS}d -> {len(panel):,}r", flush=True)

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()
    print(
        f"[clean] main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    predictor = V35Predictor(
        {b: os.path.join(MODEL_DIR, f"{b}_current.pkl") for b in BOARDS}
    )
    out = {}
    for board, df in (("main", main_df), ("dual", dual_df)):
        if len(df) == 0:
            print(f"[{board}] 空, 跳过", flush=True)
            out[board] = None
            continue
        cols = predictor.bundles[board]["feature_cols"]
        df = features.build(
            df, None, inference_cols=cols, cross_sectional_rank=(board == "dual")
        )
        print(
            f"[{board}] features build cols={len(cols)} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        lat = predictor.predict(df, board)
        print(f"[{board}] predict {len(lat)} 只 ({time.time() - t0:.0f}s)", flush=True)
        del df
        gc.collect()
        # 排序展示用列
        show = lat.sort_values("pred_ret_3d", ascending=False)
        cols_show = [
            c
            for c in (
                "symbol",
                "board",
                "industry",
                "composite_score",
                "pred_ret_1d",
                "pred_ret_2d",
                "pred_ret_3d",
                "pred_ret_5d",
                "prob_up",
                "prob_up_2d",
                "prob_up_3d",
                "prob_up_5d",
                "pred_q50_1d",
                "pred_q50_2d",
                "pred_q50_3d",
                "pred_q50_5d",
                "day_change",
            )
            if c in show.columns
        ]
        tab = show[cols_show].round(4).reset_index(drop=True)
        print(f"\n===== {board} ({len(tab)} 只, 按 pred_ret_3d 降序) =====", flush=True)
        print(tab.to_string(index=False), flush=True)
        # 概率塌缩统计
        for k in (1, 2, 3, 5):
            col = f"prob_up_{k}d" if k != 1 else "prob_up"
            if col in tab.columns:
                vals = tab[col].dropna()
                print(
                    f"[{board}] {col}: nunique={vals.nunique()} "
                    f"min={vals.min():.4f} max={vals.max():.4f} "
                    f"std={vals.std():.4f}",
                    flush=True,
                )
        out[board] = tab.to_dict(orient="records")
        del lat, tab
        gc.collect()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = f"data/_diag_dual_board_raw_{stamp}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[saved] {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
