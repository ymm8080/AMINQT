# -*- coding: utf-8 -*-
"""_refresh_parallel_checkpoints.py — 重建 parallel pipeline 的 main/dual 3y 检查点 (2026-08-04).

parallel pipeline (app/pipeline_parallel) 的 load_panel 读取两个 3y 检查点:
  data/_diag_stage_main_3y.parquet, data/_diag_stage_dual_3y.parquet
每天日更后 V3 面板前移, 检查点若仍指向旧日期 → 短名单/回测会缺最新交易日.
本脚本从 V3 面板重建两个检查点 (复用生产 build_board_slice → 与生产行集完全一致),
旧检查点改名为 <name>.stale_<ts> 而非删除 (可回溯).

用法: python scripts/_refresh_parallel_checkpoints.py
输出: 两个新检查点 + 控制台日志 (最新日期 / 行数 / 列数).
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from scripts._reclassify_all_features import (
    DUAL_CHECKPOINT,
    MAIN_CHECKPOINT,
    build_board_slice,
)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # 1. 旧检查点改名为 .stale_<ts> (保留可回溯)
    for ck in (MAIN_CHECKPOINT, DUAL_CHECKPOINT):
        if os.path.exists(ck):
            bak = f"{ck}.stale_{ts}"
            os.rename(ck, bak)
            print(f"[stale] {ck} -> {bak}", flush=True)

    # 2. 读 V3 面板 → run_train 拆 main/dual → 重建两检查点
    print("读取 V3 面板 ...", flush=True)
    panel = pd.read_parquet(PANEL_V3_PATH)
    fe = FeatureEngineV35()
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()
    print(
        f"run_train: main rows={len(main_df):,} / dual rows={len(dual_df):,}",
        flush=True,
    )

    for board, ckpt, bdf in (
        ("main", MAIN_CHECKPOINT, main_df),
        ("dual", DUAL_CHECKPOINT, dual_df),
    ):
        if bdf is None or len(bdf) == 0:
            print(f"[{board}] 空, 跳过", flush=True)
            continue
        d3 = build_board_slice(cleaner, fe, bdf, board, ckpt)
        print(
            f"[{board}] 检查点已写 {ckpt} | latest={d3['date'].max():%Y-%m-%d} "
            f"rows={len(d3):,} cols={d3.shape[1]:,}",
            flush=True,
        )
        del bdf, d3
        gc.collect()
    del main_df, dual_df, fe, cleaner
    gc.collect()
    print("完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
