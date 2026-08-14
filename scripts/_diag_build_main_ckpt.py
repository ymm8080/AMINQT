"""_diag_build_main_ckpt.py — 只重建 main 3y 检查点, 捕获真实错误 (2026-08-12).

并行 train 步骤的 refresh 反复在 main 构建上失败 (main 检查点自 08:15 起缺失),
dual 已重建成功. 本脚本只构建 main, 先释放 dual 帧 (不复建 dual), 峰值内存
与 refresh 相当, 并把 traceback 打到 stdout 以便定位失败原因.

用法: python -u scripts/_diag_build_main_ckpt.py
输出: data/_diag_stage_main_3y.parquet (成功) 或 traceback (失败).
"""

from __future__ import annotations

import gc
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")


from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from scripts._reclassify_all_features import MAIN_CHECKPOINT, build_board_slice


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("读取 V3 面板 ...", flush=True)
    panel = load_panel_v3()
    fe = FeatureEngineV35()
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()
    print(
        f"run_train: main rows={len(main_df):,} / dual rows={len(dual_df):,}",
        flush=True,
    )
    # dual 检查点已新鲜 (refresh 19:12 重建成功), 不复建 → 释放 dual 帧省内存.
    del dual_df
    gc.collect()

    d3 = build_board_slice(cleaner, fe, main_df, "main", MAIN_CHECKPOINT)
    print(
        f"MAIN CHECKPOINT BUILT rows={len(d3):,} latest={d3['date'].max():%Y-%m-%d}",
        flush=True,
    )
    return


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
