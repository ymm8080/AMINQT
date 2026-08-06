# -*- coding: utf-8 -*-
"""_select_freq_three.py — 三频选择: 真实面板跑 select_freq → {月/周/日} 三表.

复刻 train_runner.run_training 的 board 切分序列 (cleaner.run_train →
prepare_board_frame → select), 但用 FeatureSelector.select_freq 把选中特征按
基列频率路由到 {月, 周, 日} 三张表 (WORM 落盘 + 覆盖率报告).
铁律验证: 月频特征不进日频表; 未分类特征显式暴露, 不静默默认.

用法: python scripts/_select_freq_three.py [board]
   board: main (默认) | dual
输出: factor_registry/selected_{board}_{月|周|日}_{ts}.json 三张表
      + selected_{board}_freq_{ts}.json 覆盖率报告
      + data/_select_freq_three_{ts}.log 控制台摘要 (WORM)
"""
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from config.settings import PANEL_V3_PATH, data_others_path
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import (
    FREQ_ASSIGNMENT,
    FREQ_ORDER,
    FeatureSelector,
)
from app.pipeline1.train_runner import prepare_board_frame

logging.disable(logging.CRITICAL)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    board = sys.argv[1] if len(sys.argv) > 1 else "main"
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = []

    print(f"[{board}] 载入面板 {PANEL_V3_PATH} ...", flush=True)
    panel = pd.read_parquet(PANEL_V3_PATH)
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()

    board_df = {"main": main_df, "dual": dual_df}.get(board)
    if board_df is None or len(board_df) == 0:
        print(f"[{board}] 无样本, 退出", flush=True)
        return
    print(f"[{board}] 清洗后 rows={len(board_df):,} stocks={board_df['symbol'].nunique():,}",
          flush=True)

    features = FeatureEngineV35()
    use_xrank = board != "main"
    df = prepare_board_frame(
        board_df, features, None, cross_sectional_rank=use_xrank
    )
    del board_df
    gc.collect()
    print(f"[{board}] 训练面板 rows={len(df):,} cols={df.shape[1]}", flush=True)

    selector = FeatureSelector(
        registry_dir=str(data_others_path("data/factor_registry"))
    )
    buckets = selector.select_freq(df, board)

    out.append(f"=== 三频选择 [{board}] {ts} ===")
    out.append(f"训练面板 rows={len(df):,} cols={df.shape[1]}")
    out.append("覆盖率: " + "  ".join(
        f"{k}={len(v):,}" for k, v in buckets.items()
    ))
    total = sum(len(v) for v in buckets.values())
    out.append(f"选中总数 {total:,}")
    out.append("")

    for freq in FREQ_ORDER:
        feats = buckets[freq]
        out.append(f"--- {freq}频表 ({len(feats):,}) ---")
        confirmed_hit = [f for f in feats if f.split('_brute_')[0] in FREQ_ASSIGNMENT
                         and FREQ_ASSIGNMENT[f.split('_brute_')[0]][0] == freq]
        out.append(f"  与确认判定一致的基列: {len(confirmed_hit)}")
        out.append("  样例: " + ", ".join(feats[:15]))
        out.append("")

    out.append(f"--- 事件桶 ({len(buckets['事件']):,}) ---")
    out.append("  样例: " + ", ".join(buckets["事件"][:10]))
    out.append("")
    out.append(f"--- 未分类 ({len(buckets['未分类']):,}) — 需扩 FREQ_ASSIGNMENT ---")
    for f in buckets["未分类"]:
        out.append("  ? " + f)

    text = "\n".join(out)
    print(text, flush=True)
    p = os.path.join("data", f"_select_freq_three_{board}_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}", flush=True)
    print(f"三表+覆盖率报告: {selector.registry_dir}", flush=True)

    del df
    gc.collect()


if __name__ == "__main__":
    main()
