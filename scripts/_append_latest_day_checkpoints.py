# -*- coding: utf-8 -*-
"""_append_latest_day_checkpoints.py — 增量把最新交易日 append 到并行 main/dual 检查点.

背景 (2026-08-07 确认): 全量 refresh (_refresh_parallel_checkpoints.py) 在本机 402 列 V3
面板上跑 fe.build, 其内部 `df.sort_values(["symbol","date"])` 触发 pandas block
consolidation → 需 3.47 GiB 连续内存 (415×1,121,374 float64), 进程 ~52GB commit 时仍
OOM (见 memory/machine-ram-block-consolidation). 本机 15.8GB 物理, 全量重建不可行.

本脚本绕开: 只对「检查点符号 × 最近 WINDOW_DAYS 日历天」的面板窗口跑生产同款管线
(cleaner 清洗 → fe.build), 取最新交易日 (aug07) 行, 按检查点列对齐后 append.
窗口 sort 数组 ~2GB 以下, 内存安全.

与生产行集一致 (build_board_slice / _reclassify_all_features):
  run_train per-board 清洗 (step1→step2(apply_top_n=False)→step3→step5) + fe.build
  (cross_sectional_rank = (board=='dual')). 标签不持久化 — load_panel 的 _finalize_slice
  + add_mfe_labels + add_c2c_labels 在读取时现算; append 后 aug07 行标签为 NaN
  (未来价不存在, 与 mask_recent_days 语义一致).

用法:
  python scripts/_append_latest_day_checkpoints.py                  # 写回检查点
  python scripts/_append_latest_day_checkpoints.py --window-days 400
  python scripts/_append_latest_day_checkpoints.py --limit-syms 50  # 验证: 只取前 N 符号,
                                                                    # 不写回, 打印列对齐报告
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import CleaningPipeline, board_of as p1_board_of
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from config.settings import PANEL_V3_PATH
from scripts._reclassify_all_features import DUAL_CHECKPOINT, MAIN_CHECKPOINT

META = {"symbol", "date", "board", "is_suspended"}


def load_ckpt_meta(path: str) -> tuple[set[str], list[str]]:
    """返回 (symbol 集合, 列名列表) — 只读 symbol 列 + pyarrow schema, 不整读 4GB."""
    syms = set(pd.read_parquet(path, columns=["symbol"])["symbol"].astype(str))
    cols = pq.read_schema(path).names
    return syms, cols


def build_window_for_board(
    panel: pd.DataFrame,
    ckpt_syms: set[str],
    latest: pd.Timestamp,
    cutoff: pd.Timestamp,
    board: str,
    limit_syms: int = 0,
) -> pd.DataFrame:
    """按检查点符号集切最近窗口, 镜像 run_train 清洗 (返回已清洗窗口)."""
    if limit_syms:
        ckpt_syms = set(sorted(ckpt_syms)[:limit_syms])
    w = panel[panel["symbol"].astype(str).isin(ckpt_syms) & (panel["date"] >= cutoff)]
    w = w.copy().reset_index(drop=True)
    w["board"] = w["symbol"].map(p1_board_of)
    cleaner = CleaningPipeline()
    w = w.sort_values(["symbol", "date"]).reset_index(drop=True)
    w = cleaner.step5_amount_bottom(
        cleaner.step3_extreme(
            cleaner.step2_liquidity(cleaner.step1_base_state(w), apply_top_n=False)
        )
    )
    del cleaner
    gc.collect()
    return w


def process_board(
    panel: pd.DataFrame,
    ckpt_path: str,
    board: str,
    latest: pd.Timestamp,
    cutoff: pd.Timestamp,
    *,
    limit_syms: int = 0,
    write: bool,
) -> dict:
    ckpt_syms, ckpt_cols = load_ckpt_meta(ckpt_path)
    w = build_window_for_board(
        panel, ckpt_syms, latest, cutoff, board, limit_syms=limit_syms
    )
    if len(w) == 0:
        return {"board": board, "error": "窗口为空"}

    fe = FeatureEngineV35()
    d = fe.build(w, None, cross_sectional_rank=(board != "main"), registry=None)
    del w
    gc.collect()
    aug = d[d["date"] == latest].copy()
    del d
    gc.collect()

    if aug.empty:
        return {"board": board, "error": f"latest={latest:%Y-%m-%d} 无行"}
    aug["symbol"] = aug["symbol"].astype(str)
    aug["is_suspended"] = aug["is_suspended"].astype(bool)
    aug = aug.reindex(columns=ckpt_cols)

    if limit_syms:
        # 验证模式: 报告列对齐情况, 不写回
        nan_share = aug[ckpt_cols].isna().mean()
        na_cols = nan_share[nan_share > 0].sort_values(ascending=False)
        pool = ["amihud_illiq", "small_mv_premium", "amihud_illiquidity",
                "down_gap_pct", "VAR51", "ret_reversal_5d", "limit_dist_pct"]
        return {
            "board": board,
            "aug_rows": len(aug),
            "ckpt_cols": len(ckpt_cols),
            "fe_out_cols_kept": int((~aug[ckpt_cols].isna().all()).sum()),
            "pool_na": [c for c in pool if c in ckpt_cols and aug[c].isna().any()],
            "top_na_cols": list(na_cols.index[:10]),
            "sample_syms": list(aug["symbol"].head(3)),
        }

    ckpt = pd.read_parquet(ckpt_path)
    before = len(ckpt)
    new = pd.concat([ckpt, aug], ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del ckpt
    gc.collect()
    new.to_parquet(ckpt_path, index=False)
    return {
        "board": board,
        "before_rows": before,
        "aug_rows": len(aug),
        "after_rows": len(new),
        "latest": new["date"].max().strftime("%Y-%m-%d"),
        "cols": len(new.columns),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=400, help="面板窗口日历天数 (默认 400 ≈ 270 交易日)")
    ap.add_argument("--limit-syms", type=int, default=0, help=">0 验证模式: 只取前 N 符号, 不写回")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    logf = os.path.join("data", f"_append_latest_{ts}.log")
    log = open(logf, "a", encoding="utf-8")

    def emit(msg: str):
        print(msg, flush=True)
        print(msg, file=log, flush=True)

    emit(f"[{time.strftime('%H:%M:%S')}] 增量 append 检查点 | window-days={args.window_days} "
         f"limit-syms={args.limit_syms or 'all'}")
    t0 = time.time()

    latest = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].max()
    cutoff = latest - pd.DateOffset(days=args.window_days)
    emit(f"[panel] latest={latest:%Y-%m-%d} cutoff={cutoff.date()}")

    panel = pd.read_parquet(PANEL_V3_PATH)
    emit(f"[panel] 已读 {len(panel):,} 行 ({time.time()-t0:.0f}s)")

    for ckpt_path, board in ((MAIN_CHECKPOINT, "main"), (DUAL_CHECKPOINT, "dual")):
        try:
            r = process_board(
                panel, ckpt_path, board, latest, cutoff,
                limit_syms=args.limit_syms, write=(args.limit_syms == 0),
            )
        except Exception as e:
            import traceback

            emit(f"[{board}] FAIL: {e}")
            traceback.print_exc(file=log)
            r = {"board": board, "error": str(e)}
        emit(f"[{board}] {r}")

    del panel
    gc.collect()
    emit(f"[done] {time.time()-t0:.0f}s | log={logf}")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
