# -*- coding: utf-8 -*-
"""_baseline_check_topn.py — 验收基准校验 (2026-08-04).

7/473 特征在 (5,10) 视界通过. 关键问题: 若整个 3y 窗口无条件 10d 净收益
胜率本身 ~60% (市场上涨), 则 TOP-10 高胜率无价值. 本脚本:
  1. 快速路径重建行集 (复用检查点 + 补 10d 标签);
  2. 算无条件基准: 全截面 / 每股 / 每日期截面 的 5d/10d 净收益 均值+胜率;
  3. 对通过的特征重算 每日期 TOP-10 的 5d/10d 幅度+胜率, 与基准对比.
输出: 对比表 + 判定 (TOP-10 是否显著跑赢基准).
"""

import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from scripts._reclassify_all_features import (
    MAIN_CHECKPOINT,
    DUAL_CHECKPOINT,
    _finalize_slice,
)
from scripts._measure_topn import HORIZONS, LABEL_COLS

PASSED = [
    "VAR51",
    "amihud_illiq",
    "amihud_illiquidity",
    "down_gap_pct",
    "limit_dist_pct",
    "ret_reversal_5d",
    "small_mv_premium",
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # ── 1. 快速路径重建行集 ──
    slices = []
    for ckpt in (MAIN_CHECKPOINT, DUAL_CHECKPOINT):
        df = _finalize_slice(pd.read_parquet(ckpt))
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    print(
        f"行集 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"latest={work['date'].max():%Y-%m-%d}",
        flush=True,
    )

    for k in HORIZONS:
        lab = LABEL_COLS[k]
        v = work[lab].dropna()

        # 无条件基准: 全截面
        overall_mag = float(v.mean())
        overall_wr = float((v > 0).mean())

        # 每日期截面均值/胜率的均值 (等权日期)
        g = work.dropna(subset=[lab]).groupby("date")[lab]
        date_mag = g.mean().mean()
        date_wr = (g.mean() > 0).mean()

        print(f"\n=== 无条件基准 T+{k} ({lab}) ===", flush=True)
        print(
            f"  全截面:   幅度={overall_mag:+.4f} 胜率={overall_wr:.1%} n={len(v):,}",
            flush=True,
        )
        print(
            f"  日期等权: 幅度={date_mag:+.4f} 胜率={date_wr:.1%} (平均每日期截面均值为正的天数占比)",
            flush=True,
        )

    # ── 2. 通过特征 TOP-10 vs 基准 ──
    print(
        f"\n{'=' * 100}\n  通过特征 每日期TOP-10 vs 无条件基准 (高值端)\n{'=' * 100}",
        flush=True,
    )
    for col in PASSED:
        if col not in work.columns:
            print(f"  {col:<26} (列缺失)", flush=True)
            continue
        base = work[
            ["symbol", "date", col]
            + [LABEL_COLS[k] for k in HORIZONS if LABEL_COLS[k] in work.columns]
        ]
        base = base.dropna(subset=[col])
        top = (
            base.sort_values([col], ascending=False)
            .groupby("date", group_keys=False)
            .head(10)
        )
        row = f"  {col:<26}"
        for k in HORIZONS:
            v = top[LABEL_COLS[k]].dropna()
            if len(v) < 5:
                row += f"  T+{k}: n<5 "
                continue
            mag = float(v.mean())
            wr = float((v > 0).mean())
            base_mag, base_wr = (
                float(work[LABEL_COLS[k]].dropna().mean()),
                float((work[LABEL_COLS[k]].dropna() > 0).mean()),
            )
            mag - base_mag
            wr - base_wr
            row += (
                f"  T+{k} mag={mag:+.2%}(Δ{base_mag:+.2%}) wr={wr:.1%}(Δ{base_wr:+.1%})"
            )
        print(row, flush=True)


if __name__ == "__main__":
    main()
