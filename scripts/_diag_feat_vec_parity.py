"""_diag_feat_vec_parity.py — 真面板 暴力特征 逐symbol参考 vs 向量化内核 逐字节对照 (2026-08-25).

背景: 20260825 重训 main feature selection 4h10m, 其中 BruteForce 家族统计
3h38m = 逐 symbol Python 循环 (20260825 日志实测). feature_selector 新增
_family_columns_vec 向量化内核后, 必须证明与旧逐 symbol 数学
(_family_for_symbol, 2026-08-25 前生产路径) 逐字节一致, 否则 dedup 相关性/
方差漂移 → 选择结果改变 → 模型质量回归.

方法: 真面板 main 板 150d 切片 → 生产清洗 (run_train) + 特征构建
(prepare_board_frame) → 每族三重对照:
  ① generate 级: 旧逐 symbol 物化宽帧 vs 内核列流式帧, check_exact=True
  ② stats 级: 旧帧逐列 nan率 + 5000 采样行值 vs 新 family_stats
  ③ need 级: 新 generate_columns(need) vs 旧帧 need 子集 (生产后注入路径)

内存: 同一时间只驻留一族旧宽帧 (pct 族 150d ≈ 3.2GB) + 一列新值.
用法: python scripts/_diag_feat_vec_parity.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH

N_DAYS = 150


def old_family_frame(gen, df, fam, raw):
    """2026-08-25 前生产路径 (generate_family 旧体): 逐 symbol _family_for_symbol
    → concat → inf→nan. 与 git 历史逐字节一致, 作对照 oracle."""
    fd = gen.transforms[fam]
    windows, suffix = fd["windows"], fd["suffix"]
    parts = []
    for _sym, g in df.groupby("symbol"):
        g, feats = gen._family_for_symbol(g, raw, fam, windows, suffix)
        parts.append(pd.DataFrame(feats, index=g.index))
    return pd.concat(parts).replace([np.inf, -np.inf], np.nan)


def main() -> int:
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(panel["date"].unique())
    cut = dates[-N_DAYS]
    panel = panel[panel["date"] >= cut]
    print(f"[panel] {len(panel):,}r {cut.date()}.. ({time.time() - t0:.0f}s)", flush=True)

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    board_dfs = dict(zip(("main", "dual"), cleaner.run_train(panel, board="main")))
    del panel
    gc.collect()
    main_df = board_dfs["main"]
    print(f"[clean] main={len(main_df):,}r ({time.time() - t0:.0f}s)", flush=True)

    df = prepare_board_frame(main_df, features, None, cross_sectional_rank=False)
    del main_df, board_dfs
    gc.collect()
    print(
        f"[feat] main={len(df):,}r x {df.shape[1]}c ({time.time() - t0:.0f}s)",
        flush=True,
    )

    gen = BruteForceGenerator()
    raw = gen._eligible(df)
    print(f"[raw] {len(raw)} eligible cols", flush=True)
    sample_pos = df.sample(min(5000, len(df)), random_state=42).index

    ok = True
    totals = {"old": 0.0, "new": 0.0}
    for fam in BRUTE_FAMILIES:
        fd = gen.transforms[fam]
        windows, suffix = fd["windows"], fd["suffix"]

        ts = time.time()
        old = old_family_frame(gen, df, fam, raw)
        t_old = time.time() - ts

        ts = time.time()
        data = dict(gen._family_columns_vec(df, fam, raw, windows, suffix))
        new = pd.DataFrame(data, index=df.index).replace([np.inf, -np.inf], np.nan)
        t_new_stats = 0.0
        t_new = time.time() - ts
        totals["old"] += t_old
        totals["new"] += t_new

        # ① generate 级逐字节
        try:
            pd.testing.assert_frame_equal(
                new.loc[:, old.columns], old.reindex(new.index), check_exact=True
            )
            r1 = "逐字节一致"
        except AssertionError as exc:
            ok = False
            r1 = f"不一致! {str(exc)[:200]}"

        # ② stats 级: 新 family_stats vs 旧帧统计
        try:
            ts = time.time()
            cols, nan_rate, svals = gen.family_stats(
                df, fam, sample_pos, raw_cols=raw, dtype="float32"
            )
            t_new_stats = time.time() - ts
            assert list(cols) == old.columns.tolist(), "stats 列清单/列序漂移"
            for c in cols:
                assert nan_rate[c] == float(old[c].isna().mean()), f"{c}: nan率漂移"
                np.testing.assert_array_equal(
                    svals[c],
                    old.loc[sample_pos, c].to_numpy(np.float32),
                    err_msg=f"{c}: 采样行值漂移",
                )
            r2 = "一致"
        except (AssertionError, ValueError) as exc:
            ok = False
            r2 = f"不一致! {str(exc)[:200]}"

        # ③ need 级: 新 generate_columns vs 旧帧子集 (生产后注入路径)
        names = old.columns.tolist()
        need = set(names[:5] + names[-3:])
        try:
            gc_new = gen.generate_columns(df, fam, need, raw_cols=raw)
            assert gc_new is not None, "need 有交集族被误短路"
            subset = old.loc[gc_new.index, gc_new.columns]
            pd.testing.assert_frame_equal(gc_new, subset, check_exact=True)
            r3 = "一致"
        except AssertionError as exc:
            ok = False
            r3 = f"不一致! {str(exc)[:200]}"

        print(
            f"[{fam}] {len(names)} cols | old={t_old:.0f}s new={t_new:.0f}s"
            f" (stats {t_new_stats:.0f}s) speedup={t_old / max(t_new, 1e-9):.1f}x"
            f" | gen:{r1} stats:{r2} need:{r3}",
            flush=True,
        )
        del old, new, data
        gc.collect()

    print(
        f"\n[verdict] old_total={totals['old']:.0f}s new_total={totals['new']:.0f}s "
        f"speedup={totals['old'] / max(totals['new'], 1e-9):.1f}x "
        f"identical={'是' if ok else '否'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
