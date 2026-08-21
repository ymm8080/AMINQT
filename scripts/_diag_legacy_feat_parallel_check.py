"""_diag_legacy_feat_parallel_check.py — 真面板 串行 vs 并行 特征构建对照 (2026-08-21).

一次性验证 (LEGACY_PARALLEL_FEATURES 开启前执行):
  1. 真面板 300d 切片 + 生产清洗 (pool_blend=True) → main/dual 帧
  2. 串行双板构建 (计时) → 落盘参照
  3. 并行双板构建 (计时)
  4. 逐字节比较 + 计时表 → 开启开关的裁决依据

内存: 串行完成后立即释放特征帧再跑并行 (双板并行峰值 = 串行峰值 + 一个 worker).
用法: python scripts/_diag_legacy_feat_parallel_check.py
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain
from config.settings import PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}


def main() -> int:
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(panel["date"].unique())
    cut = dates[-300]
    panel = panel[panel["date"] >= cut]
    print(f"[panel] {len(panel):,}r {cut.date()}.. ({time.time() - t0:.0f}s)", flush=True)

    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    main_df, dual_df, _ = pipe.cleaner.run_inference(panel, pool_blend=True)
    del panel
    gc.collect()
    print(
        f"[clean] main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    main_cols = pipe.predictor.bundles["main"]["feature_cols"]
    dual_cols = pipe.predictor.bundles["dual"]["feature_cols"]
    print(f"[cols] main={len(main_cols)} dual={len(dual_cols)}", flush=True)

    tmp = tempfile.mkdtemp(prefix="feat_check_")
    try:
        # ---- 串行 (参照) ----
        ts = time.time()
        feat_m, feat_d = pipe._build_features_serial(main_df, dual_df, main_cols, dual_cols)
        t_serial = time.time() - ts
        feat_m.to_parquet(os.path.join(tmp, "serial_main.parquet"), index=False)
        feat_d.to_parquet(os.path.join(tmp, "serial_dual.parquet"), index=False)
        del feat_m, feat_d
        gc.collect()
        print(f"[serial] main+dual {t_serial:.0f}s ({time.time() - t0:.0f}s)", flush=True)

        # ---- 并行 ----
        ts = time.time()
        p_m, p_d = pipe._build_features_parallel(main_df, dual_df, main_cols, dual_cols)
        t_par = time.time() - ts
        print(f"[parallel] {t_par:.0f}s ({time.time() - t0:.0f}s)", flush=True)

        # ---- 逐字节比较 ----
        ref_m = pd.read_parquet(os.path.join(tmp, "serial_main.parquet"))
        ref_d = pd.read_parquet(os.path.join(tmp, "serial_dual.parquet"))
        ok = True
        for board, got, ref in (("main", p_m, ref_m), ("dual", p_d, ref_d)):
            try:
                pd.testing.assert_frame_equal(got, ref, check_exact=True)
                print(f"[compare] {board}: 逐字节一致 ({len(got):,}r x {len(got.columns)}c)", flush=True)
            except AssertionError as exc:
                ok = False
                print(f"[compare] {board}: 不一致! {exc}", flush=True)

        print(
            f"\n[verdict] serial={t_serial:.0f}s parallel={t_par:.0f}s "
            f"speedup={t_serial / max(t_par, 1e-9):.2f}x "
            f"identical={'是' if ok else '否'}"
        )
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
