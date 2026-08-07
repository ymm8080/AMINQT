"""诊断: 定位 301326/300911 在特征构建中消失的维度 (逐 dim 打点).

只跑双创侧 (cross_sectional_rank=True), 与生产推理路径一致.
用法: python scripts/_dbg_trace_dim_drop.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from config.settings import PANEL_V3_PATH

WATCH = {"301326", "300911"}


def _watch_present(df):
    if "symbol" not in df.columns:
        return "?"
    s = df["symbol"].astype(str).str.zfill(6)
    return f"{sorted(WATCH & set(s))}"


def main():
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= dates[-300]]
    cleaner = CleaningPipeline()
    _, dual_df, _ = cleaner.run_inference(panel)
    print(
        f"[clean] dual={len(dual_df):,} watch={_watch_present(dual_df)} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    print(f"[sym dtype] {dual_df['symbol'].dtype}", flush=True)

    fe = FeatureEngineV35()
    fe.float_shares_map = None

    class _T:
        def __init__(self, obj, nm, static):
            self._obj = obj
            self._nm = nm
            self._static = static

        def __call__(self, df, *a, **k):
            t = time.time()
            if self._static:
                out = self._obj(df, *a, **k)
            else:
                out = self._obj(fe, df, *a, **k)
            print(
                f"[{self._nm}] rows={len(out):,} watch={_watch_present(out)} ({time.time() - t:.0f}s)",
                flush=True,
            )
            return out

    for nm in dir(fe):
        if nm.startswith("dim"):
            obj = getattr(type(fe), nm)
            static = isinstance(vars(type(fe)).get(nm), staticmethod)
            setattr(fe, nm, _T(obj, nm, static))
    for nm in (
        "_add_time_series_changes",
        "_add_cross_sectional_ranks",
        "industry_neutralize",
        "add_missingness_flags",
    ):
        if hasattr(fe, nm):
            obj = getattr(type(fe), nm)
            static = isinstance(vars(type(fe)).get(nm), staticmethod)
            setattr(fe, nm, _T(obj, nm, static))

    # 从 bundle 取 dual 的 feature_cols (与生产推理一致)
    import pickle

    bundle = pickle.load(open("models/pipeline1/dual_current.pkl", "rb"))
    dual_cols = bundle["feature_cols"]
    print(f"[dual feature_cols] n={len(dual_cols)}", flush=True)

    out = fe.build(
        dual_df,
        float_shares_map=None,
        cross_sectional_rank=True,
        inference_cols=dual_cols,
    )
    print(
        f"[build out] rows={len(out):,} watch={_watch_present(out)} total {time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
