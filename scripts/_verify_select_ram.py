# -*- coding: utf-8 -*-
"""验证: legacy 周频 select (train_runner.select_features, MAIN) 的峰值内存.

用法:
    python scripts/_verify_select_ram.py

打印每个阶段的峰值 RSS (物理内存) 与峰值 commit (VMS ≈ 分页池, 对应 ~44GB 上限).
成功标准: 峰值 commit 远低于本机 commit 上限, 且 15.8GB 物理内存下不重抖;
选中特征数与当前 selected_main_current.json (1066) 同量级.
"""
import logging
import os
import sys
import threading
import time

import pandas as pd
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.WARNING, format="%(levelname)s - %(message)s")

from config.settings import PANEL_V3_PATH, data_others_path

from app.pipeline1.feature_selector import FeatureSelector
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.train_runner import CleaningPipeline, prepare_board_frame, select_features


def _fmt(gb):
    return f"{gb / 2**30:.1f}GB"


def main():
    proc = psutil.Process()
    peak = {"rss": 0, "vms": 0}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                mi = proc.memory_info()
            except Exception:
                break
            peak["rss"] = max(peak["rss"], mi.rss)
            peak["vms"] = max(peak["vms"], mi.vms)
            time.sleep(0.2)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t0 = time.time()

    def report(stage):
        print(
            f"[{stage:<8}] rss={_fmt(peak['rss'])} commit={_fmt(peak['vms'])} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    report(f"panel {len(panel):,}r/{panel['symbol'].nunique():,}s")

    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel)
    del dual_df
    report(f"main_df {len(main_df):,}r/{main_df['symbol'].nunique():,}s")

    registry = None
    reg_file = os.path.join(str(data_others_path("data/factor_registry")), "feature_registry.json")
    if os.path.exists(reg_file):
        registry = FeatureRegistry(path=reg_file)
    features = FeatureEngineV35()
    df = prepare_board_frame(main_df, features, cross_sectional_rank=False, registry=registry)
    del main_df
    report(f"prepared {len(df):,}r/{df.shape[1]}c")

    selector = FeatureSelector()
    cols, aug = select_features(df, "main", "RAM_verify", selector=selector)
    del aug
    report(f"SELECTED {len(cols)} feats")

    stop.set()
    t.join(timeout=2)
    print(
        f"\nPEAK rss={_fmt(peak['rss'])} commit={_fmt(peak['vms'])} "
        f"selected={len(cols)} total={time.time() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
