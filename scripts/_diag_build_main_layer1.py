# -*- coding: utf-8 -*-
"""_diag_build_main_layer1.py — 重建 main Layer1 特征 (仅 step2, 不动 registry).

等价 build_features.py --board main 的 step2 部分, 但不跑 step1 registry adoption
(避免改 feature_registry.json). 输出 data/factor_registry/features_main_{ts}.parquet (WORM).
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import PANEL_V3_PATH

import build_features as bf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    # build_features.PANEL_PATH 硬编码为本地 data/, 用 config 真值覆盖 (生产路径).
    bf.PANEL_PATH = str(PANEL_V3_PATH)
    t0 = time.time()
    panel = bf.load_panel(None)
    print(
        f"[load] panel {len(panel):,}r {panel['symbol'].nunique()} stocks "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    path = bf.step2_build_board(panel, board="main", window="3Y")
    print(f"[done] {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
