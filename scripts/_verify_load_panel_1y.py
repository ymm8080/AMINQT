# -*- coding: utf-8 -*-
"""验证: 生产 load_panel() (新 window_days=242 过滤) 的验收判定 = 消融已落盘 1y 结果 (108/108)."""
import glob
import gc
import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import load_panel
from app.pipeline_parallel.config import OOS_WINDOWS, SYSTEMS, BOARD_THRESHOLDS
from scripts._ablate_train_window_quality import accept_board

JSON = glob.glob("data/_ablate_train_window_quality_*.json")[-1]
ab = json.load(open(JSON, encoding="utf-8"))
print(f"对照: {JSON}")

t0 = time.time()
tracemalloc.start()
work = load_panel()
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"load_panel: rows={len(work):,} stocks={work['symbol'].nunique():,} "
      f"latest={work['date'].max():%Y-%m-%d} 交易日={work['date'].nunique():,} "
      f"cols={work.shape[1]:,} | 峰值内存 {peak/1e9:.2f}GB | {time.time()-t0:.0f}s")

# 用消融的 accept_board 复算 1y 判定, 与 JSON 的 "1y" 逐格对比
n_total = n_match = 0
mismatch = []
for board in ("main", "dual"):
    bcrit = (BOARD_THRESHOLDS[board]["min_winrate"], BOARD_THRESHOLDS[board]["min_mag"])
    res = accept_board(work, board, bcrit)
    for sname, spec in SYSTEMS.items():
        if not spec.enabled:
            continue
        for lab, d in OOS_WINDOWS.items():
            for kind, tn in (("primary", spec.top_n), ("alt", spec.top_n_alt)):
                cell = f"{lab}/{kind}"
                cur_cell = res[sname].get(cell, {})
                ref_cell = ab["boards"][board]["1y"][sname].get(cell, {})
                for hz in spec.horizons:
                    cur_h = cur_cell.get(hz, {})
                    ref_h = ref_cell.get(hz, {})
                    n_total += 1
                    def _eq(a, b):
                        # nan==nan 视为相等 (全 NaN 单元格)
                        if isinstance(a, float) and isinstance(b, float) and (
                                np.isnan(a) and np.isnan(b)):
                            return True
                        if a is None and b is None:
                            return True
                        return a == b
                    same = (
                        _eq(cur_h.get("ok"), ref_h.get("ok"))
                        and _eq(cur_h.get("n"), ref_h.get("n"))
                        and _eq(cur_h.get("winrate"), ref_h.get("winrate"))
                        and _eq(cur_h.get("mag"), ref_h.get("mag"))
                    )
                    if same:
                        n_match += 1
                    else:
                        mismatch.append((board, sname, cell, hz, cur_h, ref_h))

print(f"\n验收判定对比: {n_match}/{n_total} 格一致, 不一致 {len(mismatch)}")
for m in mismatch[:15]:
    print("  MISMATCH", m)
# 内存对比参照 (消融 3y 全量 ~5.35GB): 这里只量 1y 峰值
print(f"结论: load_panel 已按 window_days=242 过滤; 判定与消融 1y {'完全一致' if not mismatch else '存在差异!'}")
