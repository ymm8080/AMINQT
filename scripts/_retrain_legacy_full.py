# -*- coding: utf-8 -*-
"""全量周频重训 (V3 面板直读, 含当日) — 2026-08-05 用户确认跑全量.

与 run_weekly_retrain.py 的区别: 训练面板直接读 PANEL_V3_PATH (已含当日, 免 akshare
网络装配), 对齐 3 年周频窗口. OOS IC 过闸才把 bundle 发布为 current (镜像 weekly 语义),
current_meta.json 同步更新.

用法: python scripts/_retrain_legacy_full.py [tag]
"""
from __future__ import annotations

import gc
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.train_runner import run_training

MODEL_DIR = "models/pipeline1"


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
          f"({time.time()-t0:.0f}s)", flush=True)
    # 对齐周频 3y 训练窗口 (assemble_panel years=3 语义: 截止日前 3 个日历年)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(f"[slice] {cut.date()}.. {panel['date'].max():%Y-%m-%d} "
          f"-> {len(panel):,}r ({time.time()-t0:.0f}s)", flush=True)

    results = run_training(panel, tag, model_dir=MODEL_DIR)
    del panel
    gc.collect()

    from app.pipeline1.model_meta import load_modules, save_modules

    mods = load_modules()
    for board, res in results.items():
        if res["switched"]:
            cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
            bak = os.path.join(MODEL_DIR, f"{board}_current_retrain_backup.pkl")
            if os.path.exists(cur) and not os.path.exists(bak):
                shutil.copy(cur, bak)
                print(f"[{board}] 旧 current 备份 -> {bak}", flush=True)
            shutil.copy(res["path"], cur)
            mods[board] = {
                "tag": tag,
                "file": os.path.basename(res["path"]),
                "updated": time.strftime("%Y-%m-%d %H:%M"),
            }
            print(f"[{board}] switched -> current = {res['path']}", flush=True)
        else:
            print(f"[{board}] OOS weighted_IC={res['oos'].get('weighted_ic'):.4f} "
                  f"< {res['oos'].get('threshold', '?')}, 保留旧模型", flush=True)
    save_modules(mods)
    print(f"[meta] current_meta.json = {mods}", flush=True)
    print(f"[done] 全部完成 ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
