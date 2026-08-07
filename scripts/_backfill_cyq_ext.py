"""V3 CYQ 列整理: 删除 7 个 DELETE 基础列 + 补齐 5 个筹码形态目标列
(peak_price/chip_entropy/chip_skew_dist/peak_roc_5d/peak_roc_20d).

用法:
  python scripts/_backfill_cyq_ext.py --slice 3   # 3 只股票切片, 只验证不写盘
  python scripts/_backfill_cyq_ext.py             # 全量 (扩展列缺失时 30-60 min), 原子写回
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from app.pipeline1 import cyq_ext
from config.settings import CYQ_BASE_DELETE, PANEL_V3_PATH

NEEDED = ["symbol", "date", "open", "high", "low", "close", "turnover_rate"]


def _compute_panel_parallel(panel_slim: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    """按 symbol 分片并行计算扩展 CYQ.

    根因: compute_cyq_panel 逐股 Python 循环单核跑 ~7h+. 这里按 symbol 分片,
    每个 worker 只拿自己的切片, 避免把 6GB 全量面板反复 pickle 传输.
    """
    if n_workers <= 1:
        return cyq_ext.compute_cyq_panel(panel_slim)
    syms = sorted(panel_slim["symbol"].unique())
    chunks = [syms[i::n_workers] for i in range(n_workers)]
    chunk_frames = [
        panel_slim[panel_slim["symbol"].isin(s)].copy() for s in chunks if len(s)
    ]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunk_frames)) as ex:
        parts = list(ex.map(cyq_ext.compute_cyq_panel, chunk_frames))
    out = pd.concat(parts, ignore_index=True)
    print(f"    并行计算完成 ({time.time() - t0:.0f}s, {len(chunk_frames)} workers)")
    return out


def main() -> None:
    slice_n = None
    if "--slice" in sys.argv:
        slice_n = int(sys.argv[sys.argv.index("--slice") + 1])
    parallel = 1
    if "--parallel" in sys.argv:
        parallel = int(sys.argv[sys.argv.index("--parallel") + 1])

    panel_path = str(PANEL_V3_PATH)
    print(f"[1] 读取规范面板: {panel_path}")
    panel = pd.read_parquet(panel_path)
    missing = [c for c in NEEDED if c not in panel.columns]
    if missing:
        raise SystemExit(f"面板缺列: {missing}")
    if slice_n:
        syms = sorted(panel["symbol"].unique())[:slice_n]
        panel = panel[panel["symbol"].isin(syms)]
        print(f"[slice] 仅 {slice_n} 只股票: {len(panel)} 行")

    # V3 删列 (2026-08-02): 无条件删除 DELETE 基础列
    drop = [c for c in CYQ_BASE_DELETE if c in panel.columns]
    if drop:
        print(f"[1b] 删除 DELETE 基础列 ({len(drop)}): {drop}")
        panel = panel.drop(columns=drop)
    else:
        print("[1b] 无 DELETE 基础列需删除")

    missing_ext = [c for c in cyq_ext.TARGET_COLS if c not in panel.columns]
    if missing_ext:
        t0 = time.time()
        print(f"[2] 计算扩展列 (缺 {missing_ext}) ...")
        panel_slim = panel[NEEDED]
        cyq = _compute_panel_parallel(panel_slim, parallel)
        del panel_slim
        # 只 merge 缺失列, 避免对已有列产生 _x/_y 孪生
        cyq = cyq[["symbol", "date"] + missing_ext]
        print(f"    cyq 计算完成 ({time.time() - t0:.0f}s), {len(cyq)} 行")
        panel = panel.merge(cyq, on=["symbol", "date"], how="left")
        del cyq
    else:
        print("[2] 扩展列已存在, 跳过计算")

    n = len(panel)
    for c in cyq_ext.TARGET_COLS:
        if c in panel.columns:
            nn = panel[c].notna().sum()
            print(f"    {c}: {nn}/{n} ({nn / n * 100:.1f}%)")

    if slice_n:
        print("[slice] 切片模式, 不写盘. 完成.")
        return

    print("[4] 原子写回 ...")
    tmp = panel_path + ".tmp"
    panel.to_parquet(tmp, index=False)
    if os.path.exists(panel_path):
        os.remove(panel_path)
    os.rename(tmp, panel_path)
    print(f"    已写回: {panel_path} ({len(panel)} 行, {len(panel.columns)} 列)")
    print("[done]")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
