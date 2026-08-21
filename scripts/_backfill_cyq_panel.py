"""_backfill_cyq_panel.py — 回填 cyq_panel.parquet 缺失日期 (2026-08-19).

cyq_panel.parquet 数据停在 2026-07-17: enrich_cyq 只检查 symbol 覆盖不检查 date
覆盖 (panel_builder.py 已修), 且每日自动化不调用 enrich_cyq → cache 永不自愈.

本脚本:
  1. 备份旧 cache (cyq_panel__<ts>.parquet)
  2. 从 V3 面板读 07-17 前 130 交易日起的尾部
     (symbol/date/open/high/low/close/turnover_rate)
  3. ProcessPoolExecutor 按 symbol 分片并行重算 (cyq_calculator.compute_cyq_panel)
  4. 只取 date > 07-17 的行 concat 旧 cache, 原子写回

口径已验证: V3 原始价列重算值与 cache 逐位一致 (300911 2026-07-10..07-17 全同).

用法: python scripts/_backfill_cyq_panel.py [--workers N]
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from app.pipeline1.cyq_calculator import compute_cyq_panel
from config.settings import PANEL_V3_PATH, data_others_path

CYQ_CACHE = Path("data/cyq_panel.parquet")
PREFIX_DAYS = 130  # > RANGE_DAYS=120, 保证补算日窗口完整
NEEDED = ["symbol", "date", "open", "high", "low", "close", "turnover_rate"]


def main() -> int:
    workers = 6
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])

    cache = pd.read_parquet(CYQ_CACHE)
    cache["date"] = pd.to_datetime(cache["date"])
    cache_max = cache["date"].max()
    cache_syms = set(cache["symbol"].unique())
    print(f"[0] cache: {len(cache)} 行 {len(cache_syms)} 只, 截至 {cache_max.date()}")

    # 计算 cutoff: cache_max 前 PREFIX_DAYS 个交易日
    dates = pd.to_datetime(
        pd.read_parquet(str(PANEL_V3_PATH), columns=["date"])["date"]
    )
    dates = dates.drop_duplicates().sort_values().reset_index(drop=True)
    i = dates.searchsorted(cache_max)
    start_date = dates[max(0, i - PREFIX_DAYS)]
    print(
        f"[1] 重算起点 {start_date.date()} (前缀 {i - dates.searchsorted(start_date)} 个交易日)"
    )

    panel = pd.read_parquet(
        str(PANEL_V3_PATH), columns=NEEDED, filters=[("date", ">=", start_date)]
    )
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"[2] 尾部面板: {len(panel)} 行 {panel['symbol'].nunique()} 只")

    # 只补 cache 已有 symbol 的尾部; V3 新 symbol 单独全量算
    tail = panel[panel["symbol"].isin(cache_syms)]
    new_syms = set(panel["symbol"].unique()) - cache_syms
    print(f"[3] 增量 {len(tail)} 行, 新 symbol {len(new_syms)} 只")

    syms = sorted(tail["symbol"].unique())
    chunks = [syms[i::workers] for i in range(workers)]
    chunk_frames = [tail[tail["symbol"].isin(s)].copy() for s in chunks if len(s)]
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunk_frames)) as ex:
        parts = list(ex.map(compute_cyq_panel, chunk_frames))
    new_cyq = pd.concat(parts, ignore_index=True)
    new_cyq = new_cyq[new_cyq["date"] > cache_max]
    print(f"[4] 增量计算完成 ({time.time() - t0:.0f}s, {len(new_cyq)} 行)")

    if new_syms:
        full = compute_cyq_panel(panel[panel["symbol"].isin(new_syms)])
        new_cyq = pd.concat([new_cyq, full], ignore_index=True)
        print(f"    新 symbol 全量: {len(full)} 行")

    if len(new_cyq) == 0:
        print("[5] 无新增行, 无需写回")
        return 0

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    backup = data_others_path("diag") / f"cyq_panel__{ts}.parquet"
    shutil.copy2(CYQ_CACHE, backup)
    print(f"[5] 备份旧 cache → {backup}")

    out = pd.concat([cache, new_cyq], ignore_index=True).drop_duplicates(
        ["symbol", "date"], keep="last"
    )
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    tmp = str(CYQ_CACHE) + ".tmp"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, CYQ_CACHE)
    print(f"[6] 原子写回: {len(out)} 行, 截至 {out['date'].max().date()}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
