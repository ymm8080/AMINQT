"""全市场宇宙修复 Step 4: 新股票 CYQ 筹码扩展 8 列 (2026-08-15).

复用生产机器 cyq_ext.compute_cyq_panel (与面板既有 chip 列同算法同参数
FACTOR=150/RANGE_DAYS=120), 对新股票 base 面板全史计算, 只合并 TARGET_COLS 8 列:
peak_price/chip_entropy/chip_skew_dist/chip_gini/resistance_dist/support_dist/
peak_roc_5d/peak_roc_20d.

turnover_rate NaN → 0 对齐 _daily_fetch.py 的 cyq 前置处理 (NaN 会毒化筹码分布).
预计 ~1-2 小时 (1042 股 30-60 分钟基线), 后台运行.

WORM: data/new_symbols_panel/base_new_cyq_<ts>.parquet (base + 8 列)
"""

from __future__ import annotations

import glob
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.cyq_ext import TARGET_COLS, compute_cyq_panel  # noqa: E402

OUT_PANEL_DIR = "data/new_symbols_panel"


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_*.parquet")))[-1]
    df = pd.read_parquet(f)
    print(
        f"[cyq] input={f} rows={len(df):,} symbols={df['symbol'].nunique()}", flush=True
    )
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    cyq_in = df[
        ["symbol", "date", "open", "high", "low", "close", "turnover_rate"]
    ].copy()
    cyq_in["turnover_rate"] = cyq_in["turnover_rate"].fillna(0.0)

    t0 = time.time()
    cyq = compute_cyq_panel(cyq_in)
    mins = (time.time() - t0) / 60
    print(f"[cyq] compute done in {mins:.1f} min, rows={len(cyq):,}", flush=True)

    merge_cols = ["symbol", "date"] + [c for c in TARGET_COLS if c in cyq.columns]
    df = df.merge(cyq[merge_cols], on=["symbol", "date"], how="left")
    missing = [c for c in TARGET_COLS if c not in cyq.columns]
    if missing:
        print(f"[cyq] WARN: missing TARGET_COLS in output: {missing}", flush=True)

    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_cyq_{ts_}.parquet")
    df.to_parquet(out, index=False)
    cov = df[TARGET_COLS].notna().mean().round(3)
    print(f"[save] {out}", flush=True)
    print("[coverage]")
    print(cov.to_string(), flush=True)
    print("CYQ BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
