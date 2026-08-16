"""全市场宇宙修复 Step 5d: 删除新面板多余列 (2026-08-16).

base 面板从 supply_cache 带入 5 列生产面板没有的列
(vol/adj_factor/turnover_rate_f/winner_rate/avg_cost) — schema 强校验会拒绝合并.
生产 120 列集为准, 新面板只保留生产列 (缺列会在 merge 前暴露).

WORM: 读最新 base_new_full_<ts>.parquet, 输出新 base_new_full_<new_ts>.parquet.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime

import pandas as pd

OUT_PANEL_DIR = "data/new_symbols_panel"
DROP = ["vol", "adj_factor", "turnover_rate_f", "winner_rate", "avg_cost"]


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))[-1]
    df = pd.read_parquet(f)
    missing = [c for c in DROP if c not in df.columns]
    present = [c for c in DROP if c in df.columns]
    df = df.drop(columns=present)
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_full_{ts_}.parquet")
    df.to_parquet(out, index=False)
    print(
        f"[drop] input={os.path.basename(f)} dropped={present} "
        f"missing_anyway={missing} → cols={len(df.columns)} → {out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
