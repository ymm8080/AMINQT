# -*- coding: utf-8 -*-
"""候选筹码形态列的数据报告: chip_entropy / chip_skew_dist / peak_roc_5d/20d / peak_mass.

对 main/dual 的 July OOS 帧, 输出每个候选列的:
  1. 值分布统计 (count/mean/std/min/p1/p10/p50/p90/p99/max)
  2. 逐日 rank IC (时间均值 + 正天数占比 + N)
  3. 十分位表: 按列排序分桶 → 桶内 label_pm_3d_net 均值 (信号单调性/方向)

OOS 帧缓存到 data/_ab_cyq_models/_oos_{board}.parquet, 复用免重建.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import _verify_cyq_drop as V  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("data_chip_cols")

CANDIDATES = [
    "chip_entropy",
    "chip_skew_dist",
    "peak_roc_5d",
    "peak_roc_20d",
    "peak_mass",
]
LABEL = V.LABEL


def load_oos(board: str) -> pd.DataFrame:
    path = os.path.join(V.MODEL_DIR, f"_oos_{board}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    logger.info("构建 OOS 帧 [%s] ...", board)
    df = V.build_oos_frame(
        pd.read_parquet(os.path.join(V.MODEL_DIR, "_panel_c_full.parquet")), board
    )
    df.to_parquet(path, index=False)
    return df


def stats_table(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    q = s.quantile([0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "n": int(s.count()),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
        "p1": q[0.01],
        "p10": q[0.10],
        "p50": q[0.50],
        "p90": q[0.90],
        "p99": q[0.99],
        "max": s.max(),
    }


def daily_ic(df: pd.DataFrame, col: str) -> dict:
    t = df.dropna(subset=[col, LABEL]).copy()
    ics: list[tuple[pd.Timestamp, float]] = []
    for d, g in t.groupby("date"):
        c = g[col].corr(g[LABEL], method="spearman")
        if pd.notna(c):
            ics.append((d, float(c)))
    ic_by_day = pd.Series(dict(ics))
    return {
        "n_days": int(len(ic_by_day)),
        "mean_ic": float(ic_by_day.mean()) if len(ic_by_day) else 0.0,
        "pos_frac": float((ic_by_day > 0).mean()) if len(ic_by_day) else 0.0,
        "min_day": float(ic_by_day.min()) if len(ic_by_day) else 0.0,
        "max_day": float(ic_by_day.max()) if len(ic_by_day) else 0.0,
    }


def decile_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = df.dropna(subset=[col, LABEL]).copy()
    t["_dec"] = t.groupby("date")[col].transform(
        lambda g: pd.qcut(g.rank(method="first"), 10, labels=False) + 1
    )
    agg = (
        t.groupby("_dec", observed=True)
        .agg(mean_label=(LABEL, "mean"), mean_feat=(col, "mean"), n=(LABEL, "size"))
        .reset_index()
    )
    agg["_dec"] = agg["_dec"].astype(int)
    return agg


def main() -> None:
    frames = {b: load_oos(b) for b in ("main", "dual")}
    for col in CANDIDATES:
        print(f"\n{'='*72}\n### {col}\n{'='*72}")
        for b, df in frames.items():
            if col not in df.columns:
                print(f"[{b}] 列不存在")
                continue
            st = stats_table(df, col)
            ic = daily_ic(df, col)
            print(f"\n-- [{b}] --")
            print(
                f"值分布: n={st['n']} mean={st['mean']:.4g} std={st['std']:.4g} "
                f"min={st['min']:.4g} p1={st['p1']:.4g} p10={st['p10']:.4g} "
                f"p50={st['p50']:.4g} p90={st['p90']:.4g} p99={st['p99']:.4g} max={st['max']:.4g}"
            )
            print(
                f"逐日 rank IC: n_days={ic['n_days']} mean={ic['mean_ic']:+.4f} "
                f"pos_frac={ic['pos_frac']:.2f} min={ic['min_day']:+.4f} max={ic['max_day']:+.4f}"
            )
            dec = decile_table(df, col)
            row = "  ".join(f"D{d:.0f}={r['mean_label']*100:+.2f}%" for d, r in dec.iterrows())
            print(f"十分位 → 3d净收益: {row}")
            spread = dec["mean_label"].iloc[-1] - dec["mean_label"].iloc[0]
            print(f"  D10-D1 spread = {spread*100:+.2f}pp")


if __name__ == "__main__":
    main()
