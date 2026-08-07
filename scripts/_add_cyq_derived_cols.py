"""V3 面板落 3 个 CYQ 派生列 (cost_bias / conc_trend_20d / conc_90_industry_rank).

用户 2026-08-02 决策: 不回填 chip_gini/resistance_dist/support_dist, 只补 3 派生列,
使面板 CYQ 列 = 7 基础 + 5 扩展 + 3 派生 = 15.

公式与 dim21_chip_tushare (feature_engine_v35.py) 完全一致, 向量化 groupby:
  cost_bias             = (close_hfq - cost_50pct) / cost_50pct
  conc_trend_20d        = pct_90_con[t] / pct_90_con[t-20]   (per stock)
  conc_90_industry_rank = pct_90_con 按 date+industry 截面 rank(pct=True)
NaN 回退: cost_bias/conc_trend_20d 为 NaN 处用 OHLCV 代理 (close/close.rolling(60)-1 等) 填充.

用法:
  python scripts/_add_cyq_derived_cols.py --slice 200   # 200 只股票切片, 只验证不写盘
  python scripts/_add_cyq_derived_cols.py                # 全量, WORM 备份 + 原子写回
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH

NEW_COLS = ["cost_bias", "conc_trend_20d", "conc_90_industry_rank"]
ANCHOR = "peak_roc_20d"  # 新列插在其后, 使 CYQ 列聚在一起
NEEDED = [
    "symbol",
    "date",
    "close_hfq",
    "close",
    "volume",
    "turnover_rate",
    "cost_50pct",
    "pct_90_con",
    "industry",
]


def compute_derived(work: pd.DataFrame) -> pd.DataFrame:
    """在按 (symbol, date) 排序的工作框上计算 3 个派生列 (原地, 返回 work)."""
    g = work.groupby("symbol", sort=False)

    # 1) cost_bias = (close_hfq - cost_50pct) / cost_50pct
    work["cost_bias"] = (work["close_hfq"] - work["cost_50pct"]) / work[
        "cost_50pct"
    ].replace(0, np.nan)

    # 2) conc_trend_20d = pct_90_con[t] / pct_90_con[t-20] (per stock)
    prev = g["pct_90_con"].shift(20)
    work["conc_trend_20d"] = work["pct_90_con"] / prev.replace(0, np.nan)

    # 3) conc_90_industry_rank = pct_90_con 截面 rank (date+industry)
    work["conc_90_industry_rank"] = (
        work.groupby(["date", "industry"], observed=True)["pct_90_con"]
        .rank(pct=True)
        .fillna(0.5)
    )

    # NaN 回退 (对齐 dim21 per_stock_proxy): cost_bias/conc_trend_20d NaN 处填 OHLCV 代理
    g = work.groupby("symbol", sort=False)
    _std = g["turnover_rate"].rolling(20).std().reset_index(level=0, drop=True)
    _mean = g["turnover_rate"].rolling(20).mean().reset_index(level=0, drop=True)
    work["_conc90_proxy"] = 1 - _std / _mean.replace(0, 1)
    _c60 = g["close"].rolling(60).mean().reset_index(level=0, drop=True)
    work["_cost_bias_proxy"] = work["close"] / _c60 - 1
    work["_conc_trend_proxy"] = work["_conc90_proxy"] / g["_conc90_proxy"].shift(
        20
    ).replace(0, 1)
    for col, proxy in [
        ("cost_bias", "_cost_bias_proxy"),
        ("conc_trend_20d", "_conc_trend_proxy"),
    ]:
        mask = work[col].isna()
        work.loc[mask, col] = work.loc[mask, proxy]
    work.drop(
        columns=["_conc90_proxy", "_cost_bias_proxy", "_conc_trend_proxy"],
        inplace=True,
    )
    return work


def main() -> None:
    slice_n = None
    if "--slice" in sys.argv:
        slice_n = int(sys.argv[sys.argv.index("--slice") + 1])

    panel_path = str(PANEL_V3_PATH)
    print(f"[1] 读取面板: {panel_path}")
    df = pd.read_parquet(panel_path)
    if slice_n:
        syms = sorted(df["symbol"].unique())[:slice_n]
        df = df[df["symbol"].isin(syms)]
        print(f"[slice] 仅 {slice_n} 只股票: {len(df)} 行")

    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise SystemExit(f"面板缺列: {missing}")

    print("[2] 计算 3 个派生列 ...")
    t0 = time.time()
    work = df[NEEDED].copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    work = compute_derived(work)
    print(f"    计算完成 ({time.time() - t0:.0f}s)")
    n = len(work)
    for c in NEW_COLS:
        nn = work[c].notna().sum()
        print(f"    {c}: {nn}/{n} ({nn / n * 100:.1f}%)")

    if slice_n:
        print("[slice] 切片模式, 不写盘. 完成.")
        return

    print("[3] 合并回面板 (按 symbol+date, 保持原行序) ...")
    deriv = work.set_index(["symbol", "date"])[NEW_COLS]
    df = df.join(deriv, on=["symbol", "date"])

    # 列重排: 3 新列插到 ANCHOR 之后
    cols = list(df.columns)
    if ANCHOR in cols:
        i = cols.index(ANCHOR) + 1
        for c in NEW_COLS:
            cols.remove(c)
            cols.insert(i, c)
            i += 1
        df = df[cols]

    print("[4] WORM 备份 + 原子写回 ...")
    backup = panel_path.replace(
        ".parquet", f"_cyqderiv_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    )
    shutil.copy2(panel_path, backup)
    print(f"    备份: {backup}")
    tmp = panel_path + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, panel_path)
    print(f"    已写回: {panel_path} ({len(df)} 行, {len(df.columns)} 列)")
    print("[done]")


if __name__ == "__main__":
    main()
