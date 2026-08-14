"""_diag_cal_window_compare.py — cal_n=21 vs cal_n=126 校准窗对比 (2026-08-13).

回答: 短窗(21)导致的顺周期幅度高估, 拉长校准窗(126)是否:
  1) 幅度偏差收敛 (pred_top10 ≈ realized_top10)
  2) 选股变化可控 (同日 top-10 股票 Jaccard 相似度 + 实得收益不劣化)
只对比 T+10 (pred_ret_10d, 用户抱怨的 12-13% 主犯).
WORM 输出 data/_diag_cal_window_<ts>.csv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted({c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"})
N_TAIL = 300
WIN = 125  # 对比校准窗 (交易日, 用户 2026-08-14 建议扩窗)
# 纯横截面对照: 强制 per_stock 永不触发 (cal_n=125 下每股已实现样本 ~114, 会触发
# per_stock 回归 → 每股 slope 噪声大, 顶部外推爆表; 记忆 2026-08-07 已裁决每股回归已实亡)
FORCE_CROSS = int(1e9)


def _picks(mag: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """按日预测 mag 降序取 top-n."""
    return (
        mag.sort_values(["date", "mag"], ascending=[True, False])
        .groupby("date")
        .head(n)
    )


def main() -> int:
    out_rows = []
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
        uniq = np.unique(dates.values)
        cutoff = uniq[-(N_TAIL + 20)]
        t = pq.read_table(
            str(fp),
            columns=["symbol", "date"] + POOL_COLS + ["label_pm_10d_net"],
            filters=[("date", ">=", cutoff)],
        ).to_pandas()
        t["symbol"] = t["symbol"].astype(str)
        t["board"] = board
        sn, fu = pool_score(t, SNIPER.pool), pool_score(t, FUSION.pool)
        t["score"] = np.maximum(sn.values, fu.values)
        t = t.dropna(subset=["score"])
        work = t[["symbol", "date", "board", "score", "label_pm_10d_net"]].copy()

        mags = {}
        for lab, cal_n in (("n21", 21), (f"n{WIN}", WIN)):
            m = calibrate_mag10d(
                work,
                cal_n=cal_n,
                per_stock_min_n=FORCE_CROSS,  # 两窗都强制纯横截面 OLS (公平对比)
                target_col="label_pm_10d_net",
                label_horizon=10,
            )
            m["date"] = pd.to_datetime(m["date"])
            m = m.merge(work[["symbol", "date", "label_pm_10d_net"]], on=["symbol", "date"], how="inner")
            mags[lab] = m

        base = mags["n21"]
        reals = base.dropna(subset=["label_pm_10d_net"])
        if reals.empty:
            continue
        # 只用已实现可对比的日期 (两窗都必须有预测 + 有实得)
        common_days = sorted(
            set(mags["n21"]["date"]) & set(mags[f"n{WIN}"]["date"])
        )
        real_days = set(reals["date"].unique())
        days = sorted(set(common_days) & real_days)
        # 取末 ~250 天可实得窗口 (更长 OOS, 降单段行情噪声)
        days = days[-250:]

        p21 = _picks(mags["n21"][mags["n21"]["date"].isin(days)])
        pW = _picks(mags[f"n{WIN}"][mags[f"n{WIN}"]["date"].isin(days)])
        g21 = p21.groupby("date")[["mag", "label_pm_10d_net"]].mean()
        gW = pW.groupby("date")[["mag", "label_pm_10d_net"]].mean()
        joined = g21.join(gW, lsuffix="_21", rsuffix=f"_{WIN}")

        # 选股重叠: 每日 top-10 集合 Jaccard
        def _sym_set(df, day):
            return set(df.loc[df["date"] == day, "symbol"])

        jac = []
        for d in days:
            s21, sW = _sym_set(p21, d), _sym_set(pW, d)
            inter = len(s21 & sW)
            jac.append(inter / len(s21 | sW) if s21 | sW else 1.0)
        overlap = float(np.mean(jac))

        print(f"\n=== {board} T+10 (末 {len(days)} 可实得日) ===")
        print("           n21 pred | n21 real | nW pred | nW real")
        print(
            f"  top10   {joined['mag_21'].mean():+.4f} | {joined['label_pm_10d_net_21'].mean():+.4f}"
            f" | {joined[f'mag_{WIN}'].mean():+.4f} | {joined[f'label_pm_10d_net_{WIN}'].mean():+.4f}"
        )
        print(f"  top10 bias  n21={joined['mag_21'].mean()-joined['label_pm_10d_net_21'].mean():+.4f}"
              f"  n{WIN}={joined[f'mag_{WIN}'].mean()-joined[f'label_pm_10d_net_{WIN}'].mean():+.4f}")
        print(f"  选股重叠 (Jaccard 日均): {overlap:.2%}")
        print(f"  最近5日 n{WIN} top-10 预测: "
              + ", ".join(f"{d.date()}={v:.4f}" for d, v in gW["mag"].tail(5).items()))
        print(f"  最近5日 n21   top-10 预测: "
              + ", ".join(f"{d.date()}={v:.4f}" for d, v in g21["mag"].tail(5).items()))
        out_rows.append(
            {
                "board": board,
                "n_days": len(days),
                "overlap": round(overlap, 4),
                "pred21": round(float(joined["mag_21"].mean()), 4),
                "real21": round(float(joined["label_pm_10d_net_21"].mean()), 4),
                "bias21": round(float(joined["mag_21"].mean() - joined["label_pm_10d_net_21"].mean()), 4),
                f"pred{WIN}": round(float(joined[f"mag_{WIN}"].mean()), 4),
                f"real{WIN}": round(float(joined[f"label_pm_10d_net_{WIN}"].mean()), 4),
                f"bias{WIN}": round(float(joined[f"mag_{WIN}"].mean() - joined[f"label_pm_10d_net_{WIN}"].mean()), 4),
            }
        )
    df = pd.DataFrame(out_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_cal_window_{ts}.csv"
    df.to_csv(out, index=False)
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
