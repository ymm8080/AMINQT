"""_diag_mag10d_walkforward.py — 生产同款 mag 校准的 预测 vs 实得 逐日对比 (2026-08-13).

用 calibrate_mag10d (score→label_pm_{h}_net, 与 _shortlist_t5_t10 的 pred_ret 同源)
对整个末段历史 walk-forward 出每股预测, 与面板已实现 label_pm 对比:
  - 每日平均 预测 vs 实得 (全池)
  - 每日 TOP-10 (按预测 mag 降序) 的 预测 vs 实得
量化 pred_ret 是否系统性偏高. WORM 输出 data/_diag_mag10d_wf_<ts>.csv.
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

POOL_COLS = sorted(
    {c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"}
)
N_TAIL = 160  # 末 N 交易日


def main() -> int:
    rows = []
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
        uniq = np.unique(dates.values)
        if len(uniq) < N_TAIL + 20:
            continue
        cutoff = uniq[-(N_TAIL + 20)]
        t = pq.read_table(
            str(fp),
            columns=["symbol", "date"] + POOL_COLS + ["label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net"],
            filters=[("date", ">=", cutoff)],
        ).to_pandas()
        t["symbol"] = t["symbol"].astype(str)
        t["board"] = board
        sn = pool_score(t, SNIPER.pool)
        fu = pool_score(t, FUSION.pool)
        t["score"] = np.maximum(sn.values, fu.values)
        t = t.dropna(subset=["score"])
        for h, horizon in (("3d", 3), ("5d", 5), ("10d", 10)):
            col = f"label_pm_{h}_net"
            work = t[["symbol", "date", "board", "score", col]].copy()
            m = calibrate_mag10d(
                work,
                score_col="score",
                target_col=col,
                label_horizon=horizon,
            )
            if m.empty:
                continue
            mm = m.merge(
                work[["symbol", "date", col]],
                on=["symbol", "date"],
                how="inner",
            )
            mm["date"] = pd.to_datetime(mm["date"])
            # 最近 5 个决策日的 TOP-10 预测 (未实得日也展示 — 看当前预测是否尖峰)
            latest5 = sorted(mm["date"].unique())[-5:]
            last5 = mm[mm["date"].isin(latest5)].sort_values(
                ["date", "mag"], ascending=[True, False]
            ).groupby("date").head(10).groupby("date")["mag"].mean()
            print(
                f"[{board} T+{h[:-1]}] 最近5决策日 TOP-10 预测 mag: "
                + ", ".join(f"{d.date()}={v:.4f}" for d, v in last5.items())
            )
            # 只对比已实现行
            rr = mm.dropna(subset=[col])
            if rr.empty:
                continue
            # 全池逐日
            daily = rr.groupby("date").agg(
                pred=("mag", "mean"), realized=(col, "mean")
            )
            # TOP-10 逐日 (按当日预测 mag 降序)
            top = rr.sort_values(["date", "mag"], ascending=[True, False]).groupby("date").head(10)
            t10 = top.groupby("date").agg(
                pred_top10=("mag", "mean"), realized_top10=(col, "mean")
            )
            joined = daily.join(t10)
            joined = joined[joined.index >= joined.index.max() - pd.Timedelta(days=90)]
            bias = float((joined["pred_top10"] - joined["realized_top10"]).mean())
            rows.append(
                {
                    "board": board,
                    "horizon": h,
                    "n_days": int(len(joined)),
                    "pred_top10_mean": float(joined["pred_top10"].mean()),
                    "realized_top10_mean": float(joined["realized_top10"].mean()),
                    "bias_top10": bias,
                    "pred_pool_mean": float(joined["pred"].mean()),
                    "realized_pool_mean": float(joined["realized"].mean()),
                    "bias_pool": float(joined["pred"].mean() - joined["realized"].mean()),
                }
            )
            print(f"\n=== {board} T+{h[:-1]} TOP-10 (末 {len(joined)} 交易日) ===")
            print("pred_top10  vs  realized_top10:")
            print((joined[["pred_top10", "realized_top10"]]).round(4).to_string())
            print(
                f"  bias_top10 = {bias:+.4f}  (pred {joined['pred_top10'].mean():.4f}"
                f" vs real {joined['realized_top10'].mean():.4f})"
            )
    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_mag10d_wf_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_mag10d_wf_{ts}.json").write_text(
        json.dumps({"ts": ts, "rows": df.to_dict("records")}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
