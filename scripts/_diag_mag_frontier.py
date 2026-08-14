"""_diag_mag_frontier.py — 排名深度幅度前沿 (2026-08-14).

用户诉求: 喜欢幅度高的股, 且怀疑 3% 锚定低估. 回答两个问题:
  1) 模型排名前 1/3/5/10 名 (按 pred_mag_10d 降序) 在 250d OOS 里**真实实现**多少幅度?
  2) 幅度 vs 命中率 (实得>0) vs 大涨率 (≥+5%/+10%) 的前沿在哪, 取更深的幅度怎么付代价?

生产同款: calibrate_mag10d (cal_n=21, 纯横截面, 日界无前瞻), score=max(sniper,fusion) 池分.
只评估已实现日 (label_pm_10d_net 非 NaN), 末 250 已实现交易日. WORM 输出
data/_diag_mag_frontier_<ts>.csv.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted({c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"})
N_TAIL = 320  # 决策日载入 (>= 250 已实现 + 校准窗 + 标签视界余量)
EVAL_DAYS = 250  # 评估: 末 250 个已实现交易日
TOPS = (1, 3, 5, 10)


def main() -> int:
    rows = []
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        dates = pd.to_datetime(
            pq.read_table(str(fp), columns=["date"]).to_pandas()["date"]
        )
        uniq = np.unique(dates.values)
        if len(uniq) < N_TAIL + 20:
            continue
        cutoff = uniq[-(N_TAIL + 20)]
        t = pq.read_table(
            str(fp),
            columns=["symbol", "date"] + POOL_COLS + ["label_pm_10d_net"],
            filters=[("date", ">=", cutoff)],
        ).to_pandas()
        t["symbol"] = t["symbol"].astype(str)
        t["board"] = board
        sn = pool_score(t, SNIPER.pool)
        fu = pool_score(t, FUSION.pool)
        t["score"] = np.maximum(sn.values, fu.values)
        t = t.dropna(subset=["score"])
        work = t[["symbol", "date", "board", "score", "label_pm_10d_net"]].copy()
        m = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        if m.empty:
            continue
        mm = m.merge(
            work[["symbol", "date", "label_pm_10d_net"]],
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        if rr.empty:
            continue
        days = sorted(rr["date"].unique())[-EVAL_DAYS:]
        rr = rr[rr["date"].isin(days)]
        rr = rr.sort_values(["date", "mag"], ascending=[True, False])

        print(
            f"\n===== {board}  末 {len(days)} 已实现交易日 (T+10 close-to-close 净) ====="
        )
        print(
            f"{'topN':>5} {'个股·日':>7} {'预测mag均值':>10} {'实得均值':>9} "
            f"{'命中(>0)':>8} {'≥+5%':>7} {'≥+10%':>8}"
        )
        for n in TOPS:
            top = rr.groupby("date", sort=True).head(n)
            pred = float(top["mag"].mean())
            real = float(top["label_pm_10d_net"].mean())
            hit = float((top["label_pm_10d_net"] > 0).mean())
            ge5 = float((top["label_pm_10d_net"] >= 0.05).mean())
            ge10 = float((top["label_pm_10d_net"] >= 0.10).mean())
            n_sd = int(top["symbol"].nunique())
            print(
                f"{n:>5} {n_sd:>7} {pred:>10.2%} {real:>9.2%} {hit:>8.1%} "
                f"{ge5:>7.1%} {ge10:>8.1%}"
            )
            rows.append(
                {
                    "board": board,
                    "top_n": n,
                    "n_stocks": n_sd,
                    "pred_mag_mean": round(pred, 4),
                    "realized_mean": round(real, 4),
                    "hit_rate_gt0": round(hit, 4),
                    "pct_ge5pct": round(ge5, 4),
                    "pct_ge10pct": round(ge10, 4),
                }
            )
    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_mag_frontier_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_mag_frontier_{ts}.json").write_text(
        json.dumps(
            {"ts": ts, "rows": df.to_dict("records")}, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
