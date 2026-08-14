"""_diag_mag_rank_key.py — 入选键 A/B: mag (calibrate_mag10d, 生产入选键) vs score (池分).

背景 (2026-08-14): 锚定 `_trailing_realized` 按 score top-N, frontier 按 mag top-N, 两者
实得不同 (dual T+10: +7.4% vs +6.34%). 因 per_stock_min_n=50 > cal_n=21 → 强制纯横截面,
mag = cs·score + ci, 日内 mag 与 score 完全单调: cs>0 同序 / cs<0 反序. 故两键分歧
全部发生在 cs<0 (横截面斜率倒挂) 的日子. 本脚本逐日回答:
  1) 分歧频率: 多少天 overlap<5 (= cs<0 反序日), 占比多少?
  2) 两种入选键 250d 实得幅度 / 命中率(>0) / ≥5% / ≥10% 谁高? 若 score 系统性更高
     → 反序日 mag 入选 (= 该日 score 最低 5 只) 拖累模块质量, 命中率可从这里提升.
  3) 反序日下两键实得差, 判断反序日是「该日信号倒挂→低分股反而涨」还是「低分股跌更多」.

生产同款校准 (cal_n=21, cross_min_n=50), 只评估已实现日 (label 非 NaN), 末 250 日.
WORM: data/_diag_mag_rank_key_<ts>.csv + .json (含逐日明细).
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
from app.pipeline_parallel.config import FUSION, SNIPER, MAG10D_CAL
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted(
    {c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"}
)
N_TAIL = 320  # 决策日载入 (>= 250 已实现 + 校准窗 + 标签视界余量)
EVAL_DAYS = 250  # 评估: 末 250 个已实现交易日
TOP = 5
REALIZED_DROP = int(MAG10D_CAL["buy_lag"]) + int(MAG10D_CAL["label_horizon"])  # 11


def _cross_slope(t: pd.DataFrame, d: pd.Timestamp, all_dates: list, date_idx: dict):
    """决策日 d 的横截面斜率 cs: 对窗 [d-cal_n, d-11] (仅已实现标签) 的 score→label OLS.
    与 calibrate_mag10d 的 i_lo/i_end 边界同构 (无前瞻)."""
    di = date_idx[d]
    if di < REALIZED_DROP:
        return float("nan")
    cal_lo = all_dates[max(0, di - MAG10D_CAL["cal_n"])]
    boundary = all_dates[di - REALIZED_DROP]
    win = t[(t["date"] >= cal_lo) & (t["date"] <= boundary)]
    x = win["score"].to_numpy(float)
    y = win["label_pm_10d_net"].to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        return float("nan")
    var = float(((x - x.mean()) ** 2).sum())
    if var <= 1e-12:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).sum() / var)


def _stats(stack: pd.DataFrame) -> tuple:
    lab = stack["label_pm_10d_net"]
    return (
        float(lab.mean()),
        float((lab > 0).mean()),
        float((lab >= 0.05).mean()),
        float((lab >= 0.10).mean()),
        int(stack["symbol"].nunique()),
    )


def main() -> int:
    print(f"calibrate_mag10d 键 vs score 键 | 末 {EVAL_DAYS} 已实现日, TOP-{TOP} | "
          f"per_stock_min_n={MAG10D_CAL['per_stock_min_n']} > cal_n={MAG10D_CAL['cal_n']} → 纯横截面")
    out_rows: list[dict] = []
    detail: list[dict] = []
    for board in ("main", "dual"):
        fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
        dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
        uniq = np.unique(dates.values)
        if len(uniq) < N_TAIL + 20:
            print(f"[skip] {board}: 数据不足")
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
        t = t.dropna(subset=["score"]).reset_index(drop=True)
        t["date"] = pd.to_datetime(t["date"])
        all_dates = sorted(t["date"].unique())
        date_idx = {d: i for i, d in enumerate(all_dates)}

        work = t[["symbol", "date", "board", "score", "label_pm_10d_net"]].copy()
        m = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        if m.empty:
            print(f"[skip] {board}: calibrate_mag10d 无输出")
            continue
        mm = m.merge(
            work[["symbol", "date", "score", "label_pm_10d_net"]],
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        days = sorted(rr["date"].unique())[-EVAL_DAYS:]
        rr = rr[rr["date"].isin(days)]

        day_rows = []
        for d in days:
            dm = rr[rr["date"] == d]
            m5 = dm.sort_values("mag", ascending=False).head(TOP)
            s5 = dm.sort_values("score", ascending=False).head(TOP)
            ov = len(set(m5["symbol"]) & set(s5["symbol"]))
            cs = _cross_slope(t, d, all_dates, date_idx)
            day_rows.append(
                {
                    "date": d,
                    "n": len(dm),
                    "overlap": ov,
                    "cs": cs,
                    "real_mag5": float(m5["label_pm_10d_net"].mean()),
                    "real_score5": float(s5["label_pm_10d_net"].mean()),
                    "mag5_mean_score": float(m5["score"].mean()),
                    "score5_mean_score": float(s5["score"].mean()),
                }
            )
        df = pd.DataFrame(day_rows)
        n_neg = int((df["cs"] < 0).sum())
        n_zero = int((df["cs"] == 0).sum())
        n_pos = int((df["cs"] > 0).sum())
        n_div = int((df["overlap"] < TOP).sum())

        mag_st = rr.sort_values(["date", "mag"], ascending=[True, False]).groupby("date", sort=False).head(TOP)
        sco_st = rr.sort_values(["date", "score"], ascending=[True, False]).groupby("date", sort=False).head(TOP)
        r_m, h_m, g5_m, g10_m, ns_m = _stats(mag_st)
        r_s, h_s, g5_s, g10_s, ns_s = _stats(sco_st)

        # 反序日 (overlap<5) 内两键实得
        dv = df[df["overlap"] < TOP]
        div_m = float(dv["real_mag5"].mean()) if len(dv) else float("nan")
        div_s = float(dv["real_score5"].mean()) if len(dv) else float("nan")

        print(f"\n===== {board} | {len(days)} 已实现日 | cs 正/零/负 = {n_pos}/{n_zero}/{n_neg}"
              f" | 反序日(overlap<5) {n_div} ({n_div / len(days):.0%}) =====")
        print(f"  键       实得均值  命中(>0)  ≥+5%   ≥+10%   {f'个股':>4}")
        print(f"  mag   {r_m:>9.2%} {h_m:>8.1%} {g5_m:>6.1%} {g10_m:>7.1%} {ns_m:>5}")
        print(f"  score {r_s:>9.2%} {h_s:>8.1%} {g5_s:>6.1%} {g10_s:>7.1%} {ns_s:>5}")
        print(f"  Δ(mag-score) {r_m - r_s:>+9.2%} (命中 {h_m - h_s:+.1%})")
        if len(dv):
            print(f"  反序日({len(dv)}天): mag 实得 {div_m:+.2%} vs score 实得 {div_s:+.2%}"
                  f" (Δ {div_m - div_s:+.2%})")
        print(f"  反序日 mag-top5 均分 {float(df.loc[df.overlap<TOP,'mag5_mean_score'].mean()):.3f}"
              f" vs 该日全池均分≈score-top5 {float(df.loc[df.overlap<TOP,'score5_mean_score'].mean()):.3f}")

        out_rows.append(
            {
                "board": board,
                "eval_days": len(days),
                "cs_pos": n_pos, "cs_zero": n_zero, "cs_neg": n_neg,
                "pct_cs_neg": round(n_neg / len(days), 4),
                "diverging_days": n_div,
                "pct_diverging": round(n_div / len(days), 4),
                "realized_mag_top5": round(r_m, 4),
                "hit_mag": round(h_m, 4),
                "pct_ge5_mag": round(g5_m, 4),
                "pct_ge10_mag": round(g10_m, 4),
                "realized_score_top5": round(r_s, 4),
                "hit_score": round(h_s, 4),
                "pct_ge5_score": round(g5_s, 4),
                "pct_ge10_score": round(g10_s, 4),
                "diff_realized_mag_minus_score": round(r_m - r_s, 4),
                "diff_hit_mag_minus_score": round(h_m - h_s, 4),
                "div_realized_mag": round(div_m, 4),
                "div_realized_score": round(div_s, 4),
            }
        )
        for r in df.to_dict("records"):
            r["board"] = board
            detail.append(r)

    if not out_rows:
        print("无输出")
        return 1
    df_out = pd.DataFrame(out_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_p = DATA_DIR / f"_diag_mag_rank_key_{ts}.csv"
    df_out.to_csv(csv_p, index=False)
    (DATA_DIR / f"_diag_mag_rank_key_{ts}.json").write_text(
        json.dumps(
            {"ts": ts, "summary": df_out.to_dict("records"), "day_detail": detail},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {csv_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
