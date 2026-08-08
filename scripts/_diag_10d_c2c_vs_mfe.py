"""头对头: 并行 TOP10 排名 校准口径 close-to-close vs MFE (2026-08-07 用户).

用户决定并行系统短名单按 T10 预测排名, 唯一未定 = mag_10d 的校准目标:
  现有回测 mag_10d 用 MFE 标签校准 (label_mfe_10d_net, 持有期内最高价可兑现),
  只把"评估"换成了 close-to-close. 用户怀疑 MFE 是触摸天花板(不可兑现),
  要求校准目标也换 close-to-close 平均涨幅 (label_pm_10d_net = close[T+11]/close[T+1]-1-cost).

本脚本同流程只换校准口径, 头对头测哪组 TOP10 最后实得 performance 更好:
  - mag_10d_c2c: 每股收缩回归 target=label_pm_10d_net
  - mag_10d_mfe: 每股收缩回归 target=label_mfe_10d_net
  - 各自全板块日截面按 mag_10d 降序取 TOP10 (main/dual 独立)
评估口径 = close-to-close 点对点收益 pr_2d/3d/5d/10d (买=close[T+1], 卖=close[T+1+k])
         + 上涨率 + 两组重合度. walk-forward 无前瞻 (cal 只用 < D 数据).

范围: 只并行系统 TOP10; 慢牛/legacy 不在此范围.
用法: python scripts/_diag_10d_c2c_vs_mfe.py [eval_days=250]
WORM: BACKTEST_RESULT_DIR/diag_10d_c2c_vs_mfe_<ts>/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline1.label_engine import COST, slippage_tier
from app.pipeline_parallel import indicators
from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER, board_of
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

CAL_N = 160  # 滚动校准窗口长度 (交易日)
PER_STOCK_WINDOW = 130  # 与生产一致: 每股自用最近交易日
PER_STOCK_MIN_N = 30  # 与生产一致: 每股回归最小样本, 不足回退横截面
SHRINK_KAPPA = 40  # 与生产一致: 收缩强度
TOP_N = 10  # 每板块短名单档位
H_K = (2, 3, 5, 10)  # 点对点收益视界
BOARDS = ("main", "dual")

POOL_COLS = [
    "amihud_illiq",
    "small_mv_premium",
    "amihud_illiquidity",
    "down_gap_pct",
    "VAR51",
    "ret_reversal_5d",
    "limit_dist_pct",
]
NEEDED_COLS = sorted(
    set(
        [
            "symbol",
            "date",
            "close_hfq",
            "high_hfq",
            "low_hfq",
            "volume",
            "turnover_rate",
            "adv20",
        ]
        + POOL_COLS
    )
)


def load_panel_tail(tail_days: int) -> pd.DataFrame:
    """内存安全尾部加载: pyarrow 列选择 + 行过滤 → MFE/c2c 标签 → tradability_gate
    → board → prepare_adx (补 pv_corr_5)."""
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        t = pq.read_table(ckpt, columns=NEEDED_COLS)
        df = t.to_pandas()
        dates = sorted(df["date"].unique())
        df = df[df["date"] >= dates[-tail_days]].reset_index(drop=True)
        slices.append(df)
        del t, df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    work = add_mfe_labels(work, horizons=H_K)
    work = add_pm_labels(work, horizons=H_K)
    work, _gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    work = indicators.prepare_adx(work)
    return work


def add_pm_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """补算 close-to-close 净标签 (生产 label_pm_{k}d 口径):

    label_pm_{k}d_net = close_hfq[T+1+k] / close_hfq[T+1] - 1 - cost
    买价 close[T+1] (T+1 买), 卖价目标日收盘 close[T+1+k]; 成本口径同 add_mfe_labels.
    """
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)  # T+1 收盘买价
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for k in horizons:
        sell_px = g["close_hfq"].shift(-(1 + k))  # T+1+k 收盘卖价
        df[f"label_pm_{k}d_net"] = sell_px / exec_px - 1 - cost_total
    return df


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """单变量最小二乘 (numpy, 快速路径)."""
    xm = float(x.mean())
    ym = float(y.mean())
    var = float(((x - xm) ** 2).sum())
    if var <= 1e-12:
        return 0.0, ym
    slope = float(((x - xm) * (y - ym)).sum() / var)
    return slope, ym - slope * xm


def _safe_nanmean(a: np.ndarray) -> float:
    """nanmean 但全 NaN → nan (不报警告)."""
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def point_rets(closes: dict, sym: str, d, ks: tuple[int, ...]) -> list[float]:
    """该股在日期 d 的前向点对点收益 (close[T+1+k]/close[T+1]-1); 数据不足→nan."""
    if sym not in closes:
        return [np.nan] * len(ks)
    dates, close = closes[sym]
    d = np.datetime64(d)  # Timestamp → datetime64 (numpy searchsorted 不认 pandas Timestamp)
    i = int(np.searchsorted(dates, d))
    if i >= len(dates) or dates[i] != d or i + 1 >= len(dates):
        return [np.nan] * len(ks)
    exec_px = close[i + 1]
    if not np.isfinite(exec_px) or exec_px <= 0:
        return [np.nan] * len(ks)
    out = []
    for k in ks:
        j = i + 1 + k
        out.append(close[j] / exec_px - 1.0 if j < len(dates) else np.nan)
    return out


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_10d_c2c_vs_mfe_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = CAL_N + args.eval_days + 40
    print(f"[panel] 尾部加载 {tail_days} 交易日...", flush=True)
    work = load_panel_tail(tail_days)
    all_dates = sorted(work["date"].unique())
    eval_dates = all_dates[-args.eval_days :]
    date_idx = {d: i for i, d in enumerate(all_dates)}
    print(
        f"[panel] {len(work):,}r / {work['symbol'].nunique():,}只 / "
        f"{len(all_dates)} 交易日 | 评估 {eval_dates[0].date()}..{eval_dates[-1].date()} "
        f"({len(eval_dates)} 日) ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # 每股 (date, close_hfq) 数组 — 点对点收益评估用 (含前向 40 交易日)
    print("[closes] 每股收盘价序列...", flush=True)
    cc = work[["symbol", "date", "close_hfq"]]
    d64 = cc["date"].to_numpy().astype("datetime64[ns]")
    closes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in cc.groupby("symbol", sort=False):
        closes[str(sym)] = (d64[g.index.to_numpy()], g["close_hfq"].to_numpy())
    del cc
    gc.collect()
    print(f"[closes] {len(closes):,} 只", flush=True)

    # 全池逐系统打分 + score=max(狙击,融合) (同 _diag_parallel_rank_compare)
    print("[score] 全池 pool_score...", flush=True)
    work["score_sniper"] = np.nan
    work["score_fusion"] = np.nan
    for spec in (SNIPER, FUSION):
        col = f"score_{spec.name}"
        for board in BOARDS:
            bm = work["board"] == board
            work.loc[bm, col] = pool_score(work[bm], spec.pool).values
    work["score"] = work[["score_sniper", "score_fusion"]].max(axis=1)
    print(f"[score] done ({time.time() - t0:.0f}s)", flush=True)

    # walk-forward: 每评估日 D 用 [D-CAL_N, D-1] 拟合 → 对 D 打 mag → TOP10 (两口径)
    print("[calib] walk-forward 逐日重拟合 (c2c vs MFE)...", flush=True)
    rows: list[dict] = []
    for D in eval_dates:
        di = date_idx[D]
        cal_lo = all_dates[max(0, di - CAL_N)]
        cal = work[(work["date"] >= cal_lo) & (work["date"] < D)]
        day = work[work["date"] == D]
        if cal.empty or day.empty:
            continue
        for board in BOARDS:
            calsub = cal[cal["board"] == board]
            daysub = day[day["board"] == board]
            if len(calsub) < 200 or daysub.empty:
                continue
            day_scores = daysub["score"].to_numpy(float)
            day_sym = daysub["symbol"].astype(str).to_numpy()
            by_sym = {k: v for k, v in calsub.groupby("symbol", sort=False)}
            mag: dict[str, np.ndarray] = {}
            for tgt_name, tgt_col in (
                ("c2c", "label_pm_10d_net"),
                ("mfe", "label_mfe_10d_net"),
            ):
                c = calsub[["score", tgt_col]]
                if c[tgt_col].isna().all():
                    continue
                xcal = c["score"].to_numpy(float)
                ycal = c[tgt_col].to_numpy(float)
                m = np.isfinite(xcal) & np.isfinite(ycal)
                if m.sum() < 50:
                    continue
                xcal, ycal = xcal[m], ycal[m]
                cross_slope, cross_int = _ols_slope_intercept(xcal, ycal)
                m_arr = np.full(len(daysub), np.nan)
                for i, sym in enumerate(day_sym):
                    sc = day_scores[i]
                    if not np.isfinite(sc):
                        continue
                    ps = by_sym.get(sym)
                    if ps is None or ps.empty:
                        continue
                    gg = ps[["score", tgt_col]].dropna().tail(PER_STOCK_WINDOW)
                    if len(gg) >= PER_STOCK_MIN_N:
                        x = gg["score"].to_numpy(float)
                        y = gg[tgt_col].to_numpy(float)
                        raw_slope, _ = _ols_slope_intercept(x, y)
                        lam = len(gg) / (len(gg) + SHRINK_KAPPA)
                        slope = lam * raw_slope + (1 - lam) * cross_slope
                        intercept = float(y.mean()) - slope * float(x.mean())
                    else:
                        slope, intercept = cross_slope, cross_int
                    m_arr[i] = slope * sc + intercept
                mag[tgt_name] = m_arr

            if "c2c" not in mag or "mfe" not in mag:
                continue
            rank_df = pd.DataFrame(
                {
                    "symbol": day_sym,
                    "score": day_scores,
                    "mag_c2c": mag["c2c"],
                    "mag_mfe": mag["mfe"],
                }
            )
            tops = {
                name: rank_df.sort_values(f"mag_{name}", ascending=False).head(TOP_N)
                for name in ("c2c", "mfe")
            }
            overlap = len(set(tops["c2c"]["symbol"]) & set(tops["mfe"]["symbol"]))
            for name in ("c2c", "mfe"):
                pick = tops[name]
                pr = np.array([point_rets(closes, s, D, H_K) for s in pick["symbol"]])
                if len(pr) < 5:
                    continue
                rows.append(
                    {
                        "date": D,
                        "board": board,
                        "method": name,
                        "n": int(len(pick)),
                        "pr2d": _safe_nanmean(pr[:, 0]),
                        "pr3d": _safe_nanmean(pr[:, 1]),
                        "pr5d": _safe_nanmean(pr[:, 2]),
                        "pr10d": _safe_nanmean(pr[:, 3]),
                        "win5d": float((pr[:, 2] > 0).sum() / (~np.isnan(pr[:, 2])).sum())
                        if (~np.isnan(pr[:, 2])).sum()
                        else float("nan"),
                        "win10d": float((pr[:, 3] > 0).sum() / (~np.isnan(pr[:, 3])).sum())
                        if (~np.isnan(pr[:, 3])).sum()
                        else float("nan"),
                        "overlap": overlap,
                    }
                )
        del cal, day
        gc.collect()

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    # 汇总: 逐板块 两组均值 + 头对头逐日胜负
    print(f"\n===== TOP10 校准口径头对头: close-to-close vs MFE "
          f"(最近 {args.eval_days} 交易日) =====", flush=True)
    print("评估 = close-to-close 实得 (买=close[T+1], 卖=close[T+1+k]); 校准 = 每股收缩回归", flush=True)
    agg_rows: list[dict] = []
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}]", flush=True)
        print(
            f"  {'口径':<8}{'日':>4}{'2d':>9}{'3d':>9}{'5d':>9}{'10d':>9}"
            f"{'5d涨率':>9}{'10d涨率':>9}",
            flush=True,
        )
        for method in ("c2c", "mfe"):
            g = sub[sub["method"] == method]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            print(
                f"  {method:<8}{int(len(g)):>4}{r['pr2d']:>+9.4f}{r['pr3d']:>+9.4f}"
                f"{r['pr5d']:>+9.4f}{r['pr10d']:>+9.4f}"
                f"{r['win5d']:>9.1%}{r['win10d']:>9.1%}",
                flush=True,
            )
        a = sub[sub["method"] == "c2c"].set_index("date")[["pr5d", "pr10d", "win5d"]]
        b = sub[sub["method"] == "mfe"].set_index("date")[["pr5d", "pr10d", "win5d"]]
        common = a.index.intersection(b.index)
        if len(common):
            m5 = ~np.isnan(a.loc[common, "pr5d"].values) & ~np.isnan(b.loc[common, "pr5d"].values)
            m10 = ~np.isnan(a.loc[common, "pr10d"].values) & ~np.isnan(b.loc[common, "pr10d"].values)
            w5 = int((a.loc[common, "pr5d"].values[m5] > b.loc[common, "pr5d"].values[m5]).sum())
            w10 = int((a.loc[common, "pr10d"].values[m10] > b.loc[common, "pr10d"].values[m10]).sum())
            print(
                f"  逐日 c2c 赢 MFE: 5d实得 {w5}/{m5.sum()} 天, "
                f"10d实得 {w10}/{m10.sum()} 天, "
                f"平均重合 {float(sub['overlap'].mean()):.1f}/10",
                flush=True,
            )
            agg_rows.append(
                {
                    "board": board,
                    "n_days_5d": int(m5.sum()),
                    "n_days_10d": int(m10.sum()),
                    "c2c_wins_pr5d": w5,
                    "c2c_wins_pr10d": w10,
                    "overlap_mean": round(float(sub["overlap"].mean()), 3),
                    "note": "n_days 为两组该视界均有实得数据的共同评估日 (末几日前向价不足)",
                }
            )
        for method in ("c2c", "mfe"):
            g = sub[sub["method"] == method]
            r = g.mean(numeric_only=True)
            agg_rows.append(
                {
                    "board": board,
                    "method": method,
                    "n_days": int(len(g)),
                    "pr2d": round(float(r["pr2d"]), 5),
                    "pr3d": round(float(r["pr3d"]), 5),
                    "pr5d": round(float(r["pr5d"]), 5),
                    "pr10d": round(float(r["pr10d"]), 5),
                    "win5d": round(float(r["win5d"]), 4),
                    "win10d": round(float(r["win10d"]), 4),
                }
            )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "eval_days": args.eval_days,
        "metric": "close-to-close point returns (买=close[T+1], 卖=close[T+1+k]); NOT MFE",
        "calibration": "每股收缩回归 (κ=40, 窗130, min_n=30, 回退横截面); "
                       "c2c target=label_pm_10d_net, mfe target=label_mfe_10d_net",
        "cal_window_days": CAL_N,
        "top_n": TOP_N,
        "boards": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
