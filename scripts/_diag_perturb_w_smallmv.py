"""small_mv_premium 权重赢家的扰动检验 (2026-08-08, 依据《AI扫参风险》#8).

背景: 全参数扫描 (_diag_parallel_allparam_20260808_044134) 唯一 STABLE_WIN =
  SNIPER 池 small_mv_premium ×2 等权 (dual pr10d +0.0516 vs ref +0.0465, 4/4 季度,
  无负季度, 清单稳定). 文档硬门槛: 选定参数后必须做扰动检验 — 参数微调 ±20%,
  Top 清单变化 >50% → 参数对扰动过度敏感 → 弃用. 只有"扰动后清单稳定"才能进实盘.

本脚本: 对 SNIPER 池 small_mv_premium 权重乘子 m ∈ {1.0,1.5,2.0,2.5,3.0} 跑
  walk-forward (无前瞻, 纯横截面 OLS 校准 = 生产 MAG10D_CAL), 输出:
    overlap       = 当日 TOP10 与 ref(m=1.0) 重合
    overlap_chosen= 当日 TOP10 与选定点 m=2.0 重合 (扰动敏感度)
  判定: m=2.0 的 ±20% 邻点 (1.5/2.5) 与 2.0 清单平均重合度 < 50% → 弃用;
        否则通过 + 各季度 pr10d 不翻负.

自包含 (同 _diag_parallel_allparam_sweep: 禁 import backtest, helper 内联).
用法: python scripts/_diag_perturb_w_smallmv.py [eval_days=250]
WORM: BACKTEST_RESULT_DIR/diag_perturb_w_smallmv_<ts>/
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
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER, board_of
from app.pipeline_parallel.indicators import prepare_adx
from app.pipeline_parallel.scoring import cross_rank
from config.settings import BACKTEST_RESULT_DIR

H_K = (2, 3, 5, 10)
BOARDS = ("main", "dual")
REALIZED_DROP = 11  # buy_lag + label_horizon
TOP_N = 10
MULTS = (1.0, 1.5, 2.0, 2.5, 3.0)  # small_mv_premium 权重乘子; 1.0 = 等权 ref
TARGET = "small_mv_premium"
CHOSEN = 2.0  # 扫描赢家
# 文档 #8 硬门槛: 扰动 ±20% 后清单变化 >50% → 弃用 (TOP_N=10 → 重合 <5 弃用)
PERTURB = (1.5, 2.5)
MIN_OVERLAP_CHOSEN = 5

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
def _avail(work: pd.DataFrame, pool) -> list[str]:
    return [c for c in pool if c in work.columns]


def load_panel_tail(tail_days: int) -> pd.DataFrame:
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
    work = add_pm_labels(work, horizons=H_K)
    work, _gate = tradability_gate(work)
    work = work.reset_index(drop=True)
    work["board"] = work["symbol"].map(board_of)
    # 生产 load_panel 会调 prepare_adx → 加 pv_corr_5 等指标列; 复刻它以在真实
    # 生产特征集 (SNIPER/FUSION 各 7 特征) 上验证, 消除扫描 6 特征口径偏差.
    work = prepare_adx(work)
    return work


def add_pm_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for k in horizons:
        sell_px = g["close_hfq"].shift(-(1 + k))
        df[f"label_pm_{k}d_net"] = sell_px / exec_px - 1 - cost_total
    return df


def tradability_gate(
    work: pd.DataFrame, lookback: int = 20, min_presence: float = 0.8
) -> tuple[pd.DataFrame, dict]:
    dates = np.sort(work["date"].unique())
    di = np.searchsorted(dates, work["date"].values)
    syms = work["symbol"].unique()
    sym_pos = {s: i for i, s in enumerate(syms)}
    si = np.array([sym_pos[s] for s in work["symbol"]])
    mat = np.zeros((len(syms), len(dates)), dtype=np.int64)
    np.add.at(mat, (si, di), 1)
    np.minimum(mat, 1, out=mat)
    csum = np.cumsum(mat, axis=1)
    col_cur = csum[si, di]
    prev = di - lookback
    col_prev = np.zeros(len(work), dtype=np.int64)
    ok = prev >= 0
    col_prev[ok] = csum[si[ok], prev[ok]]
    count = col_cur - col_prev
    denom = np.minimum(lookback, di + 1).astype(float)
    ratio = count / denom
    keep = ratio >= min_presence
    return work.loc[keep], {}


def point_rets(closes: dict, sym: str, d, ks: tuple[int, ...]) -> list[float]:
    if sym not in closes:
        return [np.nan] * len(ks)
    dates, close = closes[sym]
    d = np.datetime64(d)
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


def _safe_nanmean(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def build_score_cols(work: pd.DataFrame, sniper_avail, fusion_avail) -> dict[float, np.ndarray]:
    """每乘子 m: score_m = max(SNIPER 加权, FUSION 等权) 池分 (跨板块日期截面)."""
    r_s = {c: cross_rank(work, c).to_numpy(float) for c in sniper_avail}
    r_f = {c: cross_rank(work, c).to_numpy(float) for c in fusion_avail}
    sum_s = np.sum(np.stack([r_s[c] for c in sniper_avail]), axis=0)  # 加链传播 NaN
    sum_f = np.sum(np.stack([r_f[c] for c in fusion_avail]), axis=0)
    n_s, n_f = len(sniper_avail), len(fusion_avail)
    fusion_eq = sum_f / n_f
    cols: dict[float, np.ndarray] = {}
    for m in MULTS:
        sniper_m = (sum_s + (m - 1.0) * r_s[TARGET]) / (n_s - 1.0 + m)
        cols[m] = np.fmax(sniper_m, fusion_eq)
    return cols


def build_cross(work: pd.DataFrame, score_cols: dict[float, np.ndarray]) -> dict:
    cross: dict = {}
    d64 = work["date"].to_numpy().astype("datetime64[ns]")
    y = work["label_pm_10d_net"].to_numpy(float)
    for board in BOARDS:
        bm = work["board"].values == board
        by: dict = {}
        for m, x in score_cols.items():
            valid = np.isfinite(x) & np.isfinite(y) & bm
            d = d64[valid]
            if len(d) == 0:
                by[m] = None
                continue
            bd, bin_idx = np.unique(d, return_inverse=True)
            nb = len(bd)
            xv = x[valid]
            yv = y[valid]
            by[m] = (
                bd,
                np.concatenate([[0.0], np.cumsum(np.bincount(bin_idx, minlength=nb).astype(float))]),
                np.concatenate([[0.0], np.cumsum(np.bincount(bin_idx, weights=xv, minlength=nb))]),
                np.concatenate([[0.0], np.cumsum(np.bincount(bin_idx, weights=yv, minlength=nb))]),
                np.concatenate([[0.0], np.cumsum(np.bincount(bin_idx, weights=xv * xv, minlength=nb))]),
                np.concatenate([[0.0], np.cumsum(np.bincount(bin_idx, weights=xv * yv, minlength=nb))]),
            )
        cross[board] = by
        del bm
        gc.collect()
    return cross


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_perturb_w_smallmv_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = 21 + REALIZED_DROP + args.eval_days + 40
    print(f"[panel] 尾部加载 {tail_days} 交易日...", flush=True)
    work = load_panel_tail(tail_days)
    sniper_avail = _avail(work, SNIPER.pool)
    fusion_avail = _avail(work, FUSION.pool)
    assert TARGET in sniper_avail and sniper_avail and fusion_avail, (
        f"avail 为空: sniper={sniper_avail} fusion={fusion_avail}"
    )
    print(
        f"[avail] sniper={len(sniper_avail)}特征 {sniper_avail} | "
        f"fusion={len(fusion_avail)}特征 {fusion_avail}",
        flush=True,
    )
    all_dates = sorted(work["date"].unique())
    eval_dates = all_dates[-args.eval_days:]
    date_idx = {d: i for i, d in enumerate(all_dates)}
    print(
        f"[panel] {len(work):,}r / {work['symbol'].nunique():,}只 / "
        f"{len(all_dates)} 交易日 | 评估 {eval_dates[0].date()}..{eval_dates[-1].date()} "
        f"({len(eval_dates)} 日) ({time.time() - t0:.0f}s)",
        flush=True,
    )

    print("[closes] 每股收盘价序列...", flush=True)
    cc = work[["symbol", "date", "close_hfq"]]
    d64 = cc["date"].to_numpy().astype("datetime64[ns]")
    closes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in cc.groupby("symbol", sort=False):
        closes[str(sym)] = (d64[g.index.to_numpy()], g["close_hfq"].to_numpy())
    del cc
    gc.collect()

    print(f"[score] 预计算 {len(MULTS)} 个加权 score 列 (乘子 {MULTS})...", flush=True)
    score_cols = build_score_cols(work, sniper_avail, fusion_avail)
    print("[cross] 每 (板块, 乘子) 日期桶前缀和...", flush=True)
    cross = build_cross(work, score_cols)

    print(f"[day] 逐日截面预提取 (syms + score 矩阵[{len(MULTS)}])...", flush=True)
    day_info: dict = {}
    for D in eval_dates:
        day = work[work["date"] == D]
        db = {}
        for board in BOARDS:
            sub = day[day["board"] == board]
            if sub.empty:
                db[board] = None
                continue
            smat = np.column_stack([score_cols[m][sub.index] for m in MULTS])
            db[board] = (sub["symbol"].astype(str).to_numpy(), smat)
        day_info[D] = db
    del work, day, score_cols
    gc.collect()
    print(f"[day] {len(day_info)} 日预提取完成 ({time.time() - t0:.0f}s)", flush=True)

    def cross_slope_int(board: str, m: float, cal_lo, cutoff_date):
        ent = cross[board].get(m)
        if ent is None:
            return None
        bd, n_bin, sx, sy, sxx, sxy = ent
        iLo = int(np.searchsorted(bd, np.datetime64(cal_lo), side="left"))
        iEnd = int(np.searchsorted(bd, np.datetime64(cutoff_date), side="right"))
        n = n_bin[iEnd] - n_bin[iLo]
        if n < 1:
            return None
        Sx = sx[iEnd] - sx[iLo]
        Sy = sy[iEnd] - sy[iLo]
        Sxx = sxx[iEnd] - sxx[iLo]
        Sxy = sxy[iEnd] - sxy[iLo]
        var = n * Sxx - Sx * Sx
        if var <= 1e-12:
            return (0.0, Sy / n, n)
        slope = (n * Sxy - Sx * Sy) / var
        return (slope, (Sy - slope * Sx) / n, n)

    print(f"[calib] walk-forward 逐日重拟合 × {len(MULTS)} 乘子 (无前瞻窗)...", flush=True)
    rows: list[dict] = []
    cs_cache: dict = {}
    for di_abs, D in enumerate(eval_dates):
        di = date_idx[D]
        np.datetime64(D)
        for board in BOARDS:
            ent = day_info[D].get(board)
            if ent is None:
                continue
            syms, smat = ent
            tops: dict[float, pd.DataFrame | None] = {}
            for mi, m in enumerate(MULTS):
                if di < max(21, REALIZED_DROP):
                    tops[m] = None
                    continue
                cal_lo = all_dates[di - 21]
                cutoff_date = all_dates[di - REALIZED_DROP]
                ckey = (board, m, cal_lo, cutoff_date)
                if ckey not in cs_cache:
                    cs_cache[ckey] = cross_slope_int(board, m, cal_lo, cutoff_date)
                cs_ent = cs_cache[ckey]
                if cs_ent is None or cs_ent[2] < 50:  # cross_min_n
                    tops[m] = None
                    continue
                slope, intercept, _ = cs_ent
                sc = smat[:, mi]
                mag = slope * sc + intercept
                rank_df = pd.DataFrame({"symbol": syms, "mag": mag})
                rank_df = rank_df[np.isfinite(mag)]
                if rank_df.empty:
                    tops[m] = None
                    continue
                pick = rank_df.sort_values("mag", ascending=False).head(TOP_N)
                tops[m] = pick
            base = tops.get(1.0)
            chosen = tops.get(CHOSEN)
            if base is None:
                continue
            base_syms = set(base["symbol"].astype(str))
            chosen_syms = set(chosen["symbol"].astype(str)) if chosen is not None else set()
            for m in MULTS:
                pick = tops.get(m)
                if pick is None or len(pick) < 5:
                    continue
                pr = np.array([point_rets(closes, s, D, H_K) for s in pick["symbol"]])
                ov_ref = len(set(pick["symbol"].astype(str)) & base_syms)
                ov_chosen = len(set(pick["symbol"].astype(str)) & chosen_syms)
                rows.append(
                    {
                        "date": D,
                        "board": board,
                        "m": m,
                        "config": f"w_smallmv_{m:g}",
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
                        "overlap": ov_ref,
                        "overlap_chosen": ov_chosen,
                    }
                )
        del day_info[D]
        gc.collect()
        if (di_abs + 1) % 50 == 0:
            print(f"  ... {di_abs + 1}/{len(eval_dates)} 日 ({time.time() - t0:.0f}s)", flush=True)

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    # ---------------- 汇总 + 扰动判定 ----------------
    print(f"\n===== small_mv_premium 权重乘子扰动检验 [{args.eval_days}d OOS] =====", flush=True)
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        ref_p10 = sub.loc[sub["m"] == 1.0, "pr10d"].mean()
        print(f"\n=== [{board}]  ref(m=1.0) pr10d={ref_p10:+.4f} ===", flush=True)
        print(f"  {'m':>5}{'pr10d':>9}{'pr5d':>9}{'win10d':>8}{'ov_ref':>8}"
              f"{'ov_chosen':>10}  Q1-Q4 pr10d(neg=*)/win vs ref", flush=True)
        # 季度边界 (用全期日期)
        qb = [daily["date"].min() + (daily["date"].max() - daily["date"].min()) * i / 4
              for i in range(5)]
        for m in MULTS:
            c = sub[sub["m"] == m]
            if c.empty:
                continue
            qpr, qwin = [], []
            for i in range(4):
                mask = (sub["date"] >= qb[i]) & (sub["date"] < qb[i + 1])
                cc_ = sub.loc[mask & (sub["m"] == m), "pr10d"]
                rr = sub.loc[mask & (sub["m"] == 1.0), "pr10d"]
                if cc_.empty or rr.empty:
                    continue
                p10c = cc_.mean()
                qpr.append(f"{p10c:+.3f}{'*' if p10c < 0 else ''}")
                qwin.append("W" if p10c > rr.mean() else ".")
            oc = c["overlap_chosen"].mean() if m != CHOSEN else float("nan")
            print(f"  {m:>5.1f}{c['pr10d'].mean():>+9.4f}{c['pr5d'].mean():>+9.4f}"
                  f"{c['win10d'].mean():>8.1%}{c['overlap'].mean():>8.2f}"
                  f"{oc:>10.2f}  {' '.join(qpr)} | {' '.join(qwin)}", flush=True)
        # 扰动判定 (文档 #8): m=2.0 ±20% 邻点与 2.0 清单平均重合 <5/10 → 弃用
        c_chosen = sub[sub["m"] == CHOSEN]
        for pm in PERTURB:
            c_pm = sub[sub["m"] == pm]
            if c_pm.empty or c_chosen.empty:
                continue
            ov = c_pm["overlap_chosen"].mean()
            tag = "PASS" if ov >= MIN_OVERLAP_CHOSEN else "FAIL(list大变)"
            print(f"  [扰动] m={CHOSEN:g} vs m={pm:g}: 平均重合 {ov:.2f}/10 → {tag}", flush=True)

    # 供 _diag_param_sweep_verify 复用 (需 date/board/config/pr5d/pr10d/win5d/win10d/overlap)
    summary = {
        "ts": ts,
        "script": "_diag_perturb_w_smallmv.py",
        "target": TARGET,
        "mults": list(MULTS),
        "chosen": CHOSEN,
        "perturb_neighbors": list(PERTURB),
        "min_overlap_chosen": MIN_OVERLAP_CHOSEN,
        "eval_days": args.eval_days,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
