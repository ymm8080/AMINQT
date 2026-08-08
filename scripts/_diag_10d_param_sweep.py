"""mag_10d 校准参数扫描: cal_n / per_stock_window / per_stock_min_n / shrink_kappa (2026-08-07).

背景: 生产定案 = 并行系统除慢牛外按 mag_10d (T+10 close-to-close 校准幅度)
全板块日截面排名选 TOP10. 校准 = 每股收缩回归 score→label_pm_10d_net,
横截面 OLS 兜底. 已有 cal_n 扫描结论 (diag_10d_cal_n_sweep_20260807_185450):
cal_n=42 两板块 5d/10d 实得均最优 (main 5d +8.35% vs 160 的 +6.32%).
本脚本在 cal_n=42 基础上逐个扰动其余参数, 回答"其他参数怎么定":

  参考配置 ref = (cal_n=42, per_stock_window=130, per_stock_min_n=30, shrink_kappa=40)
  逐参数一因子扰动 (其余固定 ref):
    cal_n ∈ {21, 63, 126, 160}
    per_stock_window ∈ {60, 90, 200}   (当前 130)
    per_stock_min_n  ∈ {15, 20, 50}    (当前 30)
    shrink_kappa     ∈ {10, 20, 80}    (当前 40)

评估口径 (同 cal_n 扫描): 校准目标 = label_pm_10d_net (c2c 净幅度); 每板块日截面
按 mag 降序取 TOP10; 实得 = 买 close_hfq[T+1], 卖 close_hfq[T+1+k], k∈(2,3,5,10);
相对 ref 逐日头对头 pr5d/pr10d 赢天数 + TOP10 平均重合度. walk-forward 无前瞻.

自包含: 禁止 import app.pipeline_parallel.backtest (并发修改中), 需要的 helper
add_mfe_labels / tradability_gate / add_pm_labels 全部内联. 不用 prepare_adx.

用法: python scripts/_diag_10d_param_sweep.py [eval_days=250]
WORM: BACKTEST_RESULT_DIR/diag_10d_param_sweep_<ts>/
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
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

# ---------------- 扫描设置 ----------------
REF = {
    "cal_n": 42,  # 校准窗 (交易日, ≈2 月, cal_n 扫描最优)
    "psw": 130,  # 每股自用最近交易日 (生产当前值)
    "minn": 30,  # 每股回归最小样本, 不足回退横截面 (生产当前值)
    "kappa": 40,  # 收缩强度 (生产当前值)
}
TOP_N = 10
H_K = (2, 3, 5, 10)
BOARDS = ("main", "dual")

# 逐参数一因子扰动 (param, 值列表) — 其余取 REF
PARAM_SWEEP = {
    "cal_n": (21, 63, 126, 160),
    "psw": (60, 90, 200),
    "minn": (15, 20, 50),
    "kappa": (10, 20, 80),
}
PARAM_DISPLAY = {
    "cal_n": "cal_n(校准窗)",
    "psw": "per_stock_window",
    "minn": "per_stock_min_n",
    "kappa": "shrink_kappa",
}
PARAM_CURRENT = {"cal_n": None, "psw": 130, "minn": 30, "kappa": 40}  # 生产当前值 (cal_n 已改 42)


def build_configs() -> list[dict]:
    """参考配置 + 每参数一因子扰动. 每个 config: {name, param, param_value, cal_n, psw, minn, kappa}."""
    cfg: list[dict] = []
    cfg.append({"name": "ref", "param": "ref", "param_value": 0, **REF})
    for p, vals in PARAM_SWEEP.items():
        for v in vals:
            d = dict(REF)
            d[p] = v
            cfg.append(
                {
                    "name": f"{p}_{v}",
                    "param": p,
                    "param_value": v,
                    "cal_n": d["cal_n"],
                    "psw": d["psw"],
                    "minn": d["minn"],
                    "kappa": d["kappa"],
                }
            )
    return cfg


CONFIGS = build_configs()
MAX_CAL_N = max(c["cal_n"] for c in CONFIGS)

POOL_COLS = [
    "amihud_illiq",
    "small_mv_premium",
    "amihud_illiquidity",
    "down_gap_pct",
    "VAR51",
    "ret_reversal_5d",
    "limit_dist_pct",
]
# 注: SNIPER/FUSION 池还含 pv_corr_5, 由 indicators.prepare_adx 计算; 本脚本不用
# prepare_adx, pool_score 会"缺列特征自动跳过" → 各 config 均同口径, 头对头仍有效.
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
    """内存安全尾部加载: pyarrow 列选择 + 行过滤 → c2c 标签 → tradability_gate → board."""
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
    return work


# ---------------- 内联 helper (从 _diag_10d_c2c_vs_mfe.py / backtest.py 抄) ----------------


def add_mfe_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """补算 MFE 净标签 (内联自 backtest.py; 本 c2c-only 扫描不调用, 随规格抄入)."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)
    max_off = max(horizons) + 1
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, max_off + 1)],
        axis=1,
        keys=range(2, max_off + 1),
    )
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for k in horizons:
        peak = shifts.loc[:, 2 : k + 1].max(axis=1, skipna=False)
        df[f"label_mfe_{k}d_net"] = peak / exec_px - 1 - cost_total
    del shifts
    gc.collect()
    return df


def tradability_gate(
    work: pd.DataFrame, lookback: int = 20, min_presence: float = 0.8
) -> tuple[pd.DataFrame, dict]:
    """PIT 可交易性门 (内联自 backtest.py): 前 lookback 日有行比例 < min_presence → 剔除."""
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
    stats = {
        "lookback": lookback,
        "min_presence": min_presence,
        "removed_rows": int((~keep).sum()),
        "removed_stocks": int(work.loc[~keep, "symbol"].nunique()),
        "kept_rows": int(keep.sum()),
        "kept_stocks": int(work.loc[keep, "symbol"].nunique()),
    }
    return work.loc[keep], stats


def add_pm_labels(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """补算 close-to-close 净标签: label_pm_{k}d_net = close_hfq[T+1+k]/close_hfq[T+1]-1-cost."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol")
    exec_px = g["close_hfq"].shift(-1)
    slip = df["adv20"].map(slippage_tier) if "adv20" in df.columns else 0.0015
    cost_total = COST + 2 * slip
    for k in horizons:
        sell_px = g["close_hfq"].shift(-(1 + k))
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
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else float("nan")


def point_rets(closes: dict, sym: str, d, ks: tuple[int, ...]) -> list[float]:
    """该股在日期 d 的前向点对点收益 (close[T+1+k]/close[T+1]-1); 数据不足→nan."""
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


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_10d_param_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = MAX_CAL_N + args.eval_days + 40
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

    print("[closes] 每股收盘价序列...", flush=True)
    cc = work[["symbol", "date", "close_hfq"]]
    d64 = cc["date"].to_numpy().astype("datetime64[ns]")
    closes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sym, g in cc.groupby("symbol", sort=False):
        closes[str(sym)] = (d64[g.index.to_numpy()], g["close_hfq"].to_numpy())
    del cc
    gc.collect()
    print(f"[closes] {len(closes):,} 只", flush=True)

    print("[score] 全池 pool_score (score=max(sniper,fusion))...", flush=True)
    work["score_sniper"] = np.nan
    work["score_fusion"] = np.nan
    for spec in (SNIPER, FUSION):
        col = f"score_{spec.name}"
        for board in BOARDS:
            bm = work["board"] == board
            work.loc[bm, col] = pool_score(work[bm], spec.pool).values
    work["score"] = work[["score_sniper", "score_fusion"]].max(axis=1)
    print(f"[score] done ({time.time() - t0:.0f}s)", flush=True)

    print("[stock] 每股预提取 (score, label_pm_10d_net, valid_pos)...", flush=True)
    stock = work[["symbol", "date", "score", "label_pm_10d_net"]]
    sd64 = stock["date"].to_numpy().astype("datetime64[ns]")
    sym_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, g in stock.groupby("symbol", sort=False):
        gd = sd64[g.index.to_numpy()]
        gs = g["score"].to_numpy(float)
        gy = g["label_pm_10d_net"].to_numpy(float)
        valid = np.isfinite(gs) & np.isfinite(gy)
        pos = np.nonzero(valid)[0].astype(np.int64)
        sym_data[str(sym)] = (gd, gs, gy, pos)
    del stock
    gc.collect()
    print(f"[stock] {len(sym_data):,} 只预提取完成 ({time.time() - t0:.0f}s)", flush=True)

    print("[cross] 每板块 score→label 前缀和...", flush=True)
    cross: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for board in BOARDS:
        bm = work[work["board"] == board]
        bvalid = np.isfinite(bm["score"].to_numpy(float)) & np.isfinite(
            bm["label_pm_10d_net"].to_numpy(float)
        )
        bd = bm["date"].to_numpy().astype("datetime64[ns]")[bvalid]
        bx = bm["score"].to_numpy(float)[bvalid]
        by = bm["label_pm_10d_net"].to_numpy(float)[bvalid]
        bo = np.argsort(bd, kind="stable")
        bd, bx, by = bd[bo], bx[bo], by[bo]
        psx = np.concatenate([[0.0], np.cumsum(bx)])
        psy = np.concatenate([[0.0], np.cumsum(by)])
        psxx = np.concatenate([[0.0], np.cumsum(bx * bx)])
        psxy = np.concatenate([[0.0], np.cumsum(bx * by)])
        cross[board] = (bd, psx, psy, psxx, psxy)
        del bm, bvalid, bd, bx, by
        gc.collect()
    print(f"[cross] done ({time.time() - t0:.0f}s)", flush=True)

    day_info: dict = {}
    for D in eval_dates:
        day = work[work["date"] == D]
        db = {}
        for board in BOARDS:
            sub = day[day["board"] == board]
            db[board] = (
                sub["symbol"].astype(str).to_numpy(),
                sub["score"].to_numpy(float),
            )
        day_info[D] = db
    print(f"[day] 逐日截面预提取 {len(day_info)} 日 ({time.time() - t0:.0f}s)", flush=True)

    def cross_slope_int(board: str, cal_lo, D):
        bd, psx, psy, psxx, psxy = cross[board]
        iLo = int(np.searchsorted(bd, np.datetime64(cal_lo), side="left"))
        iD = int(np.searchsorted(bd, np.datetime64(D), side="left"))
        n = iD - iLo
        if n < 50:
            return None
        Sx = psx[iD] - psx[iLo]
        Sy = psy[iD] - psy[iLo]
        Sxx = psxx[iD] - psxx[iLo]
        Sxy = psxy[iD] - psxy[iLo]
        var = n * Sxx - Sx * Sx
        if var <= 1e-12:
            return (0.0, Sy / n if n else 0.0)
        slope = (n * Sxy - Sx * Sy) / var
        return (slope, (Sy - slope * Sx) / n)

    # walk-forward: 每评估日 D × 每 config 拟合 → TOP10 (c2c 目标 label_pm_10d_net)
    print(f"[calib] walk-forward 逐日重拟合 × {len(CONFIGS)} configs...", flush=True)
    rows: list[dict] = []
    for D in eval_dates:
        di = date_idx[D]
        for board in BOARDS:
            syms, scores = day_info[D][board]
            if len(syms) == 0:
                continue
            tops: dict = {}
            for cfg in CONFIGS:
                cal_lo = all_dates[max(0, di - cfg["cal_n"])]
                cs_int = cross_slope_int(board, cal_lo, D)
                if cs_int is None:
                    tops[cfg["name"]] = None
                    continue
                cs, ci = cs_int
                cal_lo64 = np.datetime64(cal_lo)
                D64 = np.datetime64(D)
                mag_arr = np.full(len(syms), np.nan)
                for i in range(len(syms)):
                    sc = scores[i]
                    if not np.isfinite(sc):
                        continue
                    sd = sym_data.get(str(syms[i]))
                    if sd is None:
                        continue
                    gd, gs, gy, pos = sd
                    iD = int(np.searchsorted(gd, D64, side="left"))
                    iLo = int(np.searchsorted(gd, cal_lo64, side="left"))
                    vD = int(np.searchsorted(pos, iD, side="left"))
                    vLo = int(np.searchsorted(pos, iLo, side="left"))
                    vc = vD - vLo
                    if vc >= cfg["minn"]:
                        take = min(vc, cfg["psw"])
                        r = pos[vD - take : vD]
                        x = gs[r]
                        y = gy[r]
                        raw_slope, _ = _ols_slope_intercept(x, y)
                        lam = take / (take + cfg["kappa"])
                        slope = lam * raw_slope + (1 - lam) * cs
                        intercept = float(y.mean()) - slope * float(x.mean())
                    else:
                        slope, intercept = cs, ci
                    mag_arr[i] = slope * sc + intercept
                rank_df = pd.DataFrame(
                    {"symbol": syms, "score": scores, "mag": mag_arr}
                ).dropna(subset=["mag"])
                if rank_df.empty:
                    tops[cfg["name"]] = None
                    continue
                pick = rank_df.sort_values("mag", ascending=False).head(TOP_N)
                tops[cfg["name"]] = pick

            base = tops.get("ref")
            if base is None:
                continue
            base_syms = set(base["symbol"].astype(str))
            for cfg in CONFIGS:
                pick = tops.get(cfg["name"])
                if pick is None:
                    continue
                pr = np.array([point_rets(closes, s, D, H_K) for s in pick["symbol"]])
                if len(pr) < 5:
                    continue
                overlap = len(set(pick["symbol"].astype(str)) & base_syms)
                rows.append(
                    {
                        "date": D,
                        "board": board,
                        "param": cfg["param"],
                        "param_value": cfg["param_value"],
                        "config": cfg["name"],
                        "cal_n": cfg["cal_n"],
                        "psw": cfg["psw"],
                        "minn": cfg["minn"],
                        "kappa": cfg["kappa"],
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
        del day_info[D]
        gc.collect()
        if (eval_dates.index(D) + 1) % 50 == 0:
            print(
                f"  ... {eval_dates.index(D) + 1}/{len(eval_dates)} 日 "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    # ---------------- 汇总 ----------------
    ref_cfg = next(c for c in CONFIGS if c["name"] == "ref")
    print(f"\n===== mag_10d 校准参数扫描: TOP10 c2c 实得 "
          f"(最近 {args.eval_days} 交易日) =====", flush=True)
    print("评估 = close-to-close 点对点 (买=close[T+1], 卖=close[T+1+k]); "
          "校准 = 每股收缩回归, target=label_pm_10d_net", flush=True)
    print(
        f"参考配置 ref = (cal_n={ref_cfg['cal_n']}, psw={ref_cfg['psw']}, "
        f"minn={ref_cfg['minn']}, kappa={ref_cfg['kappa']}); 头对头 vs ref",
        flush=True,
    )

    agg_rows: list[dict] = []
    summary_boards: dict = {}

    def h2h_wins(g: pd.DataFrame, b: pd.DataFrame) -> tuple[int, int, int, int]:
        a = g.set_index("date")[["pr5d", "pr10d"]]
        common = a.index.intersection(b.index)
        w5 = w10 = m5 = m10 = 0
        if len(common):
            m5v = ~np.isnan(a.loc[common, "pr5d"].values) & ~np.isnan(
                b.loc[common, "pr5d"].values
            )
            m10v = ~np.isnan(a.loc[common, "pr10d"].values) & ~np.isnan(
                b.loc[common, "pr10d"].values
            )
            m5, m10 = int(m5v.sum()), int(m10v.sum())
            if m5:
                w5 = int(
                    (a.loc[common, "pr5d"].values[m5v] > b.loc[common, "pr5d"].values[m5v]).sum()
                )
            if m10:
                w10 = int(
                    (a.loc[common, "pr10d"].values[m10v] > b.loc[common, "pr10d"].values[m10v]).sum()
                )
        return w5, m5, w10, m10

    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        b = sub[sub["config"] == "ref"].set_index("date")
        summary_boards[board] = {}
        print(f"\n[{board}]  (ref = {ref_cfg['name']})", flush=True)
        for p, vals in PARAM_SWEEP.items():
            pname = PARAM_DISPLAY[p]
            cur = PARAM_CURRENT.get(p)
            cur_str = f" (生产 {cur})" if cur is not None else ""
            print(f"\n  ── {pname}{cur_str} ──", flush=True)
            print(
                f"  {'值':>8}{'日':>4}{'2d':>9}{'3d':>9}{'5d':>9}{'10d':>9}"
                f"{'5d涨率':>9}{'10d涨率':>9}{'重合':>7}",
                flush=True,
            )
            best5, best10 = None, None
            best5v, best10v = -1e9, -1e9
            # ref 本身也打印一次 (它不属于任何 param 组, 归入 p=首参组展示)
            shown_ref = False
            for _v in (("ref",) if not shown_ref else ()) + tuple(vals):
                pass
            for name, pv in [("ref", 0)] + [(f"{p}_{v}", v) for v in vals]:
                g = sub[sub["config"] == name]
                if g.empty:
                    continue
                r = g.mean(numeric_only=True)
                n = int(len(g))
                w5, m5, w10, m10 = h2h_wins(g, b)
                ov = float(g["overlap"].mean()) if not g["overlap"].isna().all() else float("nan")
                tag = f"{pv:>8}" if name != "ref" else "  ref "
                print(
                    f"  {tag}{n:>4}{r['pr2d']:>+9.4f}{r['pr3d']:>+9.4f}"
                    f"{r['pr5d']:>+9.4f}{r['pr10d']:>+9.4f}"
                    f"{r['win5d']:>9.1%}{r['win10d']:>9.1%}{ov:>7.1f}",
                    flush=True,
                )
                if name != "ref":
                    print(
                        f"      vs ref: 5d实得 {w5}/{m5} 天赢, 10d实得 {w10}/{m10} 天赢",
                        flush=True,
                    )
                row = {
                    "board": board,
                    "param": p,
                    "param_value": pv,
                    "config": name,
                    "n_days": n,
                    "pr2d": round(float(r["pr2d"]), 5),
                    "pr3d": round(float(r["pr3d"]), 5),
                    "pr5d": round(float(r["pr5d"]), 5),
                    "pr10d": round(float(r["pr10d"]), 5),
                    "win5d": round(float(r["win5d"]), 4),
                    "win10d": round(float(r["win10d"]), 4),
                    "overlap_mean": round(ov, 3),
                    "vs_ref_wins_pr5d": w5,
                    "vs_ref_n_days_5d": m5,
                    "vs_ref_wins_pr10d": w10,
                    "vs_ref_n_days_10d": m10,
                }
                agg_rows.append(row)
                if name != "ref":
                    if float(r["pr5d"]) > best5v:
                        best5, best5v = pv, float(r["pr5d"])
                    if float(r["pr10d"]) > best10v:
                        best10, best10v = pv, float(r["pr10d"])
            summary_boards[board][p] = {
                "best_by_pr5d": best5,
                "best_pr5d": round(best5v, 5),
                "best_by_pr10d": best10,
                "best_pr10d": round(best10v, 5),
                "ref_pr5d": round(float(b["pr5d"].mean()), 5),
                "ref_pr10d": round(float(b["pr10d"].mean()), 5),
            }
            if best5 is not None:
                print(
                    f"      → {pname}: 5d实得最优={best5} ({best5v:+.4f}, ref {summary_boards[board][p]['ref_pr5d']:+.4f}), "
                    f"10d实得最优={best10} ({best10v:+.4f}, ref {summary_boards[board][p]['ref_pr10d']:+.4f})",
                    flush=True,
                )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "eval_days": args.eval_days,
        "metric": "close-to-close point returns (买=close[T+1], 卖=close[T+1+k]); NOT MFE",
        "calibration": "每股收缩回归 (target=label_pm_10d_net, 回退横截面)",
        "reference_config": ref_cfg,
        "param_sweep": {p: list(v) for p, v in PARAM_SWEEP.items()},
        "top_n": TOP_N,
        "score_note": "score=max(sniper,fusion); 未用 prepare_adx → pv_corr_5 自动跳过 "
                      "(各 config 同口径, 头对头仍有效)",
        "boards": summary_boards,
        "agg": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
