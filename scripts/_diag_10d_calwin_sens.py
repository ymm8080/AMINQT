"""敏感度: 并行 TOP10 排名 mag_10d 校准参数扫描 (2026-08-07 用户).

用户在定案 T10 close-to-close 校准口径后, 唯一未定 = 校准窗口长度 cal_n
(现 160 交易日, 想知道 6m≈126 / 3m≈63 / 2m≈42 / 1m≈21 是否更好).
顺带检查 per_stock_window(现130) / per_stock_min_n(现30) / shrink_kappa(现40).

设计 (每个 config = 一次完整 walk-forward, 同一批评估日):
  1. 主扫描 cal_n ∈ {160,126,63,42,21}, 其余生产默认 (w=130, min_n=30, kappa=40)
     cal_n=160 为 BASELINE.
  2. 次要单变量 (围绕 baseline cal_n=160):
     per_stock_window ∈ {90,60}; per_stock_min_n ∈ {20}; shrink_kappa ∈ {20,80}
     (均相对 baseline 一次只动一个参数, 共 10 个唯一 config)

口径同 _diag_10d_c2c_vs_mfe.py:
  - 每股收缩回归 target=label_pm_10d_net (close[T+11]/close[T+1]-1-cost), 无前瞻
  - 逐板块按 mag_10d 降序取 TOP10, 评估 = close-to-close 点对点 pr_2d/3d/5d/10d
  - 头对头 = 逐评估日本 config TOP10 平均 pr 是否高于 BASELINE

RAM: 面板尾部 ONCE (pyarrow 列选择+行过滤), 全部 config 复用同一 frame, 单进程跑完.
WORM: BACKTEST_RESULT_DIR/diag_10d_calwin_sens_<ts>/
用法:
  python scripts/_diag_10d_calwin_sens.py [eval_days=250]
  python scripts/_diag_10d_calwin_sens.py 5 --smoke   # 快速迭代 (eval_days=5)
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

# --- 扫描参数 ---
CAL_NS = (160, 126, 63, 42, 21)  # 主扫描: 校准窗口长度 (交易日)
WINDOWS = (130, 90, 60)  # per_stock_window 候选 (130=生产默认)
MIN_NS = (30, 20)  # per_stock_min_n 候选 (30=生产默认)
KAPPAS = (40, 20, 80)  # shrink_kappa 候选 (40=生产默认)
BASELINE = dict(cal_n=160, per_stock_window=130, per_stock_min_n=30, shrink_kappa=40)
TOP_N = 10  # 每板块短名单档位
H_K = (2, 3, 5, 10)  # 点对点收益视界
BOARDS = ("main", "dual")
TARGET_COL = "label_pm_10d_net"  # c2c 校准目标 (close-to-close 净涨幅)

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


def build_configs() -> list[dict]:
    """10 个唯一 config (主扫描 5 + 次要单变量 5), 每个带 tag/group."""
    cfg_map: dict[tuple, dict] = {}

    def add(cal_n, w, mn, k, group):
        key = (cal_n, w, mn, k)
        if key in cfg_map:
            return
        cfg_map[key] = dict(
            cal_n=cal_n,
            per_stock_window=w,
            per_stock_min_n=mn,
            shrink_kappa=k,
            tag=f"cal{cal_n}_w{w}_mn{mn}_k{k}",
            group=group,
            baseline=(key == (BASELINE["cal_n"], BASELINE["per_stock_window"],
                              BASELINE["per_stock_min_n"], BASELINE["shrink_kappa"])),
        )

    # 主扫描: 只动 cal_n, 其余生产默认
    for cal_n in CAL_NS:
        add(cal_n, 130, 30, 40, "primary_cal_n")
    # 次要单变量 (围绕 baseline, 一次只动一个):
    for w in WINDOWS:  # 130 已含 (baseline), 跳过重复
        if w != 130:
            add(160, w, 30, 40, "secondary_per_stock_window")
    for mn in MIN_NS:  # 30 已含
        if mn != 30:
            add(160, 130, mn, 40, "secondary_per_stock_min_n")
    for k in KAPPAS:  # 40 已含
        if k != 40:
            add(160, 130, 30, k, "secondary_shrink_kappa")
    return list(cfg_map.values())


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


def _fit_mag_arr(
    day_sym: np.ndarray,
    day_scores: np.ndarray,
    by_sym: dict,
    cross_slope: float,
    cross_int: float,
    w: int,
    mn: int,
    kappa: int,
) -> np.ndarray:
    """每股收缩 mag = λ*raw_slope + (1-λ)*cross_slope, λ=len/(len+κ);
    len = 该股最近 w 行 (dropna 后), 不足 min_n → 回退横截面."""
    m_arr = np.full(len(day_sym), np.nan)
    for i, sym in enumerate(day_sym):
        sc = day_scores[i]
        if not np.isfinite(sc):
            continue
        xy = by_sym.get(sym)
        if xy is None:
            continue
        xs, ys = xy
        x = xs[-w:]
        y = ys[-w:]
        if len(x) >= mn:
            raw_slope, _ = _ols_slope_intercept(x, y)
            lam = len(x) / (len(x) + kappa)
            slope = lam * raw_slope + (1.0 - lam) * cross_slope
            intercept = float(y.mean()) - slope * float(x.mean())
        else:
            slope, intercept = cross_slope, cross_int
        m_arr[i] = slope * sc + intercept
    return m_arr


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    ap.add_argument("--smoke", action="store_true", help="快速冒烟: eval_days=5")
    args = ap.parse_args()
    eval_days = 5 if args.smoke else args.eval_days

    configs = build_configs()
    baseline = next(c for c in configs if c["baseline"])
    by_cal_n: dict[int, list[dict]] = {}
    for c in configs:
        by_cal_n.setdefault(c["cal_n"], []).append(c)
    max_cal_n = max(c["cal_n"] for c in configs)
    print(
        f"[cfg] {len(configs)} 唯一 config; BASELINE={baseline['tag']} "
        f"(cal_n={BASELINE['cal_n']}, w={BASELINE['per_stock_window']}, "
        f"mn={BASELINE['per_stock_min_n']}, kappa={BASELINE['shrink_kappa']})",
        flush=True,
    )

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_10d_calwin_sens_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = max_cal_n + eval_days + 40
    print(f"[panel] 尾部加载 {tail_days} 交易日 (单进程复用于全部 {len(configs)} config)...",
          flush=True)
    work = load_panel_tail(tail_days)
    all_dates = sorted(work["date"].unique())
    eval_dates = all_dates[-eval_days:]
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

    # walk-forward: 逐评估日 D, 各 cal_n 切 [D-cal_n, D-1] 校准窗 (复用),
    # 同一 (date, cal_n, board) 的 by_sym/横截面 只算一次, 各 config 只换收缩参数.
    print("[calib] walk-forward 逐日重拟合 × config 扫描 (c2c 目标)...", flush=True)
    rows: list[dict] = []
    skipped = 0
    for k, D in enumerate(eval_dates):
        di = date_idx[D]
        day = work[work["date"] == D]
        for cal_n, cal_cfgs in by_cal_n.items():
            cal_lo = all_dates[max(0, di - cal_n)]
            cal = work[(work["date"] >= cal_lo) & (work["date"] < D)]
            if cal.empty or day.empty:
                skipped += 1
                continue
            for board in BOARDS:
                calsub = cal[cal["board"] == board]
                daysub = day[day["board"] == board]
                if len(calsub) < 200 or daysub.empty:
                    skipped += 1
                    continue
                # 每股 (score, label) 有限序列 (dropna 等价) — 该 (date,cal_n,board) 只建一次
                m = np.isfinite(calsub["score"].to_numpy(float)) & np.isfinite(
                    calsub[TARGET_COL].to_numpy(float)
                )
                if m.sum() < 50:
                    skipped += 1
                    continue
                subf = calsub.loc[m, ["symbol", "score", TARGET_COL]]
                cross_slope, cross_int = _ols_slope_intercept(
                    subf["score"].to_numpy(float), subf[TARGET_COL].to_numpy(float)
                )
                by_sym: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for sym, g in subf.groupby("symbol", sort=False):
                    by_sym[str(sym)] = (
                        g["score"].to_numpy(float),
                        g[TARGET_COL].to_numpy(float),
                    )
                day_sym = daysub["symbol"].astype(str).to_numpy()
                day_scores = daysub["score"].to_numpy(float)

                # 各 config 打 mag → TOP10 → 评估
                top_sets: dict[str, set] = {}
                mags: dict[str, np.ndarray] = {}
                for cfg in cal_cfgs:
                    mag = _fit_mag_arr(
                        day_sym, day_scores, by_sym, cross_slope, cross_int,
                        cfg["per_stock_window"], cfg["per_stock_min_n"],
                        cfg["shrink_kappa"],
                    )
                    mags[cfg["tag"]] = mag
                    rk = pd.DataFrame({"symbol": day_sym, "mag": mag}).dropna(subset=["mag"])
                    top_sets[cfg["tag"]] = set(
                        rk.sort_values("mag", ascending=False).head(TOP_N)["symbol"]
                    )
                base_set = top_sets[baseline["tag"]]

                for cfg in cal_cfgs:
                    pick = list(top_sets[cfg["tag"]])
                    pr = np.array([point_rets(closes, s, D, H_K) for s in pick])
                    if len(pr) < 5:
                        continue
                    n_pick = len(pr)
                    win5 = pr[:, 2]
                    win10 = pr[:, 3]
                    rows.append(
                        {
                            "date": D,
                            "board": board,
                            "config": cfg["tag"],
                            "group": cfg["group"],
                            "cal_n": cfg["cal_n"],
                            "per_stock_window": cfg["per_stock_window"],
                            "per_stock_min_n": cfg["per_stock_min_n"],
                            "shrink_kappa": cfg["shrink_kappa"],
                            "baseline": cfg["baseline"],
                            "n": n_pick,
                            "pr2d": _safe_nanmean(pr[:, 0]),
                            "pr3d": _safe_nanmean(pr[:, 1]),
                            "pr5d": _safe_nanmean(pr[:, 2]),
                            "pr10d": _safe_nanmean(pr[:, 3]),
                            "win5d": float(np.sum(win5 > 0) / np.sum(~np.isnan(win5)))
                            if np.sum(~np.isnan(win5))
                            else float("nan"),
                            "win10d": float(np.sum(win10 > 0) / np.sum(~np.isnan(win10)))
                            if np.sum(~np.isnan(win10))
                            else float("nan"),
                            "overlap": len(set(pick) & base_set),
                        }
                    )
        del cal, day
        gc.collect()
        if (k + 1) % 50 == 0 or k + 1 == len(eval_dates):
            print(f"[calib] {k + 1}/{len(eval_dates)} 评估日 done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    daily = pd.DataFrame(rows)
    if daily.empty:
        raise SystemExit("no valid daily rows — check panel/board data")
    daily = daily.sort_values(["date", "board", "config"], ignore_index=True)
    daily.to_csv(out_dir / "daily.csv", index=False)
    print(f"[out] daily.csv {len(daily):,} 行 ({time.time() - t0:.0f}s)", flush=True)

    # 汇总: 逐 config×board 均值 + 日级上涨率 + 头对头 vs BASELINE
    print(f"\n===== mag_10d 校准参数敏感度 (最近 {eval_days} 交易日, c2c 目标) =====", flush=True)
    agg_rows: list[dict] = []
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}] 评估 = close-to-close 实得 (买=close[T+1], 卖=close[T+1+k])", flush=True)
        base_daily = sub[sub["baseline"]].set_index("date")[["pr5d", "pr10d"]]
        for cfg in configs:
            g = sub[sub["config"] == cfg["tag"]]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            g5 = g["pr5d"].dropna()
            g10 = g["pr10d"].dropna()
            day_win5 = float((g5 > 0).mean()) if len(g5) else float("nan")
            day_win10 = float((g10 > 0).mean()) if len(g10) else float("nan")
            # 头对头 vs baseline (共同评估日)
            gi = g.set_index("date")[["pr5d", "pr10d"]]
            common5 = base_daily["pr5d"].index.intersection(gi["pr5d"].index)
            common10 = base_daily["pr10d"].index.intersection(gi["pr10d"].index)
            b5 = base_daily.loc[common5, "pr5d"].to_numpy(float)
            c5 = gi.loc[common5, "pr5d"].to_numpy(float)
            v5 = np.isfinite(b5) & np.isfinite(c5)
            b10 = base_daily.loc[common10, "pr10d"].to_numpy(float)
            c10 = gi.loc[common10, "pr10d"].to_numpy(float)
            v10 = np.isfinite(b10) & np.isfinite(c10)
            w5 = int((c5[v5] > b5[v5]).sum()) if v5.sum() else 0
            w10 = int((c10[v10] > b10[v10]).sum()) if v10.sum() else 0
            star = " *BASE" if cfg["baseline"] else ""
            print(
                f"  {cfg['tag']:<26}{'(' + cfg['group'] + ')':<28}"
                f"2d {r['pr2d']:>+7.4f}  3d {r['pr3d']:>+7.4f}  "
                f"5d {r['pr5d']:>+7.4f}  10d {r['pr10d']:>+7.4f}  "
                f"5d涨率 {day_win5:>5.0%}  10d涨率 {day_win10:>5.0%}  "
                f"H2H 5d {w5}/{v5.sum()}  10d {w10}/{v10.sum()}  "
                f"重合 {float(g['overlap'].mean()):.1f}{star}",
                flush=True,
            )
            agg_rows.append(
                {
                    "board": board,
                    "config": cfg["tag"],
                    "group": cfg["group"],
                    "cal_n": cfg["cal_n"],
                    "per_stock_window": cfg["per_stock_window"],
                    "per_stock_min_n": cfg["per_stock_min_n"],
                    "shrink_kappa": cfg["shrink_kappa"],
                    "is_baseline": cfg["baseline"],
                    "n_days": int(len(g)),
                    "pr2d": round(float(r["pr2d"]), 5),
                    "pr3d": round(float(r["pr3d"]), 5),
                    "pr5d": round(float(r["pr5d"]), 5),
                    "pr10d": round(float(r["pr10d"]), 5),
                    "win5d_day": round(day_win5, 4),   # 日级: 平均 pr5d>0 的天数占比
                    "win10d_day": round(day_win10, 4),
                    "h2h_wins_pr5d": w5,               # 逐日 TOP10 平均 pr5d 高于 baseline 的天数
                    "h2h_n_pr5d": int(v5.sum()),
                    "h2h_wins_pr10d": w10,
                    "h2h_n_pr10d": int(v10.sum()),
                    "overlap_mean": round(float(g["overlap"].mean()), 3),
                }
            )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "eval_days": eval_days,
        "smoke": args.smoke,
        "metric": "close-to-close point returns (买=close[T+1], 卖=close[T+1+k]); NOT MFE",
        "calibration": "每股收缩回归 target=label_pm_10d_net; "
                       "mag = λ*raw_slope+(1-λ)*cross_slope, λ=len/(len+κ), "
                       "len=该股最近 per_stock_window 行, 不足 per_stock_min_n 回退横截面",
        "baseline": BASELINE,
        "baseline_tag": baseline["tag"],
        "scanned": {
            "cal_n": list(CAL_NS),
            "per_stock_window": list(WINDOWS),
            "per_stock_min_n": list(MIN_NS),
            "shrink_kappa": list(KAPPAS),
        },
        "configs": [
            {k: c[k] for k in ("tag", "group", "cal_n", "per_stock_window",
                               "per_stock_min_n", "shrink_kappa", "baseline")}
            for c in configs
        ],
        "definitions": {
            "daily.win5d/win10d": "当日该 config TOP10 内 pr5d/pr10d>0 的股票占比",
            "agg.win5d_day/win10d_day": "评估日间 平均 pr5d/pr10d>0 的天数占比",
            "agg.h2h_wins_pr*": "逐日 TOP10 平均 pr* 高于 BASELINE 的天数 / 共同有效日",
            "agg.overlap_mean": "TOP10 与本 config 的重合度 0-10 (baseline 与自身=10)",
        },
        "boards": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)
    print(f"总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
