"""诊断: 并行模块内 特征排名 vs 预测排名 选股对比 (2026-08-07 用户).

用户: "comparison on same module paraller.. whether feature ranking or
      prediction ranking will pick better stock in short list"

并行模块内两种短名单排名口径, 在同一宇宙上对比:
  A. 特征排名   score = max(狙击分位分, 融合分位分) 降序取 TOP-N
  B. 预测排名   pred_mag_h = 该股自己最近 130 日 (score→MFE) 收缩线性回归外推
                (生产 _shortlist_t5_t10.calibrate 同口径: 每股独立, 样本<30 回退横截面)
  C. 概率排名   pred_prob_h = 横截面 Platt P(MFE≥ABS_TARGET) — 理论上随 score 单调,
                作为对照列示 (预期与 A 相同).

walk-forward: 评估窗口 = 末 30 交易日; 每个评估日 D 用其前 130 交易日滚动重拟合
(校准不偷看评估). 评估口径 = 已实现 MFE (label_mfe_*_net), 选股门 = tradability_gate.

输出 (WORM) → BACKTEST_RESULT_DIR/parallel_rank_compare_<ts>/
  picks_metrics.csv  每(板块,排名视界,方法) 汇总: 平均 MFE/上涨率/达标率
  overlap.csv        每(板块,排名视界,日) A∩B 重合数 (两种排名到底差多少)
  daily.csv          逐日明细
  summary.json
用法: python scripts/_diag_parallel_rank_compare.py
"""

from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline_parallel import indicators
from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.config import FUSION, PANEL, SNIPER, board_of
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

CAL_N = 160      # 滚动校准窗口长度 (交易日)
EVAL_DAYS = 30   # 评估窗口
PER_STOCK_WINDOW = 130  # 与生产一致: 每股自用最近交易日
PER_STOCK_MIN_N = 30    # 与生产一致: 每股回归最小样本, 不足回退横截面
SHRINK_KAPPA = 40       # 与生产一致: 收缩强度
TOP_N = 10              # 每板块短名单档位 (与交付 10 main + 10 dual 对齐)
ABS_TARGET = {"2d": 0.02, "3d": 0.03, "5d": 0.04, "10d": 0.06}
HORIZONS = ("3d", "2d", "5d", "10d")
CLS_THRESHOLD = 0.005

POOL_COLS = [
    "amihud_illiq", "small_mv_premium", "amihud_illiquidity",
    "down_gap_pct", "VAR51", "ret_reversal_5d", "limit_dist_pct",
]
NEEDED_COLS = sorted(
    set(
        ["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "volume",
         "turnover_rate", "adv20"]
        + POOL_COLS
    )
)


def load_panel_tail(tail_days: int = CAL_N + 40) -> pd.DataFrame:
    """内存安全尾部加载: pyarrow 列选择 + 行过滤 → add_mfe_labels → tradability_gate
    → board → prepare_adx (补 pv_corr_5, checkpoint 无此列)."""
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
    work = add_mfe_labels(work, horizons=(2, 3, 5, 10))
    work, _gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    work = indicators.prepare_adx(work)
    return work


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """单变量最小二乘 (numpy, 快速路径)."""
    xm = float(x.mean())
    ym = float(y.mean())
    var = float(((x - xm) ** 2).sum())
    if var <= 1e-12:
        return 0.0, ym
    slope = float(((x - xm) * (y - ym)).sum() / var)
    return slope, ym - slope * xm


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"parallel_rank_compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[panel] 内存安全尾部加载...", flush=True)
    work = load_panel_tail()
    all_dates = sorted(work["date"].unique())
    eval_dates = all_dates[-EVAL_DAYS:]
    cal_window_days = CAL_N  # 每个评估日使用其前 CAL_N 个交易日
    print(
        f"[panel] {len(work):,}r / {work['symbol'].nunique():,}只 / "
        f"{len(all_dates)} 交易日 | 评估 {eval_dates[0].date()}..{eval_dates[-1].date()} "
        f"({len(eval_dates)} 日) ({time.time() - t0:.0f}s)",
        flush=True,
    )
    date_idx = {d: i for i, d in enumerate(all_dates)}

    # ── 全池逐系统打分 + score=max(狙击,融合) ──
    print("[score] 全池 pool_score...", flush=True)
    work["score_sniper"] = np.nan
    work["score_fusion"] = np.nan
    for spec in (SNIPER, FUSION):
        col = f"score_{spec.name}"
        for board in ("main", "dual"):
            bm = work["board"] == board
            work.loc[bm, col] = pool_score(work[bm], spec.pool).values
    work["score"] = work[["score_sniper", "score_fusion"]].max(axis=1)
    print(f"[score] done ({time.time() - t0:.0f}s)", flush=True)

    # 预计算已实现 MFE 列 (每 horizon 一个)
    real = {h: f"label_mfe_{h}_net" for h in HORIZONS}
    if not all(c in work.columns for c in real.values()):
        print(f"[fatal] 缺 MFE 标签列: {[c for c in real.values() if c not in work.columns]}")
        return

    # ── walk-forward: 每评估日 D 用 [D-CAL_N, D-1] 拟合 → 对 D 打分排名 ──
    print("[calib] walk-forward 逐日重拟合 + 双口径排名...", flush=True)
    rows_daily = []  # 逐日明细
    rows_overlap = []
    dump_frames = []  # 逐日逐股 (score, mag/prob_h, 已实现 MFE) 供两段式策略离线复测
    summary: dict = {
        "ts": ts, "type": "parallel_rank_compare",
        "cal_days": cal_window_days, "eval_days": len(eval_dates),
        "top_n": TOP_N, "per_stock": {"window": PER_STOCK_WINDOW,
                                      "min_n": PER_STOCK_MIN_N, "kappa": SHRINK_KAPPA},
        "boards": {},
    }

    for D in eval_dates:
        di = date_idx[D]
        cal_lo = all_dates[max(0, di - cal_window_days)]
        cal = work[(work["date"] >= cal_lo) & (work["date"] < D)]
        day = work[work["date"] == D].copy()
        if cal.empty or day.empty:
            continue
        for board in ("main", "dual"):
            calsub = cal[cal["board"] == board]
            daysub = day[day["board"] == board]
            if len(calsub) < 200 or daysub.empty:
                continue
            day_scores = daysub["score"].to_numpy(float)
            day_sym = daysub["symbol"].astype(str).to_numpy()
            by_sym = {k: v for k, v in calsub.groupby("symbol", sort=False)}
            mag_all: dict[str, np.ndarray] = {}
            prob_all: dict[str, np.ndarray] = {}
            for h in HORIZONS:
                realcol = real[h]
                c = calsub[["score", realcol]]
                if c[realcol].isna().all():
                    continue
                xcal = c["score"].to_numpy(float)
                ycal = c[realcol].to_numpy(float)
                m = np.isfinite(xcal) & np.isfinite(ycal)
                if m.sum() < 50:
                    continue
                xcal, ycal = xcal[m], ycal[m]
                # 横截面回归 + Platt (prob 单调于 score, 对照用)
                cross_slope, cross_int = _ols_slope_intercept(xcal, ycal)
                hit = (ycal >= ABS_TARGET[h]).astype(int)
                plat = LogisticRegression(max_iter=1000)
                plat.fit(xcal.reshape(-1, 1), hit)
                # 每股收缩回归
                mag = np.full(len(daysub), np.nan)
                prob = np.full(len(daysub), np.nan)
                for i, sym in enumerate(day_sym):
                    sc = day_scores[i]
                    if not np.isfinite(sc):
                        continue
                    ps = by_sym.get(sym)
                    if ps is None or ps.empty:
                        continue
                    gg = ps[["score", realcol]].dropna().tail(PER_STOCK_WINDOW)
                    if len(gg) >= PER_STOCK_MIN_N:
                        x = gg["score"].to_numpy(float)
                        y = gg[realcol].to_numpy(float)
                        raw_slope, _ = _ols_slope_intercept(x, y)
                        lam = len(gg) / (len(gg) + SHRINK_KAPPA)
                        slope = lam * raw_slope + (1 - lam) * cross_slope
                        intercept = float(y.mean()) - slope * float(x.mean())
                    else:
                        slope, intercept = cross_slope, cross_int
                    mag[i] = slope * sc + intercept
                    prob[i] = float(plat.predict_proba([[sc]])[0, 1])

                rank_df = pd.DataFrame(
                    {"symbol": day_sym, "score": day_scores, f"mag_{h}": mag,
                     f"prob_{h}": prob}
                )
                mag_all[h] = mag
                prob_all[h] = prob
                # A 特征排名 / B 预测(mag)排名 / C 概率排名 (对照)
                F = rank_df.sort_values("score", ascending=False).head(TOP_N)
                P = rank_df.sort_values(f"mag_{h}", ascending=False).head(TOP_N)
                C = rank_df.sort_values(f"prob_{h}", ascending=False).head(TOP_N)
                # 两种排名重合度
                fs, ps_ = set(F["symbol"]), set(P["symbol"])
                overlap = len(fs & ps_)
                rows_overlap.append(
                    {"board": board, "h": h, "date": str(D.date()),
                     "n_overlap": overlap, "n_total": TOP_N,
                     "overlap_pct": overlap / TOP_N}
                )
                # 已实现 MFE
                lab = daysub.set_index("symbol")[realcol]
                for name, pick in (("F_feature", F), ("P_predmag", P), ("C_predprob", C)):
                    syms = pick["symbol"]
                    y = lab.reindex(syms).dropna()
                    if y.empty:
                        continue
                    win = (y > 0).sum()
                    (y >= ABS_TARGET[h]).sum()
                    rows_daily.append(
                        {"board": board, "h": h, "date": str(D.date()),
                         "method": name, "n": int(len(y)),
                         "mfe": float(y.mean()), "win": float(win / len(y))}
                    )
                # Spearman(score, mag) 当日截面 (排名差异有多大)
                sm = rank_df.dropna(subset=["score", f"mag_{h}"])
                rho = float("nan")
                if len(sm) >= 10 and sm[f"mag_{h}"].nunique() > 1:
                    rho = float(spearmanr(sm["score"], sm[f"mag_{h}"]).statistic)
                sd = summary.setdefault("boards", {}).setdefault(board, {}).setdefault(h, {})
                sd["spearman_days"] = sd.get("spearman_days", []) + [rho]
            # 逐日逐股帧 (两段式策略离线复测底稿)
            dump = pd.DataFrame({"symbol": day_sym, "score": day_scores})
            dump["date"] = D
            dump["board"] = board
            lab_map = daysub.set_index("symbol")
            for h in HORIZONS:
                dump[f"mag_{h}"] = mag_all.get(h)
                dump[f"prob_{h}"] = prob_all.get(h)
                lab = f"label_mfe_{h}_net"
                if lab in lab_map.columns:
                    dump[f"real_{h}"] = lab_map[lab].reindex(day_sym).to_numpy()
            dump_frames.append(dump)
    print(f"[eval] 逐日排名完成 ({time.time() - t0:.0f}s)", flush=True)

    # ── 汇总 ──
    daily = pd.DataFrame(rows_daily)
    if daily.empty:
        print("[fatal] 无任何评估结果 — 检查 MFE 标签/评估窗口")
        return
    dumpdf = pd.concat(dump_frames, ignore_index=True)
    dumpdf.to_parquet(out_dir / "rank_daily.parquet", index=False)
    print(f"[dump] rank_daily.parquet {len(dumpdf):,}r", flush=True)
    agg = (
        daily.groupby(["board", "h", "method"])
        .agg(n_days=("date", "nunique"), n=("n", "sum"),
             mfe_mean=("mfe", "mean"), win_pct=("win", "mean"))
        .reset_index()
    )
    hit = (
        daily.groupby(["board", "h", "method"])["mfe"]
        .apply(lambda s: float((s > CLS_THRESHOLD).mean()))
        .rename("hit_cls_pct")
    )
    agg = agg.merge(hit, on=["board", "h", "method"])
    agg.to_csv(out_dir / "picks_metrics.csv", index=False)
    daily.to_csv(out_dir / "q_daily.csv", index=False)
    pd.DataFrame(rows_overlap).to_csv(out_dir / "overlap.csv", index=False)

    print("\n========== 并行模块内: 特征排名 vs 预测排名 (每板块 TOP-10) ==========", flush=True)
    print(f"    {'板块':<5}{'视界':>5}{'方法':<13}{'日':>4}{'均MFE':>9}{'上涨率':>9}{'达标率':>9}", flush=True)
    for b in ("main", "dual"):
        for h in HORIZONS:
            sub = agg[(agg["board"] == b) & (agg["h"] == h)]
            if sub.empty:
                continue
            rho = np.nanmean(summary["boards"][b][h]["spearman_days"]) if h in summary["boards"].get(b, {}) else np.nan
            for _, r in sub.iterrows():
                print(
                    f"    {b:<5}{h:>5}{r['method']:<13}{int(r['n_days']):>4}"
                    f"{r['mfe_mean']:>+9.4f}{r['win_pct']:>9.1%}{r['hit_cls_pct']:>9.1%}",
                    flush=True,
                )
            line = sub[sub["method"] == "F_feature"]
            pln = sub[sub["method"] == "P_predmag"]
            if not line.empty and not pln.empty:
                F, P = line.iloc[0], pln.iloc[0]
                print(
                    f"      → 预测排名 vs 特征排名: ΔMFE {P['mfe_mean']-F['mfe_mean']:+.4f} "
                    f"Δ上涨率 {P['win_pct']-F['win_pct']:+.1%} "
                    f"(Spearman(score,mag) {rho:+.3f})", flush=True
                )
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
