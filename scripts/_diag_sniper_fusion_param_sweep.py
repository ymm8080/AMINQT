"""SNIPER/FUSION 模块参数扫描 (2026-08-08).

用户: "focus on sniper and fusion module parameters, there are so many you can sweep".
第一轮 (2026-08-08, ±2× 单特征权重阶梯 + score combo + top_n) 判定: 等权近最优,
  减半显著伤 / 翻倍不显著 → 响应不对称 = 信号饱和; 单板显著不采信 (38 检验 ~2 假阳性).
第二轮 (2026-08-08, 细权重 {0.75/1.25/1.5} + amihud 近重复对再平衡 + 核心3vs尾部) 判定:
  响应面平坦于噪声 → 等权保持, 权重优化停止 (diag_sniper_fusion_param_sweep_20260808_070230).
第三轮 (2026-08-08, c2c leave-one-out 池重审): 唯一双板全窗稳定赢家 = 剔除
  down_gap_pct (MFE 目标选入但 c2c 不兑现收盘) → 已落地生产池 7→6 (见同目录 verdict.json).
本轮 (第四轮) 在**剔除 down_gap_pct 后的生产池**上测**池特征加法** (c2c 口径):
  每个候选特征同时加入两池 (等权), 看 OOS 是否边际增益 — 候选 = pv_sync 族
  (corr_marginal 文档化, MFE 时代测过 c2c 未重测) + down_gap 对称特征
  (up_gap/limit_down_dist, 验证跳空族 c2c 行为) + 趋势/筹码代表 (MA20_dist/bias_5d/winner_ratio).

口径 = 生产: 剔除后 6 特征 (prepare_adx 复刻, 含 pv_corr_5), walk-forward 无前瞻 mag10d
  校准 (cal_n=21, 拟合窗 [D-cal_n, D-REALIZED_DROP) 只用已实现标签, cross_min_n=50),
  合并排名 = 组合 score 经每股收缩回归 → 日截面降序 → TOP-N.
实得收益: point_rets 原始 c2c (与既有诊断一致, 成本对全部配置恒定 → 相对比较有效),
  另记 pr10d_net (label_pm_10d_net, 含成本, 生产实得口径).

判定 (依据《AI扫参风险》#1 子窗口 + #8 扰动):
  - 子窗口: 季度 pr10d 无负季度 且 ≥2/4 季度赢 ref
  - 扰动粗筛: 与生产 ref (等权+max+top10) 每日清单平均重合度 (权重 ×2/×0.5 是
    比文档 ±20% 更剧烈的扰动 → 重合度仍高 = 真实稳定; 重合 <50% = 换手/边界噪声风险)
  - 稳定 > 最高: 收益增益 < ~0.3pp 判噪声, 不改生产

自包含 (禁 import backtest, helper 内联). 用法: python scripts/_diag_sniper_fusion_param_sweep.py [eval_days=250]
WORM: BACKTEST_RESULT_DIR/diag_sniper_fusion_param_sweep_<ts>/
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
CAL_N = 21  # 生产 MAG10D_CAL cal_n
CROSS_MIN_N = 50  # 生产 cross_min_n
CUTS = (5, 10, 15, 20)

# ── 第四轮扫描配置 (c2c 池特征加法, 2026-08-08) ──
# 前三轮 (权重×2 + LOO 剔除 down_gap_pct) 判定见同目录 verdict.json。
# 生产池已剔 down_gap_pct → SNIPER/FUSION 池 7→6。此轮在剔除后的生产池上测"加法":
# 每个候选特征同时加入两池 (等权, 与生产口径一致), 看 c2c OOS 是否边际增益。
# 候选来源: corr_marginal 文档化 pv_sync 族 (MFE 时代测过, c2c 未重测) +
# down_gap 对称特征 (up_gap/limit_down_dist, 验证跳空族 c2c 行为) +
# 趋势/筹码代表 (MA20_dist/bias_5d/winner_ratio)。
ADD_CANDIDATES: tuple[str, ...] = (
    "pv_sync_direct_5d",
    "pv_sync_direct_20d",
    "pv_sync_5d",
    "pv_sync_20d",
    "up_gap_pct",
    "limit_down_dist_pct",
    "MA20_dist",
    "bias_5d",
    "winner_ratio",
)

# 两池并集特征 (生产口径: load_panel 经 prepare_adx 加 pv_corr_5)
# 第四轮在剔除 down_gap_pct 后的生产池上测加法 → 把 ADD_CANDIDATES 一并纳入 rank 预计算
_BASE_UNION = (
    "amihud_illiq",
    "amihud_illiquidity",
    "small_mv_premium",
    "down_gap_pct",
    "VAR51",
    "ret_reversal_5d",
    "pv_corr_5",
    "limit_dist_pct",
)
UNION = list(_BASE_UNION) + [c for c in ADD_CANDIDATES if c not in _BASE_UNION]

CONFIGS: list[dict] = []


def _add(
    name: str,
    weights: dict | None = None,
    combo: str = "max",
    drop: tuple[str, ...] = (),
    add: tuple[str, ...] = (),
) -> None:
    CONFIGS.append(
        {
            "name": name,
            "weights": dict(weights or {}),
            "combo": combo,
            "drop": tuple(drop),
            "add": tuple(add),
        }
    )


_add("ref_eq_max")  # 生产剔除后全池: 等权 + max(sniper, fusion)
# 第四轮: 每候选特征加入两池 (一次一个, 测单特征边际)
for f in ADD_CANDIDATES:
    _add(f"add_{f}", add=(f,))
# 参考对照: 加最相关的 pv_corr_5 本身 (已在池内, 再加固 → 应无增益, 验证方法)

# 面板本身无 pv_corr_5 (prepare_adx 后才有); 读列不含它, 权重扫描仍覆盖 (prepare_adx 后存在)
POOL_COLS = [c for c in UNION if c != "pv_corr_5"]
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
    # 生产 load_panel 会调 prepare_adx → 加 pv_corr_5; 复刻以在真实 7 特征集上扫
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


def weighted_score(
    r_arrays: dict[str, np.ndarray], pool: list[str], weights: dict
) -> np.ndarray:
    ws = [weights.get(c, 1.0) for c in pool]
    total = sum(ws)
    return np.sum([w * r_arrays[c] for w, c in zip(ws, pool)], axis=0) / total


def combined_score(sniper: np.ndarray, fusion: np.ndarray, combo: str) -> np.ndarray:
    if combo == "max":
        return np.fmax(sniper, fusion)
    if combo == "mean":
        return 0.5 * (sniper + fusion)
    if combo == "sniper":
        return sniper
    if combo == "fusion":
        return fusion
    raise ValueError(combo)


def build_cross(work: pd.DataFrame, score_arrs: dict[str, np.ndarray]) -> dict:
    cross: dict = {}
    d64 = work["date"].to_numpy().astype("datetime64[ns]")
    y = work["label_pm_10d_net"].to_numpy(float)
    board_val = work["board"].values
    for board in BOARDS:
        bm = board_val == board
        by: dict = {}
        for name, x in score_arrs.items():
            valid = np.isfinite(x) & np.isfinite(y) & bm
            d = d64[valid]
            if len(d) == 0:
                by[name] = None
                continue
            bd, bin_idx = np.unique(d, return_inverse=True)
            nb = len(bd)
            xv = x[valid]
            yv = y[valid]
            by[name] = (
                bd,
                np.concatenate(
                    [[0.0], np.cumsum(np.bincount(bin_idx, minlength=nb).astype(float))]
                ),
                np.concatenate(
                    [[0.0], np.cumsum(np.bincount(bin_idx, weights=xv, minlength=nb))]
                ),
                np.concatenate(
                    [[0.0], np.cumsum(np.bincount(bin_idx, weights=yv, minlength=nb))]
                ),
                np.concatenate(
                    [
                        [0.0],
                        np.cumsum(np.bincount(bin_idx, weights=xv * xv, minlength=nb)),
                    ]
                ),
                np.concatenate(
                    [
                        [0.0],
                        np.cumsum(np.bincount(bin_idx, weights=xv * yv, minlength=nb)),
                    ]
                ),
            )
        cross[board] = by
        del bm
        gc.collect()
    return cross


def cross_slope_int(ent, cal_lo, cutoff_date):
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


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_sniper_fusion_param_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = CAL_N + REALIZED_DROP + args.eval_days + 40
    print(f"[panel] 尾部加载 {tail_days} 交易日...", flush=True)
    work = load_panel_tail(tail_days)
    sniper_avail = _avail(work, SNIPER.pool)
    fusion_avail = _avail(work, FUSION.pool)
    assert sniper_avail and fusion_avail, (
        f"avail 为空: sniper={sniper_avail} fusion={fusion_avail}"
    )
    print(
        f"[avail] sniper={len(sniper_avail)}特征 {sniper_avail} | "
        f"fusion={len(fusion_avail)}特征 {fusion_avail}",
        flush=True,
    )
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

    print(
        f"[score] 预计算 {len(UNION)} 特征截面分位 + {len(CONFIGS)} 配置组合 score...",
        flush=True,
    )
    r_all = {c: cross_rank(work, c).to_numpy(float) for c in UNION}
    # r_s/r_f 需覆盖 ADD_CANDIDATES (第四轮加法): 池特征 + 候选都在 UNION 内
    r_s = {c: r_all[c] for c in UNION if c in work.columns}
    r_f = dict(r_s)
    del r_all
    gc.collect()
    score_arrs: dict[str, np.ndarray] = {}
    for cfg in CONFIGS:
        s_avail = [c for c in sniper_avail if c not in cfg["drop"]]
        f_avail = [c for c in fusion_avail if c not in cfg["drop"]]
        for c in cfg["add"]:
            if c in r_s and c not in s_avail:
                s_avail.append(c)
            if c in r_f and c not in f_avail:
                f_avail.append(c)
        if not s_avail or not f_avail:
            raise ValueError(f"配置 {cfg['name']} drop={cfg['drop']} 导致空池")
        sn = weighted_score(r_s, s_avail, cfg["weights"])
        fu = weighted_score(r_f, f_avail, cfg["weights"])
        score_arrs[cfg["name"]] = combined_score(sn, fu, cfg["combo"])
        del sn, fu
    del r_s, r_f
    gc.collect()

    print("[cross] 每 (板块, 配置) 日期桶前缀和...", flush=True)
    cross = build_cross(work, score_arrs)
    y_label = work["label_pm_10d_net"].to_numpy(float)

    print(f"[day] 逐日截面预提取 ({len(CONFIGS)} 配置)...", flush=True)
    day_info: dict = {}
    for D in eval_dates:
        day = work[work["date"] == D]
        db = {}
        for board in BOARDS:
            sub = day[day["board"] == board]
            if sub.empty:
                db[board] = None
                continue
            smat = np.column_stack([score_arrs[c["name"]][sub.index] for c in CONFIGS])
            db[board] = (
                sub["symbol"].astype(str).to_numpy(),
                smat,
                y_label[sub.index],
            )
        day_info[D] = db
    del work, day, score_arrs
    gc.collect()
    print(f"[day] {len(day_info)} 日预提取完成 ({time.time() - t0:.0f}s)", flush=True)

    print(
        f"[calib] walk-forward 逐日重拟合 × {len(CONFIGS)} 配置 (无前瞻窗)...",
        flush=True,
    )
    rows: list[dict] = []
    ref_cfg = CONFIGS[0]["name"]
    for di_abs, D in enumerate(eval_dates):
        di = date_idx[D]
        for board in BOARDS:
            ent = day_info[D].get(board)
            if ent is None:
                continue
            syms, smat, yday = ent
            tops: dict[str, pd.DataFrame | None] = {}
            for ci, cfg in enumerate(CONFIGS):
                name = cfg["name"]
                if di < max(CAL_N, REALIZED_DROP):
                    tops[name] = None
                    continue
                cal_lo = all_dates[di - CAL_N]
                cutoff_date = all_dates[di - REALIZED_DROP]
                cs_ent = cross_slope_int(cross[board].get(name), cal_lo, cutoff_date)
                if cs_ent is None or cs_ent[2] < CROSS_MIN_N:
                    tops[name] = None
                    continue
                slope, intercept, _ = cs_ent
                sc = smat[:, ci]
                mag = slope * sc + intercept
                rank_df = pd.DataFrame({"symbol": syms, "mag": mag})
                rank_df = rank_df[np.isfinite(mag)]
                if rank_df.empty:
                    tops[name] = None
                    continue
                pick = rank_df.sort_values("mag", ascending=False).head(max(CUTS))
                tops[name] = pick
            base = tops.get(ref_cfg)
            if base is None:
                continue
            pos = {s: i for i, s in enumerate(syms)}
            for cfg in CONFIGS:
                pick_all = tops.get(cfg["name"])
                if pick_all is None or len(pick_all) < min(CUTS):
                    continue
                for k in CUTS:
                    pick = pick_all.head(k)
                    pr = np.array(
                        [point_rets(closes, s, D, H_K) for s in pick["symbol"]]
                    )
                    # 净 10d (生产实得口径): 从决策行 label 取 (含成本, 与 syms 同序)
                    pick_syms = pick["symbol"].astype(str).to_numpy()
                    net10 = _safe_nanmean(yday[np.array([pos[s] for s in pick_syms])])
                    base_syms = set(base.head(k)["symbol"].astype(str))
                    ov = len(set(pick_syms) & base_syms)
                    rows.append(
                        {
                            "date": D,
                            "board": board,
                            "config": cfg["name"],
                            "combo": cfg["combo"],
                            "n": k,
                            "pr2d": _safe_nanmean(pr[:, 0]),
                            "pr3d": _safe_nanmean(pr[:, 1]),
                            "pr5d": _safe_nanmean(pr[:, 2]),
                            "pr10d": _safe_nanmean(pr[:, 3]),
                            "pr10d_net": net10,
                            "win5d": float(
                                (pr[:, 2] > 0).sum() / (~np.isnan(pr[:, 2])).sum()
                            )
                            if (~np.isnan(pr[:, 2])).sum()
                            else float("nan"),
                            "win10d": float(
                                (pr[:, 3] > 0).sum() / (~np.isnan(pr[:, 3])).sum()
                            )
                            if (~np.isnan(pr[:, 3])).sum()
                            else float("nan"),
                            "overlap_ref": ov,
                        }
                    )
        del day_info[D]
        gc.collect()
        if (di_abs + 1) % 50 == 0:
            print(
                f"  ... {di_abs + 1}/{len(eval_dates)} 日 ({time.time() - t0:.0f}s)",
                flush=True,
            )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    # ---------------- 汇总 + 判定 ----------------
    print(
        f"\n===== SNIPER/FUSION 模块参数扫描 [{args.eval_days}d OOS] =====", flush=True
    )
    summary: dict = {}
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        sub10 = sub[sub["n"] == 10]  # 汇总基准 = 生产 top10 档
        ref = sub10[sub10["config"] == ref_cfg]
        ref_p10 = ref["pr10d"].mean()
        ref_n10 = ref["win10d"].mean()
        print(
            f"\n=== [{board}]  ref(等权,max,top10) pr10d={ref_p10:+.4f} win10d={ref_n10:.1%} ===",
            flush=True,
        )
        print(
            f"  {'config':<22}{'pr10d':>9}{'pr5d':>9}{'win10d':>8}{'ov_ref':>8}  "
            f"Q1-Q4 pr10d(neg=*)/win vs ref",
            flush=True,
        )
        qb = [
            daily["date"].min() + (daily["date"].max() - daily["date"].min()) * i / 4
            for i in range(5)
        ]
        board_res: dict = {}
        for cfg in CONFIGS:
            name = cfg["name"]
            c = sub10[sub10["config"] == name]
            if c.empty:
                continue
            qpr, qwin = [], []
            for i in range(4):
                mask = (sub10["date"] >= qb[i]) & (sub10["date"] < qb[i + 1])
                cc_ = sub10.loc[mask & (sub10["config"] == name), "pr10d"]
                rr = sub10.loc[mask & (sub10["config"] == ref_cfg), "pr10d"]
                if cc_.empty or rr.empty:
                    continue
                p10c = cc_.mean()
                qpr.append(f"{p10c:+.3f}{'*' if p10c < 0 else ''}")
                qwin.append("W" if p10c > rr.mean() else ".")
            # 稳定性: 无负季度 且 ≥2/4 季度赢 ref
            n_win = qwin.count("W")
            gain = c["pr10d"].mean() - ref_p10
            stable = (not any(q.startswith("-") for q in qpr)) and n_win >= 2
            ov = c["overlap_ref"].mean()
            board_res[name] = {
                "pr10d": c["pr10d"].mean(),
                "pr5d": c["pr5d"].mean(),
                "pr10d_net": c["pr10d_net"].mean(),
                "win10d": c["win10d"].mean(),
                "overlap_ref": ov,
                "gain_vs_ref": gain,
                "n_win_quarters": n_win,
                "stable": stable,
                "quarters": [q for q in qpr],
            }
            tag = ""
            if gain > 0 and stable:
                tag = " STABLE_WIN"
            elif 0 < gain <= 0.003:
                tag = " (噪声增益<0.3pp)"
            print(
                f"  {name:<22}{c['pr10d'].mean():>+9.4f}{c['pr5d'].mean():>+9.4f}"
                f"{c['win10d'].mean():>8.1%}{ov:>8.2f}  "
                f"{' '.join(qpr)} | {' '.join(qwin)}{tag}",
                flush=True,
            )
        summary[board] = board_res

    # top_n 分档 (从每配置 top-20 截取, 展示 ref 与最优稳定性候选)
    print("\n===== top_n 分档 pr10d (每配置从 top-20 截取) =====", flush=True)
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        print(f"\n=== [{board}] ===", flush=True)
        for cfg in CONFIGS:
            name = cfg["name"]
            c = sub[sub["config"] == name]
            if c.empty:
                continue
            cut_s = " ".join(f"{k}:{c[c['n'] == k]['pr10d'].mean():+.4f}" for k in CUTS)
            print(f"  {name:<22}{cut_s}", flush=True)

    summary_meta = {
        "ts": ts,
        "eval_days": args.eval_days,
        "ref_cfg": ref_cfg,
        "board_thresholds": {"main": 0.003, "dual": 0.003},
        "note": "STABLE_WIN = pr10d>ref 且无负季度 且 ≥2/4 季度赢 ref; gain<0.3pp 判噪声",
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {"meta": summary_meta, "boards": summary}, f, ensure_ascii=False, indent=1
        )

    print(f"\nWORM: {out_dir}", flush=True)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
