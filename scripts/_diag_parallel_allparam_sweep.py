"""并行系统全参数 + 排名旋钮扫描 — 无前瞻(realized labels) 扩展版 (2026-08-08).

背景: 生产 mag_10d 校准参数已定案 (config.MAG10D_CAL, 2026-08-07 决策):
  cal_n=21, per_stock_window=130, per_stock_min_n=50 (强制纯横截面 OLS),
  shrink_kappa=10, cross_min_n=50, score_col='score'=max(sniper,fusion),
  buy_lag=1, label_horizon=10.
前序无前瞻扫描 (_diag_10d_param_sweep_nl, WORM diag_10d_param_sweep_nl_20260807_200540)
已扫 cal_n{21,63,126,160}/psw{60,90,200}/minn{15,20,50}/kappa{10,20,80} 并定案生产值.

本脚本扩展: 围绕生产 ref 一因子扫描剩余/更细旋钮 + 少量联合确认:
  a. mag10d 细格: cal_n∈{14,28}, shrink_kappa∈{5,20}, per_stock_min_n∈{70}
  b. cross_min_n∈{30,80} (prod 50)
  c. top_n∈{5,15} (排名清单长度; ref=10, prod 狙击5/融合10)
  d. score 组合: ref=max(sniper,fusion); 备选 mean / sniper-only / fusion-only
  e. 池特征权重: 每池每可用特征 2× 等权 (其余等权) → 2×6=12 权重变体 + 等权 ref
  + 4 个联合确认 (joint_*)

设计: 全部 score 变体一次性预计算为独立列 (score_max/mean/sniper/fusion/
  score_w_{sniper,fusion}_<feat>), 各 config 引用其列 → 单次面板尾部加载 +
  单趟 walk-forward. 横截面前缀和按**日期桶**聚合 (模板按行前缀在日期边界
  截取等价, 但 16 列×2 板块内存从 ~1.4GB 降到 ~1MB).

无前瞻协议 (与模板一致, 铁律): 拟合窗 [D-cal_n, D-REALIZED_DROP) 只用已实现
  标签 (行 t 的卖点收盘 close_hfq[T+11] 严格早于决策日 D); 横截面按日期裁.
评估 = 每板块每 config 日截面 mag_10d 降序 head(top_n); 实得 = 买 close_hfq[T+1],
  卖 close_hfq[T+1+k], k∈{2,3,5,10}; 相对 ref 逐日头对头 pr5d/pr10d 赢天数 + 重合度.

注: 本扫描所有 minn≥50 且无前瞻下每股已实现样本 ~10-17 (cal_n≤28), 每股回归分支
  永不触发 → 实际=纯横截面 OLS (shrink_kappa/psw 结构性 no-op, 结果应与 ref 重合).

自包含: 禁止 import app.pipeline_parallel.backtest, helper 内联 (池分用 scoring.cross_rank).
用法: python scripts/_diag_parallel_allparam_sweep.py [eval_days=250]
WORM: BACKTEST_RESULT_DIR/diag_parallel_allparam_<ts>/
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
from app.pipeline_parallel.scoring import cross_rank, pool_score
from config.settings import BACKTEST_RESULT_DIR

# ---------------- 扫描设置 ----------------
REF = {
    "cal_n": 21,  # 校准窗 (交易日, 生产)
    "psw": 130,  # 每股自用最近交易日 (纯横截面下 no-op)
    "minn": 50,  # 每股回归最小样本, 不足回退横截面; =50 强制纯横截面
    "kappa": 10,  # 收缩强度 (纯横截面下 no-op)
    "cross_min_n": 50,  # 横截面有效样本下限, 不足该板块当日不出票
    "top_n": 10,  # 排名清单长度 (日截面 mag 降序 head; 生产狙击5/融合10)
    "score_col": "score_max",  # 校准输入 score = max(sniper, fusion)
}
H_K = (2, 3, 5, 10)
BOARDS = ("main", "dual")

# 无前瞻常量: label_pm_10d_net[t] = close[t+1+10]/close[t+1]-1 → 行 t 卖点收盘在第 11 个交易日.
REALIZED_DROP = 1 + 10  # buy_lag + label_horizon = 11

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
# prepare_adx → pool_score 缺列自动跳过 → 各 config 同口径, 头对头仍有效.
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


def _resolve_avail(pool) -> list[str]:
    return [c for c in pool if c in NEEDED_COLS]


# 池可用特征: SNIPER/FUSION 池各 7 特征, pv_corr_5 需 prepare_adx (本脚本不用) → 6 实际可用
SNIPER_AVAIL = _resolve_avail(SNIPER.pool)
FUSION_AVAIL = _resolve_avail(FUSION.pool)
assert SNIPER_AVAIL and FUSION_AVAIL, "池特征在面板中无可用列"

# score 变体列 (一次性预计算)
SCORE_COLS = (
    ["score_max", "score_mean", "score_sniper", "score_fusion"]
    + [f"score_w_sniper_{f}" for f in SNIPER_AVAIL]
    + [f"score_w_fusion_{f}" for f in FUSION_AVAIL]
)
SCORE_COL_IDX = {c: i for i, c in enumerate(SCORE_COLS)}


def build_configs() -> list[dict]:
    """参考配置 + 各旋钮一因子扰动 + 联合确认. 每个 config 含全部校准字段."""
    cfg: list[dict] = []
    cfg.append({"name": "ref", "param": "ref", "param_value": 0, **REF})

    # a. mag10d 细格
    for v in (14, 28):
        d = dict(REF, cal_n=v)
        cfg.append({"name": f"cal_n_{v}", "param": "cal_n", "param_value": v, **d})
    for v in (5, 20):
        d = dict(REF, kappa=v)
        cfg.append({"name": f"kappa_{v}", "param": "kappa", "param_value": v, **d})
    for v in (70,):
        d = dict(REF, minn=v)
        cfg.append({"name": f"minn_{v}", "param": "minn", "param_value": v, **d})

    # b. cross_min_n
    for v in (30, 80):
        d = dict(REF, cross_min_n=v)
        cfg.append(
            {"name": f"cross_{v}", "param": "cross_min_n", "param_value": v, **d}
        )

    # c. top_n
    for v in (5, 15):
        d = dict(REF, top_n=v)
        cfg.append({"name": f"topn_{v}", "param": "top_n", "param_value": v, **d})

    # d. score 组合
    for name, col in (
        ("mean", "score_mean"),
        ("sniper", "score_sniper"),
        ("fusion", "score_fusion"),
    ):
        d = dict(REF, score_col=col)
        cfg.append(
            {"name": f"score_{name}", "param": "score", "param_value": name, **d}
        )

    # e. 池特征权重 (每池每可用特征 2× 等权)
    for f in SNIPER_AVAIL:
        d = dict(REF, score_col=f"score_w_sniper_{f}")
        cfg.append({"name": f"wsniper_{f}", "param": "w_sniper", "param_value": f, **d})
    for f in FUSION_AVAIL:
        d = dict(REF, score_col=f"score_w_fusion_{f}")
        cfg.append({"name": f"wfusion_{f}", "param": "w_fusion", "param_value": f, **d})

    # 联合确认 (a-few 配对)
    for name, kw in (
        ("joint_cal14_cross30", dict(cal_n=14, cross_min_n=30)),
        ("joint_cal28_cross30", dict(cal_n=28, cross_min_n=30)),
        ("joint_cal14_mean", dict(cal_n=14, score_col="score_mean")),
        ("joint_mean_cross30", dict(score_col="score_mean", cross_min_n=30)),
    ):
        d = dict(REF, **kw)
        cfg.append({"name": name, "param": "joint", "param_value": name, **d})
    return cfg


CONFIGS = build_configs()
MAX_CAL_N = max(c["cal_n"] for c in CONFIGS)

PARAM_DISPLAY = {
    "cal_n": "cal_n(校准窗)",
    "kappa": "shrink_kappa",
    "minn": "per_stock_min_n",
    "cross_min_n": "cross_min_n(横截面样本下限)",
    "top_n": "top_n(排名清单长度)",
    "score": "score组合",
    "w_sniper": "sniper池权重(特征×2)",
    "w_fusion": "fusion池权重(特征×2)",
    "joint": "联合确认",
}
PARAM_CURRENT = {
    "cal_n": 21,
    "kappa": 10,
    "minn": 50,
    "cross_min_n": 50,
    "top_n": 10,
    "score": "max",
    "w_sniper": "等权",
    "w_fusion": "等权",
}
PARAM_ORDER = [
    "cal_n",
    "kappa",
    "minn",
    "cross_min_n",
    "top_n",
    "score",
    "w_sniper",
    "w_fusion",
    "joint",
]


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


# ---------------- 内联 helper (从模板抄) ----------------


def tradability_gate(
    work: pd.DataFrame, lookback: int = 20, min_presence: float = 0.8
) -> tuple[pd.DataFrame, dict]:
    """PIT 可交易性门: 前 lookback 日有行比例 < min_presence → 剔除."""
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


def add_score_variants(work: pd.DataFrame) -> None:
    """一次性预计算全部 score 变体列 (等权/加权 池分 + 组合)."""
    for spec, avail in ((SNIPER, SNIPER_AVAIL), (FUSION, FUSION_AVAIL)):
        eq = np.full(len(work), np.nan)
        boost = {f: np.full(len(work), np.nan) for f in avail}
        for board in BOARDS:
            bm = work["board"].values == board
            idx = np.nonzero(bm)[0]
            sub = work.iloc[idx]
            sum_all = None
            ranks = {}
            for c in avail:
                r = cross_rank(sub, c).to_numpy(float)
                ranks[c] = r
                sum_all = (
                    r if sum_all is None else sum_all + r
                )  # 加链传播 NaN (与 pool_score 一致)
            eq[idx] = sum_all / len(avail)
            for f in avail:
                # pool_score 权重 {f:2.0} → 归一化 w_f=2/7, 其余 1/7 → (Σall + rank_f)/7
                boost[f][idx] = (sum_all + ranks[f]) / (len(avail) + 1)
        work[f"pool_{spec.name}"] = eq
        for f in avail:
            work[f"pool_{spec.name}_w_{f}"] = boost[f]

    # sanity: 等权算术 == pool_score(等权) (仅主板块抽查, 防归一化/NaN 语义漂移)
    _bm = work["board"].values == "main"
    _sub = work.iloc[np.nonzero(_bm)[0]]
    _sp = pool_score(_sub, SNIPER.pool).to_numpy(float)
    if not np.allclose(
        work.loc[_bm, "pool_sniper"].to_numpy(float), _sp, equal_nan=True
    ):
        raise RuntimeError("sniper 等权分数与 pool_score 不一致")
    del _sp
    gc.collect()

    work["score_sniper"] = work["pool_sniper"]
    work["score_fusion"] = work["pool_fusion"]
    # max/mean 用 np.fmax / 手动均值 (skipna 语义, 与模板 max(axis=1) 一致)
    work["score_max"] = np.fmax(
        work["score_sniper"].to_numpy(float), work["score_fusion"].to_numpy(float)
    )
    _a = work["score_sniper"].to_numpy(float)
    _b = work["score_fusion"].to_numpy(float)
    _both = np.isfinite(_a) & np.isfinite(_b)
    _ona = np.isfinite(_a) & ~np.isfinite(_b)
    _onb = ~np.isfinite(_a) & np.isfinite(_b)
    work["score_mean"] = np.where(
        _both, (_a + _b) / 2.0, np.where(_ona, _a, np.where(_onb, _b, np.nan))
    )
    for f in SNIPER_AVAIL:
        work[f"score_w_sniper_{f}"] = np.fmax(
            work[f"pool_sniper_w_{f}"].to_numpy(float),
            work["score_fusion"].to_numpy(float),
        )
    for f in FUSION_AVAIL:
        work[f"score_w_fusion_{f}"] = np.fmax(
            work["score_sniper"].to_numpy(float),
            work[f"pool_fusion_w_{f}"].to_numpy(float),
        )
    drop = [c for c in work.columns if c.startswith("pool_")]
    work.drop(columns=drop, inplace=True)
    gc.collect()


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


def build_cross(work: pd.DataFrame) -> dict:
    """每 (板块, score_col) 按日期桶聚合 → 前缀和 (窗口按日期边界截取, 与模板行级等价)."""
    cross: dict = {}
    for board in BOARDS:
        bm = work["board"].values == board
        sub = work.iloc[np.nonzero(bm)[0]]
        d64 = sub["date"].to_numpy().astype("datetime64[ns]")
        by: dict = {}
        for sc in SCORE_COLS:
            x = sub[sc].to_numpy(float)
            y = sub["label_pm_10d_net"].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y)
            d = d64[valid]
            if len(d) == 0:
                by[sc] = None
                continue
            bd, bin_idx = np.unique(d, return_inverse=True)
            nb = len(bd)
            n_bin = np.bincount(bin_idx, minlength=nb).astype(float)
            sx = np.bincount(bin_idx, weights=x[valid], minlength=nb)
            sy = np.bincount(bin_idx, weights=y[valid], minlength=nb)
            sxx = np.bincount(bin_idx, weights=x[valid] ** 2, minlength=nb)
            sxy = np.bincount(bin_idx, weights=x[valid] * y[valid], minlength=nb)
            by[sc] = (
                bd,
                np.concatenate([[0.0], np.cumsum(n_bin)]),
                np.concatenate([[0.0], np.cumsum(sx)]),
                np.concatenate([[0.0], np.cumsum(sy)]),
                np.concatenate([[0.0], np.cumsum(sxx)]),
                np.concatenate([[0.0], np.cumsum(sxy)]),
            )
        cross[board] = by
        del sub
        gc.collect()
    return cross


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_days", nargs="?", type=int, default=250)
    args = ap.parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_parallel_allparam_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tail_days = MAX_CAL_N + REALIZED_DROP + args.eval_days + 40
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

    print(
        f"[score] 预计算 {len(SCORE_COLS)} 个 score 变体列 (等权/加权池分+组合)...",
        flush=True,
    )
    add_score_variants(work)
    print(f"[score] done ({time.time() - t0:.0f}s)", flush=True)

    print("[cross] 每 (板块,score_col) 日期桶前缀和...", flush=True)
    cross = build_cross(work)
    print(f"[cross] done ({time.time() - t0:.0f}s)", flush=True)

    # 每股预提取: 只存 (gd, gs_ref, gy, pos). 本扫描全部 minn≥50 且无前瞻下每股已实现
    # 样本 ≤ ~17 (cal_n≤28) → 每股回归分支永不触发, gs_ref 不会被读取 (纯横截面).
    ref_col = REF["score_col"]
    print(
        f"[stock] 每股预提取 (score_col={ref_col}, label_pm_10d_net, valid_pos)...",
        flush=True,
    )
    stock = work[["symbol", "date", ref_col, "label_pm_10d_net"]]
    sd64 = stock["date"].to_numpy().astype("datetime64[ns]")
    sym_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, g in stock.groupby("symbol", sort=False):
        gd = sd64[g.index.to_numpy()]
        gs = g[ref_col].to_numpy(float)
        gy = g["label_pm_10d_net"].to_numpy(float)
        valid = np.isfinite(gs) & np.isfinite(gy)
        pos = np.nonzero(valid)[0].astype(np.int64)
        sym_data[str(sym)] = (gd, gs, gy, pos)
    del stock
    gc.collect()
    print(
        f"[stock] {len(sym_data):,} 只预提取完成 ({time.time() - t0:.0f}s)", flush=True
    )

    print("[day] 逐日截面预提取 (syms + score 矩阵)...", flush=True)
    day_info: dict = {}
    for D in eval_dates:
        day = work[work["date"] == D]
        db = {}
        for board in BOARDS:
            sub = day[day["board"] == board]
            db[board] = (
                sub["symbol"].astype(str).to_numpy(),
                sub[SCORE_COLS].to_numpy(float),
            )
        day_info[D] = db
    del work, day
    gc.collect()
    print(
        f"[day] 逐日截面预提取 {len(day_info)} 日 ({time.time() - t0:.0f}s)", flush=True
    )

    def cross_slope_int(board: str, score_col: str, cal_lo, cutoff_date):
        """横截面窗 = [cal_lo, cutoff_date] 只用已实现标签 (行 t 卖价 ≤ cutoff_date)."""
        ent = cross[board].get(score_col)
        if ent is None:
            return None
        bd, pn, psx, psy, psxx, psxy = ent
        iLo = int(np.searchsorted(bd, np.datetime64(cal_lo), side="left"))
        iEnd = int(np.searchsorted(bd, np.datetime64(cutoff_date), side="right"))
        n = pn[iEnd] - pn[iLo]
        if n < 1:
            return None
        Sx = psx[iEnd] - psx[iLo]
        Sy = psy[iEnd] - psy[iLo]
        Sxx = psxx[iEnd] - psxx[iLo]
        Sxy = psxy[iEnd] - psxy[iLo]
        var = n * Sxx - Sx * Sx
        if var <= 1e-12:
            return (0.0, Sy / n, n)
        slope = (n * Sxy - Sx * Sy) / var
        return (slope, (Sy - slope * Sx) / n, n)

    print(
        f"[calib] walk-forward 逐日重拟合 × {len(CONFIGS)} configs (无前瞻窗)...",
        flush=True,
    )
    rows: list[dict] = []
    cs_cache: dict = {}
    mag_cache: dict = {}
    for di_abs, D in enumerate(eval_dates):
        di = date_idx[D]
        D64 = np.datetime64(D)
        for board in BOARDS:
            syms, smat = day_info[D][board]
            if len(syms) == 0:
                continue
            tops: dict = {}
            for cfg in CONFIGS:
                if di < max(cfg["cal_n"], REALIZED_DROP):
                    tops[cfg["name"]] = None
                    continue
                cal_lo = all_dates[di - cfg["cal_n"]]
                cutoff_date = all_dates[di - REALIZED_DROP]
                ckey = (board, cfg["score_col"], cal_lo, cutoff_date)
                if ckey not in cs_cache:
                    cs_cache[ckey] = cross_slope_int(
                        board, cfg["score_col"], cal_lo, cutoff_date
                    )
                cs_ent = cs_cache[ckey]
                if cs_ent is None:
                    tops[cfg["name"]] = None
                    continue
                cs, ci, ncross = cs_ent
                if ncross < cfg["cross_min_n"]:
                    tops[cfg["name"]] = None
                    continue
                mkey = (D, board, cfg["score_col"], cfg["cal_n"])
                if mkey not in mag_cache:
                    sc = smat[:, SCORE_COL_IDX[cfg["score_col"]]]
                    mag_arr = np.full(len(syms), np.nan)
                    cal_lo64 = np.datetime64(cal_lo)
                    for i in range(len(syms)):
                        scv = sc[i]
                        if not np.isfinite(scv):
                            continue
                        sd = sym_data.get(str(syms[i]))
                        if sd is None:
                            continue
                        gd, gs, gy, pos = sd
                        iLo = int(np.searchsorted(gd, cal_lo64, side="left"))
                        # 无前瞻: 该股行 t 可用 ⟺ t+11 交易日严格早于 D ⟺ 索引 < N_lt(D) - 11
                        iEnd = (
                            int(np.searchsorted(gd, D64, side="left")) - REALIZED_DROP
                        )
                        vLo = int(np.searchsorted(pos, iLo, side="left"))
                        vEnd = int(np.searchsorted(pos, iEnd, side="left"))
                        vc = vEnd - vLo
                        if vc >= cfg["minn"]:  # 本扫描永不触发 (minn≥50 > ~17)
                            take = min(vc, cfg["psw"])
                            r = pos[vEnd - take : vEnd]
                            x = gs[r]
                            y = gy[r]
                            raw_slope, _ = _ols_slope_intercept(x, y)
                            lam = take / (take + cfg["kappa"])
                            slope = lam * raw_slope + (1 - lam) * cs
                            intercept = float(y.mean()) - slope * float(x.mean())
                        else:
                            slope, intercept = cs, ci
                        mag_arr[i] = slope * scv + intercept
                    mag_cache[mkey] = mag_arr
                mag_arr = mag_cache[mkey]
                rank_df = pd.DataFrame(
                    {
                        "symbol": syms,
                        "score": smat[:, SCORE_COL_IDX[cfg["score_col"]]],
                        "mag": mag_arr,
                    }
                ).dropna(subset=["mag"])
                if rank_df.empty:
                    tops[cfg["name"]] = None
                    continue
                pick = rank_df.sort_values("mag", ascending=False).head(cfg["top_n"])
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
                        "cross_min_n": cfg["cross_min_n"],
                        "top_n": cfg["top_n"],
                        "score_col": cfg["score_col"],
                        "n": int(len(pick)),
                        "pr2d": _safe_nanmean(pr[:, 0]),
                        "pr3d": _safe_nanmean(pr[:, 1]),
                        "pr5d": _safe_nanmean(pr[:, 2]),
                        "pr10d": _safe_nanmean(pr[:, 3]),
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
                        "overlap": overlap,
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

    # ---------------- 汇总 ----------------
    print(
        f"\n===== 并行全参数/旋钮扫描 [无前瞻 realized labels]: TOP-N c2c 实得 "
        f"(最近 {args.eval_days} 交易日) =====",
        flush=True,
    )
    print(
        "评估 = close-to-close 点对点 (买=close[T+1], 卖=close[T+1+k]); "
        "校准 = 每股收缩回归, target=label_pm_10d_net",
        flush=True,
    )
    print(
        f"拟合窗 = [D-cal_n, D-{REALIZED_DROP}) 只用已实现标签 (无前瞻); 每 config 独立 cross_min_n",
        flush=True,
    )
    print(
        f"参考配置 ref = (cal_n={REF['cal_n']}, psw={REF['psw']}, minn={REF['minn']}, "
        f"kappa={REF['kappa']}, cross_min_n={REF['cross_min_n']}, top_n={REF['top_n']}, "
        f"score={REF['score_col']}); 头对头 vs ref",
        flush=True,
    )
    print(
        "注: 全部 minn≥50 且无前瞻下每股已实现样本 ≤ ~17 → 每股回归分支永不触发, "
        "shrink_kappa/psw 结构性 no-op (结果应与 ref 重合, 属预期)",
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
                    (
                        a.loc[common, "pr5d"].values[m5v]
                        > b.loc[common, "pr5d"].values[m5v]
                    ).sum()
                )
            if m10:
                w10 = int(
                    (
                        a.loc[common, "pr10d"].values[m10v]
                        > b.loc[common, "pr10d"].values[m10v]
                    ).sum()
                )
        return w5, m5, w10, m10

    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        b = sub[sub["config"] == "ref"].set_index("date")
        summary_boards[board] = {}
        print(f"\n[{board}]  (ref = 生产默认)", flush=True)
        for p in PARAM_ORDER:
            pname = PARAM_DISPLAY[p]
            cur = PARAM_CURRENT.get(p)
            cur_str = f" (生产 {cur})" if cur is not None else ""
            vals = [(c["name"], c["param_value"]) for c in CONFIGS if c["param"] == p]
            print(f"\n  ── {pname}{cur_str} ──", flush=True)
            print(
                f"  {'值':>22}{'日':>4}{'2d':>9}{'3d':>9}{'5d':>9}{'10d':>9}"
                f"{'5d涨率':>9}{'10d涨率':>9}{'重合':>7}",
                flush=True,
            )
            best5, best10 = None, None
            best5v, best10v = -1e9, -1e9
            for name, pv in [("ref", 0)] + vals:
                g = sub[sub["config"] == name]
                if g.empty:
                    continue
                r = g.mean(numeric_only=True)
                n = int(len(g))
                w5, m5, w10, m10 = h2h_wins(g, b)
                ov = (
                    float(g["overlap"].mean())
                    if not g["overlap"].isna().all()
                    else float("nan")
                )
                tag = f"{str(pv):>22}" if name != "ref" else "  ref    "
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
                    "param_value": str(pv),
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
                    f"      → {pname}: 5d实得最优={best5} ({best5v:+.4f}, ref "
                    f"{summary_boards[board][p]['ref_pr5d']:+.4f}), "
                    f"10d实得最优={best10} ({best10v:+.4f}, ref "
                    f"{summary_boards[board][p]['ref_pr10d']:+.4f})",
                    flush=True,
                )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "eval_days": args.eval_days,
        "metric": "close-to-close point returns (买=close[T+1], 卖=close[T+1+k]); NOT MFE",
        "calibration": "每股收缩回归 (target=label_pm_10d_net, 回退横截面; 本扫描全部纯横截面)",
        "no_lookahead": (
            f"拟合窗 [D-cal_n, D-{REALIZED_DROP}) 只用已实现标签 (行 t 的卖点收盘严格早于 D); "
            f"横截面按日期桶截取"
        ),
        "reference_config": REF,
        "configs": [{k: v for k, v in c.items() if k != "param"} for c in CONFIGS],
        "score_cols": SCORE_COLS,
        "sweep_summary": summary_boards,
        "agg": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """单变量最小二乘 (numpy, 快速路径)."""
    xm = float(x.mean())
    ym = float(y.mean())
    var = float(((x - xm) ** 2).sum())
    if var <= 1e-12:
        return 0.0, ym
    slope = float(((x - xm) * (y - ym)).sum() / var)
    return slope, ym - slope * xm


if __name__ == "__main__":
    main()
