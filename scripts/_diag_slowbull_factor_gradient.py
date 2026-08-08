"""慢牛"守得住涨幅"因子梯度诊断 (2026-08-08).

问题: 慢牛退出已定案 (上升段 trail8 + 下降段 no_open), 门槛/退出参数扫描平坦
(diag_slowbull_paramsweep_20260808_042151) → 下一级杠杆 = 选股. 但记忆定论
"合成 score 无法预测前瞻幅度" (MFE 五分位扁平). 本诊断在**实得 op_rule 口径**
下重测: gate 内的单因子截面分位是否预测 trail8 实得收益. 若有单调梯度 → 该因子
可作第二道门 (保留高分位) 或调权; 若全平坦 → 选股旋钮死, 慢牛到最优.

方法 (无前瞻, 忠实生产):
- 窄列集重建面板 (免宽表 OOM), prepare_adx + add_signal_columns + gate 同生产.
- 对每 board × OOS 日: gate 内全体候选, 逐因子在当日截面内取百分位 (rank pct).
- 逐票模拟 trail8 实得退出 (T+1 收盘入场, 收盘判定, 净成本) — 复用 paramsweep
  exit_rets. op_rule = 仅上升段票 (生产才开仓); all_trail = 不分 regime 参考.
- 聚合: 每个因子按截面百分位切 10 档 (跨日合并), 每档 mean realized / n /
  p_win; 报告 d10-d1 梯度 + 分位 Spearman + 前后半对比.

验收只看 OOS (末 250 交易日). WORM: BACKTEST_RESULT_DIR/diag_slowbull_factor_gradient_<ts>/
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline_parallel import indicators, screener, signals
from app.pipeline_parallel.backtest import COST, slippage_tier, tradability_gate
from app.pipeline_parallel.config import (
    ADX_SPEC,
    PANEL,
    SLOW_BULL,
    SLOW_BULL_REGIME,
    board_of,
)
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

# ── 候选因子 ──
# 打分池 7 因子 (prepare_adx 产出) + 追高/守得住代理 (面板列)
POOL_FACTORS = (
    "adx_score", "ma_tightness", "sharpe_20", "rps_60",
    "pv_corr_5", "margin_balance_chg_5d", "pct_70_con",
)
EXTRA_FACTORS = (
    "close_position", "ret_reversal_5d", "up_streak", "bias_5d", "pct_90_con",
)
FACTORS = POOL_FACTORS + EXTRA_FACTORS

NEED = [
    "symbol", "date", "close_hfq", "high_hfq", "low_hfq", "open_hfq",
    "adv20", "volume", "turnover_rate", "volume_ratio", "margin_balance_chg_5d",
] + list(EXTRA_FACTORS)

SELL_COLS = (
    "below_ma20", "adx_broken", "big_drop", "below_ma5_3d", "turnover_spike", "tp_80_div",
)

OOS_DAYS = 250


def build_base_panel() -> pd.DataFrame:
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        df = pd.read_parquet(ckpt, columns=NEED)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(["symbol", "date"], ignore_index=True)
    del slices
    gc.collect()
    work["board"] = work["symbol"].map(board_of)
    work, gate = tradability_gate(work)
    work = indicators.prepare_adx(work)
    signals.add_signal_columns(work)
    work["gate_slow_bull"] = screener.slow_bull_gate(work, ADX_SPEC)
    signals.add_market_regime(work, SLOW_BULL_REGIME)
    work["adx_score"] = work["adx"].clip(lower=0.0, upper=float(ADX_SPEC["adx_optimal_max"]))
    print(
        f"基础面板 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"dates={work['date'].nunique():,} | 可交易门剔除 {gate['removed_stocks']} 只",
        flush=True,
    )
    return work


def build_arrays(work: pd.DataFrame) -> dict:
    uniques, codes = np.unique(work["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    any_sell = np.zeros(len(work), dtype=bool)
    for c in SELL_COLS:
        any_sell |= work[c].values.astype(bool)
    A = {
        "sym_code": {s: int(c) for c, s in enumerate(uniques)},
        "starts": starts,
        "ends": ends,
        "dates": work["date"].values.astype("datetime64[ns]"),
        "close": work["close_hfq"].values,
        "low": work["low_hfq"].values,
        "ma20": work["ma20"].values,
        "any_sell": any_sell,
        "cost": COST + 2 * np.array([slippage_tier(v) for v in work["adv20"].values]),
        "regime_lut": dict(zip(work["date"].values, work["slow_bull_regime"].values, strict=False)),
    }
    return A


def exit_rets(picks: pd.DataFrame, A: dict, trail_pct: float, hard_stop: float, max_hold: int) -> tuple[np.ndarray, np.ndarray]:
    """trail8 退出 (收盘自峰值回落 trail_pct 走 + 硬止损 hard_stop), 返回 (净收益, 持有天数)."""
    sym_code, starts, ends = A["sym_code"], A["starts"], A["ends"]
    dates_dt, close, _any_sell, cost_arr = A["dates"], A["close"], A["any_sell"], A["cost"]
    M = int(max_hold)
    hard = float(hard_stop)
    rets = np.full(len(picks), np.nan)
    holds = np.full(len(picks), np.nan)
    for i, (sym, T) in enumerate(zip(picks["symbol"], picks["date"], strict=False)):
        c = sym_code[str(sym)]
        lo, hi = starts[c], ends[c]
        base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
        r0 = base + 1
        if r0 + M >= hi:
            continue
        entry = close[r0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        cost = cost_arr[base]
        peak = 0.0
        exit_ret = None
        for k in range(1, M + 1):
            r = base + k
            ret = close[r] / entry - 1.0
            peak = max(peak, ret)
            if ret <= peak - trail_pct or ret <= hard - 1.0:
                exit_ret = ret - cost
                holds[i] = k
                break
        if exit_ret is None:
            exit_ret = close[r0 + M] / entry - 1 - cost
            holds[i] = M
        rets[i] = exit_ret
    return rets, holds


def _agg(rets: np.ndarray, holds: np.ndarray, dates: np.ndarray) -> dict:
    m = np.isfinite(rets)
    rr = rets[m]
    if not len(rr):
        return {"n": 0, "realized": None, "p_win": None, "median_hold": None}
    hh = holds[m]
    return {
        "n": int(len(rr)),
        "realized": round(float(rr.mean()), 5),
        "p_win": round(float((rr > 0).mean()), 4),
        "median_hold": round(float(np.median(hh)), 1),
    }


def percentile_rank(work: pd.DataFrame, cols) -> pd.DataFrame:
    """板内日截面百分位 (0-1, 高分=因子高). 返回按原行序的 rank 列."""
    r = pd.DataFrame(index=work.index)
    for c in cols:
        r[f"rk_{c}"] = work.groupby("date")[c].rank(pct=True)
    return r


def gradient_table(ranks: np.ndarray, rets: np.ndarray, holds: np.ndarray, dates: np.ndarray) -> dict:
    """按截面百分位切 10 档, 每档聚合; 返回档表 + d10-d1 + Spearman."""
    m = np.isfinite(rets) & np.isfinite(ranks)
    if m.sum() < 20:
        return {"n": 0, "note": "样本不足"}
    r = ranks[m]
    q = pd.qcut(r, 10, labels=False, duplicates="drop")
    out = {"n": int(m.sum())}
    dec_rows = []
    d10s = []
    for di in range(int(q.max()) + 1):
        sel = np.asarray(q) == di
        agg = _agg(rets[m][sel], holds[m][sel], dates[m][sel])
        dec_rows.append({"decile": int(di), "lo": round(float(r[sel].min()), 3),
                         "hi": round(float(r[sel].max()), 3), **agg})
        if agg["realized"] is not None:
            d10s.append((di, agg["realized"]))
    out["deciles"] = dec_rows
    if len(d10s) >= 2:
        dvals = sorted(d10s)
        out["d10_minus_d1"] = round(float(dvals[-1][1] - dvals[0][1]), 5)
        arr = np.array([v for _, v in dvals])
        # 单调性: 相邻差同号比例 + Spearman (档序 vs 均值)
        diffs = np.diff(arr)
        out["mono_frac"] = round(float((diffs > 0).mean()), 3)
        if len(arr) >= 3:
            sp = pd.Series(arr).rank().corr(pd.Series(range(len(arr))).rank(), method="spearman")
            out["spearman"] = round(float(sp), 3) if not np.isnan(sp) else None
        hi_idx = arr >= arr.mean()
        out["top_half_minus_bottom"] = round(float(arr[hi_idx].mean() - arr[~hi_idx].mean()), 5) if hi_idx.any() and (~hi_idx).any() else None
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("构建窄面板...", flush=True)
    work = build_base_panel()
    dates = np.sort(work["date"].unique())
    oos_start = dates[-OOS_DAYS]
    print(f"全窗 {dates[0]} → {dates[-1]} ({len(dates)}d) | OOS {OOS_DAYS}d 起 {oos_start}", flush=True)
    A = build_arrays(work)

    # 合成 pool_score 的截面百分位 (同样板内日截面)
    score = pool_score(work, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
    work["_composite"] = score.fillna(-1e9)
    FACTOR_LIST = ["_composite"] + list(FACTORS)
    rk = percentile_rank(work, FACTOR_LIST)
    # 合并 extra 因子缺失检查
    for c in FACTORS:
        if c not in work.columns:
            print(f"  !! 因子列缺失: {c}", flush=True)

    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "objective": "gate 内单因子截面百分位 → trail8 实得收益梯度 (守得住涨幅第二道门筛查)",
        "oos_days": OOS_DAYS,
        "window": {"start": str(pd.Timestamp(oos_start).date()), "end": str(pd.Timestamp(dates[-1]).date()), "n_days": OOS_DAYS},
        "entry_exit": "close_hfq[T+1] 入场, trail8 收盘判定退出, 成本=COST+2×滑点",
        "boards": {},
    }

    for b in ("main", "dual"):
        mask = (work["board"] == b) & work["gate_slow_bull"] & (work["date"] >= oos_start)
        wb = work[mask].copy()
        rk_b = rk[mask]
        print(f"\n=== board={b}: gated OOS picks={len(wb):,} ===", flush=True)
        pk = wb[["symbol", "date"]]
        tr, th = exit_rets(pk, A, SLOW_BULL_REGIME["trail_pct"], SLOW_BULL_REGIME["hard_stop"], SLOW_BULL_REGIME["max_hold"])
        up = pk["date"].map(A["regime_lut"]).fillna(False).values
        pdate = pk["date"].values
        # op_rule = 仅上升段 (生产开仓); all = 不分 regime
        bres = {"n_gated": int(len(pk)), "factors": {}}
        agg_all = _agg(tr, th, pdate)
        agg_op = _agg(tr[up], th[up], pdate[up])
        bres["all_trail"] = agg_all
        bres["op_rule"] = agg_op
        bres["n_up"] = int(up.sum())
        print(f"  all_trail: {agg_all['realized']} n={agg_all['n']} | op_rule(up): {agg_op['realized']} n={agg_op['n']}", flush=True)
        for c in FACTOR_LIST:
            rk_col = rk_b[f"rk_{c}"].values
            g = gradient_table(rk_col, tr, th, pdate)
            g_up = gradient_table(rk_col[up], tr[up], th[up], pdate[up])
            bres["factors"][c] = {"all": g, "op_rule": g_up}
            f"d10-d1={g_up.get('d10_minus_d1')}" if g_up.get("d10_minus_d1") is not None else "n/a"
            print(f"    {c:<24} op_rule d10-d1={g_up.get('d10_minus_d1')} "
                  f"sp={g_up.get('spearman')} mono={g_up.get('mono_frac')} n={g_up.get('n')} | all d10-d1={g.get('d10_minus_d1')}", flush=True)
        out["boards"][b] = bres

    del work, rk
    gc.collect()

    ts = out["ts"]
    run_dir = BACKTEST_RESULT_DIR / f"diag_slowbull_factor_gradient_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / "gradient.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
