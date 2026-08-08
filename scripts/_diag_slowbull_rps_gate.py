"""慢牛 rps_60 第二道门 策略模拟 + 子窗口稳定性 (2026-08-08).

因子梯度诊断 (diag_slowbull_factor_gradient_20260808_080116) 发现 rps_60 截面分位
是 gate 内唯一双板强单调的"守得住"预测因子 (main d10-d1 +4.08pp / dual +5.71pp,
spearman 0.73/0.65; 底部 10% 主板实得为负). 本脚本把该发现落实为可上生产的策略:

- 在 gate_slow_bull ∩ 上升段 (op_rule 开仓) 内, 逐板日对 gate 候选按 rps_60 截面分位
  做第二道门 (thresh), 剩余池 pool_score → Top-20 → trail8 实得.
- thresh ∈ {0.0(=ref, 无第二门), 0.3, 0.5, 0.7}: 报告 picks/日 (填充度), op_rule 实得,
  p_win, 中位持有.
- 子窗口稳定性: OOS 250d 切 4 季度, ref vs thresh=0.5 逐季对比 (AI扫参#1: 选稳定>最高).
- 交叉核对: thresh=0 的 op_rule 应吻合 paramsweep prod_ref (main +3.21% / dual +2.75%).

无前瞻 (截面分位用当日值, 入场 T+1), 净成本. WORM 落盘.
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
from app.pipeline_parallel.scoring import pool_score, select_topn
from config.settings import BACKTEST_RESULT_DIR

NEED = [
    "symbol",
    "date",
    "close_hfq",
    "high_hfq",
    "low_hfq",
    "open_hfq",
    "adv20",
    "volume",
    "turnover_rate",
    "volume_ratio",
    "margin_balance_chg_5d",
]

SELL_COLS = (
    "below_ma20",
    "adx_broken",
    "big_drop",
    "below_ma5_3d",
    "turnover_spike",
    "tp_80_div",
)

OOS_DAYS = 250
THRESHOLDS = (0.0, 0.3, 0.5, 0.7)
GATE_THRESH = 0.5  # 稳定性详检的门槛


def build_base_panel() -> pd.DataFrame:
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        df = pd.read_parquet(ckpt, columns=NEED)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work["board"] = work["symbol"].map(board_of)
    work, gate = tradability_gate(work)
    work = indicators.prepare_adx(work)
    signals.add_signal_columns(work)
    work["gate_slow_bull"] = screener.slow_bull_gate(work, ADX_SPEC)
    signals.add_market_regime(work, SLOW_BULL_REGIME)
    work["adx_score"] = work["adx"].clip(
        lower=0.0, upper=float(ADX_SPEC["adx_optimal_max"])
    )
    print(
        f"面板 rows={len(work):,} stocks={work['symbol'].nunique():,} dates={work['date'].nunique():,}",
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
    return {
        "sym_code": {s: int(c) for c, s in enumerate(uniques)},
        "starts": starts,
        "ends": ends,
        "dates": work["date"].values.astype("datetime64[ns]"),
        "close": work["close_hfq"].values,
        "low": work["low_hfq"].values,
        "ma20": work["ma20"].values,
        "any_sell": any_sell,
        "cost": COST + 2 * np.array([slippage_tier(v) for v in work["adv20"].values]),
        "regime_lut": dict(
            zip(work["date"].values, work["slow_bull_regime"].values, strict=False)
        ),
    }


def exit_rets(
    picks: pd.DataFrame, A: dict, trail_pct: float, hard_stop: float, max_hold: int
) -> np.ndarray:
    sym_code, starts, ends = A["sym_code"], A["starts"], A["ends"]
    dates_dt, close, cost_arr = A["dates"], A["close"], A["cost"]
    M, hard = int(max_hold), float(hard_stop)
    rets = np.full(len(picks), np.nan)
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
                break
        if exit_ret is None:
            exit_ret = close[r0 + M] / entry - 1 - cost
        rets[i] = exit_ret
    return rets


def _agg(rets: np.ndarray) -> dict:
    m = np.isfinite(rets)
    rr = rets[m]
    if not len(rr):
        return {"n": 0, "realized": None, "p_win": None}
    return {
        "n": int(len(rr)),
        "realized": round(float(rr.mean()), 5),
        "p_win": round(float((rr > 0).mean()), 4),
    }


def sim_board(
    work: pd.DataFrame, A: dict, board: str, oos_start, thresh: float
) -> dict:
    """板内逐日: gate ∩ rps60≥thresh ∩ 上升段 → Top-20 → trail8 实得 (op_rule)."""
    mask = (
        (work["board"] == board) & work["gate_slow_bull"] & (work["date"] >= oos_start)
    )
    wb = work[mask].copy()
    # gate 内日截面 rps_60 百分位
    wb["rk_rps60"] = wb.groupby("date")["rps_60"].rank(pct=True)
    dates = np.sort(wb["date"].unique())
    picks_rows = []
    for d in dates:
        if not A["regime_lut"].get(d, False):  # 下降段 no_open
            continue
        day = wb[wb["date"] == d]
        cand = day[day["rk_rps60"] >= thresh] if thresh > 0 else day
        if cand.empty:
            continue
        score = pool_score(cand, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
        top = select_topn(cand, score, SLOW_BULL.top_n)
        picks_rows.append(top[["symbol", "date"]])
    if not picks_rows:
        return {
            "n_picks": 0,
            "n_days_open": 0,
            "picks_per_open_day": 0.0,
            **{k: None for k in ("realized", "p_win", "n")},
        }
    pk = pd.concat(picks_rows, ignore_index=True)
    rets = exit_rets(
        pk,
        A,
        SLOW_BULL_REGIME["trail_pct"],
        SLOW_BULL_REGIME["hard_stop"],
        SLOW_BULL_REGIME["max_hold"],
    )
    agg = _agg(rets)
    return {
        "n_picks": int(len(pk)),
        "n_days_open": int(len(dates)),
        "picks_per_open_day": round(len(pk) / len(dates), 2),
        "realized": agg["realized"],
        "p_win": agg["p_win"],
        "n": agg["n"],
    }


def sim_quarter(work: pd.DataFrame, A: dict, board: str, q_start, q_end) -> dict:
    """季度子窗: ref vs thresh=0.5 的 op_rule (开仓日数少, 仅对比 realized)."""
    out = {}
    for th in (0.0, GATE_THRESH):
        mask = (
            (work["board"] == board)
            & work["gate_slow_bull"]
            & (work["date"] >= q_start)
            & (work["date"] <= q_end)
        )
        wb = work[mask].copy()
        if wb.empty:
            out[th] = {"n": 0, "realized": None}
            continue
        wb["rk_rps60"] = wb.groupby("date")["rps_60"].rank(pct=True)
        picks_rows = []
        for d in np.sort(wb["date"].unique()):
            if not A["regime_lut"].get(d, False):
                continue
            day = wb[wb["date"] == d]
            cand = day[day["rk_rps60"] >= th] if th > 0 else day
            if cand.empty:
                continue
            score = pool_score(cand, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
            picks_rows.append(
                select_topn(cand, score, SLOW_BULL.top_n)[["symbol", "date"]]
            )
        if not picks_rows:
            out[th] = {"n": 0, "realized": None}
            continue
        pk = pd.concat(picks_rows, ignore_index=True)
        rets = exit_rets(
            pk,
            A,
            SLOW_BULL_REGIME["trail_pct"],
            SLOW_BULL_REGIME["hard_stop"],
            SLOW_BULL_REGIME["max_hold"],
        )
        out[th] = _agg(rets)
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("构建面板...", flush=True)
    work = build_base_panel()
    dates = np.sort(work["date"].unique())
    oos_start = dates[-OOS_DAYS]
    A = build_arrays(work)
    print(f"OOS {OOS_DAYS}d 起 {oos_start} | 末日 {dates[-1]}", flush=True)

    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "objective": "rps_60 第二道门策略模拟 (gate∩rps60≥thresh∩上升段 → Top-20 → trail8 op_rule) + 子窗口稳定性",
        "oos_days": OOS_DAYS,
        "entry_exit": "close_hfq[T+1] 入场, trail8 收盘退出, 净成本",
        "crosscheck": "thresh=0 应吻合 paramsweep prod_ref (main +3.21% / dual +2.75%)",
        "boards": {},
    }
    for b in ("main", "dual"):
        bres = {"thresholds": {}, "quarters": {}}
        for th in THRESHOLDS:
            s = sim_board(work, A, b, oos_start, th)
            bres["thresholds"][str(th)] = s
            print(
                f"  [{b}] thresh={th}: picks/day={s.get('picks_per_open_day')} "
                f"realized={s.get('realized')} p_win={s.get('p_win')} n={s.get('n')}",
                flush=True,
            )
        # 4 季度稳定性
        q_bounds = pd.date_range(oos_start, dates[-1], periods=5)
        for qi in range(4):
            qs, qe = q_bounds[qi], (q_bounds[qi + 1] - pd.Timedelta(days=1))
            qd = sim_quarter(work, A, b, np.datetime64(qs), np.datetime64(qe))
            bres["quarters"][f"Q{qi + 1}"] = {str(k): v for k, v in qd.items()}
            r0 = qd[0.0].get("realized") if qd.get(0.0) else None
            r5 = qd.get(GATE_THRESH, {}).get("realized")
            print(f"  [{b}] Q{qi + 1}: ref={r0} thresh0.5={r5}", flush=True)
        out["boards"][b] = bres

    del work
    gc.collect()
    ts = out["ts"]
    run_dir = BACKTEST_RESULT_DIR / f"diag_slowbull_rps_gate_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / "rps_gate.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
