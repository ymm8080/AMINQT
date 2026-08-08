"""慢牛排名键 A/B (2026-08-08): 合成 score 排名 vs rps_60 排名.

背景: 主系统排名键已定 mag_10d; 慢牛仍用 ADX 合成 score (7 加权因子). 梯度诊断
(diag_slowbull_factor_gradient_20260808_080116) 显示合成 score 在门内截面分位→
trail8 实得是平的 (main −0.06pp), 而 rps_60 双板最强单调 (main +4.08pp / dual
+5.71pp) → 合成 score 排名键可能是慢牛最弱一环. 本脚本做受控 A/B.

方法 (无前瞻, 忠实生产):
- gate ∩ 上升段 (op_rule 开仓) 内, 逐板 × Top-N 对比:
    score 排名 (生产现状)  vs  rps_60 排名  → trail8 实得 (净成本, OOS 250d).
- prod 门照配置: SLOW_BULL_RPS_GATE 第二道门仅 dual (floor=0.5).
- Top-N ∈ {10, 5, 3} — 慢牛现状 top_n=20 全收 (门内池 ~6-13/日 < 20), 排名不改变
  名单; 收紧档位排名才可能生效. 报告每 cell picks/日 (填充度) + diff_days (两种
  排名选出集合不同的天数) — diff_days=0 直接证明该档位下排名是 no-op.
- 交互检查: rps60_nogate (不设第二道门, 纯 rps 排名) → 若 ≥ 生产(门+score), 可简化.
- 子窗口稳定性: top_n=10 处 score vs rps60 切 4 季度 (AI扫参#1: 选稳定>最高).
- 交叉核对: anchor (score, top-20, 无门) 应吻合 paramsweep prod_ref main +3.21% /
  dual +2.75%.

验收只看 OOS (末 250 交易日). WORM: BACKTEST_RESULT_DIR/diag_slowbull_rank_ab_<ts>/
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
    SLOW_BULL_RPS_GATE,
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
TOP_N_LIST = (10, 5, 3)
RPS_FLOOR = float(SLOW_BULL_RPS_GATE.get("floor", 0.5))
GATE_BOARDS = frozenset(
    b for b, on in SLOW_BULL_RPS_GATE.get("boards", {}).items() if on
)


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
        f"面板 rows={len(work):,} stocks={work['symbol'].nunique():,} "
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
        "regime_lut": dict(
            zip(work["date"].values, work["slow_bull_regime"].values, strict=False)
        ),
    }
    return A


def exit_rets(
    picks: pd.DataFrame, A: dict, trail_pct: float, hard_stop: float, max_hold: int
) -> tuple[np.ndarray, np.ndarray]:
    """trail8 退出 (收盘自峰值回落 trail_pct 走 + 硬止损 hard_stop), 返回 (净收益, 持有天数)."""
    sym_code, starts, ends = A["sym_code"], A["starts"], A["ends"]
    dates_dt, close, cost_arr = A["dates"], A["close"], A["cost"]
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


def _agg(rets: np.ndarray, holds: np.ndarray) -> dict:
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


def sim_variants(
    work: pd.DataFrame, A: dict, board: str, d0, d1, top_n: int, use_gate: bool
) -> dict | None:
    """gate∩[d0,d1]∩上升段 → Top-N: score 排名 vs rps_60 排名 (同门配置).

    返回两变体的聚合 + diff_days (两种排名选出集合不同的天数) + 填充度.
    diff_days=0 → 该档位下排名不改变名单 (no-op).
    """
    mask = (
        (work["board"] == board)
        & work["gate_slow_bull"]
        & (work["date"] >= d0)
        & (work["date"] <= d1)
    )
    wb = work[mask].copy()
    if wb.empty:
        return None
    wb["rk_rps60"] = wb.groupby("date")["rps_60"].rank(pct=True)
    if use_gate and board in GATE_BOARDS:
        wb = wb[wb["rk_rps60"] >= RPS_FLOOR]
    dates = np.sort(wb["date"].unique())
    sc_rows, rp_rows = [], []
    diff_days = 0
    n_open = 0
    for d in dates:
        if not A["regime_lut"].get(d, False):  # 下降段 no_open
            continue
        day = wb[wb["date"] == d]
        if day.empty:
            continue
        n_open += 1
        sc = pool_score(day, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
        t_sc = select_topn(day, sc, top_n)
        t_rp = select_topn(day, day["rk_rps60"], top_n)
        if set(t_sc["symbol"]) != set(t_rp["symbol"]):
            diff_days += 1
        sc_rows.append(t_sc[["symbol", "date"]])
        rp_rows.append(t_rp[["symbol", "date"]])
    if not sc_rows:
        return None
    res: dict = {}
    for tag, rows in (("score", sc_rows), ("rps60", rp_rows)):
        pk = pd.concat(rows, ignore_index=True)
        rets, holds = exit_rets(
            pk,
            A,
            SLOW_BULL_REGIME["trail_pct"],
            SLOW_BULL_REGIME["hard_stop"],
            SLOW_BULL_REGIME["max_hold"],
        )
        res[tag] = _agg(rets, holds)
    total_sc = sum(len(r) for r in sc_rows)
    res["diff_days"] = diff_days
    res["n_open"] = n_open
    res["picks_per_open_day"] = round(total_sc / n_open, 2) if n_open else None
    return res


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("构建窄面板...", flush=True)
    work = build_base_panel()
    dates = np.sort(work["date"].unique())
    oos_start = dates[-OOS_DAYS]
    print(
        f"全窗 {dates[0]} → {dates[-1]} ({len(dates)}d) | OOS {OOS_DAYS}d 起 {oos_start}",
        flush=True,
    )
    A = build_arrays(work)

    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "objective": "慢牛排名键 A/B: 合成 score vs rps_60 (gate∩上升段→Top-N→trail8 op_rule 实得)",
        "oos_days": OOS_DAYS,
        "window": {
            "start": str(pd.Timestamp(oos_start).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
            "n_days": OOS_DAYS,
        },
        "entry_exit": "close_hfq[T+1] 入场, trail8 收盘判定退出, 成本=COST+2×滑点",
        "top_n_list": list(TOP_N_LIST),
        "rps_gate": {"floor": RPS_FLOOR, "boards": sorted(GATE_BOARDS)},
        "crosscheck": "anchor (score top-20 无门) 应吻合 paramsweep prod_ref main +3.21% / dual +2.75%",
        "boards": {},
    }

    for b in ("main", "dual"):
        bres = {"anchor_top20_nogate": None, "variants": {}, "quarters_top10": {}}
        anchor = sim_variants(work, A, b, oos_start, dates[-1], 20, use_gate=False)
        if anchor:
            bres["anchor_top20_nogate"] = anchor["score"]
            print(
                f"  [{b}] anchor score top-20 无门: realized={anchor['score']['realized']} "
                f"n={anchor['score']['n']} fills={anchor['picks_per_open_day']}",
                flush=True,
            )
        for tn in TOP_N_LIST:
            v = sim_variants(work, A, b, oos_start, dates[-1], tn, use_gate=True)
            v_nogate = sim_variants(
                work, A, b, oos_start, dates[-1], tn, use_gate=False
            )
            cell: dict = {}
            if v is not None:
                cell["score"] = v["score"]
                cell["rps60"] = v["rps60"]
                cell["diff_days"] = v["diff_days"]
                cell["n_open"] = v["n_open"]
                cell["picks_per_open_day"] = v["picks_per_open_day"]
            if v_nogate is not None:
                cell["rps60_nogate"] = v_nogate["rps60"]
            bres["variants"][str(tn)] = cell
            sc_r = (
                v["score"]["realized"]
                if v and v["score"].get("realized") is not None
                else "n/a"
            )
            rp_r = (
                v["rps60"]["realized"]
                if v and v["rps60"].get("realized") is not None
                else "n/a"
            )
            dd = v["diff_days"] if v else "n/a"
            fill = v["picks_per_open_day"] if v else "n/a"
            print(
                f"  [{b}] top{tn}: score={sc_r} rps60={rp_r} diff_days={dd}/{v['n_open'] if v else 'n/a'} fills={fill}",
                flush=True,
            )
        # 季度稳定 (top-10, 生产门配置)
        q_bounds = pd.date_range(oos_start, dates[-1], periods=5)
        for qi in range(4):
            qs = q_bounds[qi]
            qe = q_bounds[qi + 1] - pd.Timedelta(days=1)
            q = sim_variants(
                work, A, b, np.datetime64(qs), np.datetime64(qe), 10, use_gate=True
            )
            if q is not None:
                bres["quarters_top10"][f"Q{qi + 1}"] = {
                    "score": q["score"],
                    "rps60": q["rps60"],
                    "diff_days": q["diff_days"],
                    "n_open": q["n_open"],
                }
        out["boards"][b] = bres

    del work
    gc.collect()
    ts = out["ts"]
    run_dir = BACKTEST_RESULT_DIR / f"diag_slowbull_rank_ab_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / "rank_ab.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
