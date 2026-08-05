# -*- coding: utf-8 -*-
"""慢牛 开仓选择性 (预期收益门) 诊断 (2026-08-05).

用户原则: 预期收益不够高就不开仓. 前四条诊断证明退出/买入规则改动救不回
实得≈0, 根因=退出~3日, 离场时峰值仅+6% 而全窗 MFE +26~35%. 若选择性能
把高预期幅度子集挑出来, 再配合"持有到底"退出, 才可能放大. 本脚本验证:

1. 全 gated 池 (非仅 top-20) 按 score 五分位 → 每分位 n / MFE40 / close40
   (含成本) / P(close40>0). 回答: score 是否预测前瞻幅度.
2. 运营 top-20 池, 对比两种退出: cur (现行 6 信号+移动止盈) vs
   ht (持有到底: 仅锁盈棘轮 + 硬止损 -8%, 无视早段强度信号, max_hold=40).
   回答: 高预期幅度子集用 ht 能否兑现.
WORM 落盘 data/_diag_slowbull_selectivity_<ts>.json.
"""
from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import COST, load_panel, slippage_tier
from app.pipeline_parallel.config import ADX_SPEC, OOS_WINDOWS, SLOW_BULL
from app.pipeline_parallel.scoring import pool_score
from app.pipeline_parallel.signals import daily_slowbull_pool, trailing_stop_price

SELL_COLS = ("below_ma20", "adx_broken", "big_drop",
             "below_ma5_3d", "turnover_spike", "tp_80_div")
M = 40
HARD_STOP = 0.92  # 硬止损: 跌破成本-8% 平


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    oos_dates = dates[-OOS_WINDOWS["6m"]:]
    gc_col = f"gate_{SLOW_BULL.gate}"

    # ---- 全 gated 池: 逐 OOS 日逐板取掩码 + score, 收集 (board,symbol,date,score) ----
    rows = []
    for d in oos_dates:
        for b in ("main", "dual"):
            mask = ((work["date"] == d) & (work["board"] == b) & work[gc_col])
            day = work[mask]
            if day.empty:
                continue
            sc = pool_score(day, SLOW_BULL.pool, weights=SLOW_BULL.pool_weights)
            sub = day[["symbol", "date"]].copy()
            sub["score"] = sc.values
            sub["board"] = b
            rows.append(sub.dropna(subset=["score"]))
    gated = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    del rows
    gc.collect()
    if gated.empty:
        print("全 gated 池为空")
        return 0
    print(f"全 gated 池: {len(gated)} picks / "
          f"{gated['symbol'].nunique()} 只")

    # ---- top-20 运营池 (逐 OOS 日) ----
    picks_all = {}
    for b in ("main", "dual"):
        picks = []
        for d in oos_dates:
            pool = daily_slowbull_pool(work, d, b, SLOW_BULL, SLOW_BULL.top_n)
            if len(pool):
                picks.append(pool[["symbol", "date", "score"]])
        picks_all[b] = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    # ---- 块索引 (per-symbol) ----
    w = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    uniques, codes = np.unique(w["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    sym_code = {s: int(c) for c, s in enumerate(uniques)}
    dates_dt = w["date"].values.astype("datetime64[ns]")
    close = w["close_hfq"].values
    high = w["high_hfq"].values
    low = w["low_hfq"].values
    ma20 = w["ma20"].values
    sell = {c: w[c].values.astype(bool) for c in SELL_COLS}
    any_sell = np.zeros(len(w), dtype=bool)
    for c in SELL_COLS:
        any_sell |= sell[c]
    cost_arr = COST + 2 * np.array([slippage_tier(v) for v in w["adv20"].values])

    def fwd(df: pd.DataFrame) -> dict[str, pd.Series]:
        """对池内每个 pick 算前瞻幅度 (Series 按 df.index 标签对齐, 无数据处 NaN)."""
        lbl = df.index.to_numpy()
        syms = df["symbol"].to_numpy()
        Ts = df["date"].to_numpy()
        out = {k: pd.Series(np.nan, index=lbl, dtype=float)
               for k in ("mfe40", "c10", "c20", "c40")}
        for pos in range(len(df)):
            c = sym_code[str(syms[pos])]
            lo, hi = starts[c], ends[c]
            base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(Ts[pos])))
            r0 = base + 1
            if r0 + M >= hi:
                continue
            entry = close[r0]
            if not np.isfinite(entry) or entry <= 0:
                continue
            cost = cost_arr[base]
            out["mfe40"].at[lbl[pos]] = high[r0 + 1:r0 + M + 1].max() / entry - 1 - cost
            out["c10"].at[lbl[pos]] = close[r0 + 10] / entry - 1 - cost
            out["c20"].at[lbl[pos]] = close[r0 + 20] / entry - 1 - cost
            out["c40"].at[lbl[pos]] = close[r0 + 40] / entry - 1 - cost
        return out

    def summarize(arr: np.ndarray) -> dict:
        if len(arr) < 5:
            return {"n": int(len(arr)), "avg": None, "p_pos": None}
        return {"n": int(len(arr)), "avg": round(float(arr.mean()), 4),
                "p_pos": round(float((arr > 0).mean()), 4)}

    def sim_exit(picks: pd.DataFrame, mode: str) -> dict:
        """mode=cur 现行 | ht 持有到底(锁盈棘轮+硬止损, max_hold=40)."""
        rets, holds = [], []
        for sym, T in zip(picks["symbol"], picks["date"]):
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
                peak = max(peak, high[r] / entry - 1.0)
                if mode == "cur":
                    stop_hit = low[r] < trailing_stop_price(entry, peak, ma20[r])
                    sig_hit = any_sell[r]
                    if sig_hit or stop_hit:
                        exit_ret = close[r] / entry - 1 - cost
                        holds.append(k)
                        break
                else:  # ht: 峰值>=20% 才挂锁盈棘轮; <20% 仅硬止损
                    if peak >= 0.2:
                        stop_hit = low[r] < trailing_stop_price(entry, peak, ma20[r])
                    else:
                        stop_hit = False
                    hard = low[r] < entry * HARD_STOP
                    if stop_hit or hard:
                        exit_ret = close[r] / entry - 1 - cost
                        holds.append(k)
                        break
            if exit_ret is None:
                exit_ret = close[r0 + M] / entry - 1 - cost
                holds.append(M)
            rets.append(exit_ret)
        rr = np.array(rets)
        return {"n": int(len(rr)),
                "p_win": round(float((rr > 0).mean()), 4),
                "avg": round(float(rr.mean()), 4),
                "median_hold": float(np.median(holds)) if holds else None}

    out = {"oos_6m": {"start": str(pd.Timestamp(oos_dates[0]).date()),
                      "end": str(pd.Timestamp(dates[-1]).date()),
                      "max_hold": M, "hard_stop": HARD_STOP,
                      "note": "score=加权截面分位(gated池内); "
                              "mfemax40=离场前峰值; close_k=持有k日收盘(含成本). "
                              "ht=持有到底(仅锁盈棘轮+硬止损-8%)"}
           }

    # 1) 全 gated 池 score 五分位 → 前瞻幅度
    out["gated_quintiles"] = {}
    for b in ("main", "dual"):
        sub = gated[gated["board"] == b].copy().reset_index(drop=True)
        if sub.empty:
            out["gated_quintiles"][b] = {"n": 0}
            continue
        sub["q"] = pd.qcut(sub["score"], 5, labels=False, duplicates="drop")
        fwd_all = fwd(sub)
        blob = {}
        for q in sorted(sub["q"].unique()):
            idx = sub.index[sub["q"] == q]
            blob[f"q{int(q)}"] = {"score_range": [
                    round(float(sub["score"].loc[idx].min()), 3),
                    round(float(sub["score"].loc[idx].max()), 3)],
                **{k: summarize(fwd_all[k].loc[idx].dropna().to_numpy())
                   for k in fwd_all}}
        out["gated_quintiles"][b] = {"n_picks": len(sub), **blob}
        del sub
        gc.collect()

    # 2) top-20 运营池: 总览 + cur vs ht 退出
    out["top20"] = {}
    for b, pk in picks_all.items():
        if pk.empty:
            out["top20"][b] = {"n_picks": 0}
            continue
        fwd_top = fwd(pk)
        out["top20"][b] = {
            "n_picks": len(pk),
            "mfe40": summarize(fwd_top["mfe40"].dropna().to_numpy()),
            "close40": summarize(fwd_top["c40"].dropna().to_numpy()),
            "exit_cur": sim_exit(pk, "cur"),
            "exit_ht": sim_exit(pk, "ht"),
        }
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_selectivity_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
