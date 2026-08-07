"""慢牛 买入纪律(均线低吸)门控入场回测 (2026-08-05, route A 补全).

exit_variants 证明退出层改动救不回幅度 (median hold 恒 3 日, realized≈0)。
根因 = 入场在 T+1 收盘(高位), 而文档 §3.1 本就要求"均线低吸"回踩才买、
§3.2 不追高 (chase_high)。本脚本把买入纪律实现进回测:
  选股日 T 后 10 日内, 首个 (pullback_ma5 或 pullback_ma10 或 shrink_vol)
  且无任一不买flag (chase_high/vol_spike_up/adx_falling) 之日 → 当日收盘买入;
  10 日内无买点 → 放弃 (n_no_entry)。
买入后按现行 6 卖出信号 + 移动止盈 (cur) 离场, max_hold=40 (自买入日起)。
对比: 同池 T+1 收盘直买 (现行模拟) vs 低吸门控。
WORM 落盘 data/_diag_slowbull_buy_entry_<ts>.json.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import COST, load_panel, slippage_tier
from app.pipeline_parallel.config import OOS_WINDOWS, SLOW_BULL
from app.pipeline_parallel.signals import daily_slowbull_pool, trailing_stop_price

SELL_COLS = (
    "below_ma20",
    "adx_broken",
    "big_drop",
    "below_ma5_3d",
    "turnover_spike",
    "tp_80_div",
)
BUY_COLS = ("pullback_ma5", "pullback_ma10", "shrink_vol")
NOBUY_COLS = ("chase_high", "vol_spike_up", "adx_falling")
BUY_LOOKBACK = 10
M = 40


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    oos_dates = dates[-OOS_WINDOWS["6m"] :]

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
    buy = {c: w[c].values.astype(bool) for c in BUY_COLS}
    no_buy = {c: w[c].values.astype(bool) for c in NOBUY_COLS}
    sell = {c: w[c].values.astype(bool) for c in SELL_COLS}
    any_sell = np.zeros(len(w), dtype=bool)
    for c in SELL_COLS:
        any_sell |= sell[c]
    cost_arr = COST + 2 * np.array([slippage_tier(v) for v in w["adv20"].values])

    picks_all = {}
    for board in ("main", "dual"):
        picks = []
        for d in oos_dates:
            pool = daily_slowbull_pool(work, d, board, SLOW_BULL, SLOW_BULL.top_n)
            if len(pool):
                picks.append(pool[["symbol", "date"]])
        picks_all[board] = (
            pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
        )
    del work
    gc.collect()

    def sim(picks: pd.DataFrame, mode: str) -> dict:
        # mode = "cur"(T+1 直买) | "dip"(低吸门控)
        rets, holds, peaks, no_entry = [], [], [], 0
        for sym, T in zip(picks["symbol"], picks["date"]):
            c = sym_code[str(sym)]
            lo, hi = starts[c], ends[c]
            base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
            if mode == "cur":
                entry_k = 1
            else:
                entry_k = None
                for k in range(1, BUY_LOOKBACK + 1):
                    r = base + k
                    if r >= hi:
                        break
                    sig = any(buy[cx][r] for cx in BUY_COLS)
                    bad = any(no_buy[cx][r] for cx in NOBUY_COLS)
                    if sig and not bad:
                        entry_k = k
                        break
                if entry_k is None:
                    no_entry += 1
                    continue
            entry_r = base + entry_k
            if entry_r >= hi:
                continue
            entry = close[entry_r]
            if not np.isfinite(entry) or entry <= 0:
                continue
            cost = cost_arr[base + entry_k - 1]
            peak = 0.0
            exit_ret = None
            for k in range(entry_k + 1, entry_k + M + 1):
                r = base + k
                if r >= hi:
                    break
                peak = max(peak, high[r] / entry - 1.0)
                sp = trailing_stop_price(entry, peak, ma20[r])
                if any_sell[r] or low[r] < sp:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(k - entry_k)
                    break
            if exit_ret is None:
                r = base + entry_k + M
                if r < hi:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(M)
                else:
                    continue
            rets.append(exit_ret)
            peaks.append(peak)
        rr = np.array(rets)
        return {
            "n_trades": int(len(rr)),
            "n_no_entry": int(no_entry),
            "p_win": round(float((rr > 0).mean()), 4) if len(rr) else None,
            "avg": round(float(rr.mean()), 4) if len(rr) else None,
            "median_hold": float(np.median(holds)) if holds else None,
            "avg_mfe_from_entry": round(float(np.mean(peaks)), 4) if peaks else None,
        }

    out = {
        "oos_6m": {
            "start": str(pd.Timestamp(oos_dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
            "max_hold": M,
            "buy_lookback": BUY_LOOKBACK,
            "note": "cur=T+1收盘直买; dip=10日内均线低吸(回踩且不追高)买",
        }
    }
    for board, pk in picks_all.items():
        if pk.empty:
            out[board] = {"n_picks": 0}
            continue
        out[board] = {"n_picks": len(pk), "cur": sim(pk, "cur"), "dip": sim(pk, "dip")}
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_buy_entry_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
