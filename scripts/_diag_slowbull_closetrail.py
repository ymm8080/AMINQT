# -*- coding: utf-8 -*-
"""慢牛 收盘价移动止盈 (close-trailing) 退出回测 (2026-08-05).

selectivity 证明 score 无法区分预期收益 (MFE40 扁平 24-40%), 瓶颈是兑现
(MFE 26-35% 但 close40≈0). ht 退出 (low 触线) 只拿到 +1.2~1.3%. 本脚本测
最后一个兑现杠杆: 收盘价移动止盈 —— 跟踪**收盘**峰值, 收盘从峰值回落
trail_pct 才走 (非日内 low 触线, 更现实/更温柔), + 硬止损 -8% + max_hold 40.

变体:
  cur       = 现行 6 信号 + low 触线移动止盈 (参考)
  trail5    = 收盘回落 5% 走 + 硬止损 -8%
  trail8    = 收盘回落 8% 走 + 硬止损 -8%
  trail15   = 收盘回落 15% 走 + 硬止损 -8%
  hardonly  = 无移动止盈, 仅硬止损 -8% (纯砍亏, 其余持有到期)
退出在收盘判定, exit_ret=close/entry-1-cost. WORM 落盘.
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

SELL_COLS = ("below_ma20", "adx_broken", "big_drop",
             "below_ma5_3d", "turnover_spike", "tp_80_div")
M = 40
HARD_STOP = 0.92


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    oos_dates = dates[-OOS_WINDOWS["6m"]:]

    picks_all = {}
    for b in ("main", "dual"):
        picks = []
        for d in oos_dates:
            pool = daily_slowbull_pool(work, d, b, SLOW_BULL, SLOW_BULL.top_n)
            if len(pool):
                picks.append(pool[["symbol", "date"]])
        picks_all[b] = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()

    w = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    uniques, codes = np.unique(w["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    sym_code = {s: int(c) for c, s in enumerate(uniques)}
    dates_dt = w["date"].values.astype("datetime64[ns]")
    close = w["close_hfq"].values
    low = w["low_hfq"].values
    ma20 = w["ma20"].values
    sell = {c: w[c].values.astype(bool) for c in SELL_COLS}
    any_sell = np.zeros(len(w), dtype=bool)
    for c in SELL_COLS:
        any_sell |= sell[c]
    cost_arr = COST + 2 * np.array([slippage_tier(v) for v in w["adv20"].values])

    def sim(picks: pd.DataFrame, mode: str) -> dict:
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
                if mode == "cur":
                    peak = max(peak, low[r] / entry - 1.0)
                    stop_hit = low[r] < trailing_stop_price(entry, peak, ma20[r])
                    if any_sell[r] or stop_hit:
                        exit_ret = close[r] / entry - 1 - cost
                        holds.append(k)
                        break
                else:
                    ret = close[r] / entry - 1.0
                    peak = max(peak, ret)
                    trail = 0.0 if mode == "hardonly" else float(mode[5:]) / 100.0
                    if ret <= peak - trail or ret <= HARD_STOP - 1.0:
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

    modes = ("cur", "trail5", "trail8", "trail15", "hardonly")
    out = {"oos_6m": {"start": str(pd.Timestamp(oos_dates[0]).date()),
                      "end": str(pd.Timestamp(dates[-1]).date()),
                      "max_hold": M, "hard_stop": HARD_STOP,
                      "note": "收盘价移动止盈: 收盘从峰值回落 X% 走 (非日内触线), "
                              "收盘判定, 硬止损 -8%"}
           }
    for b, pk in picks_all.items():
        if pk.empty:
            out[b] = {"n_picks": 0}
            continue
        out[b] = {"n_picks": len(pk),
                  **{m: sim(pk, m) for m in modes}}
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_closetrail_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
