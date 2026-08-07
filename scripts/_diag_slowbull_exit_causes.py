"""慢牛 离场触发归因 (2026-08-05).

route A 回测发现: 卖出规则在 ~3 日接近保本离场, 捕获不到 MFE。
本脚本归因每个 pick 的**首个触发规则**, 并对比三种离场口径:
  combined    = 6 信号 OR 移动止盈 (现行规则, route A 结果)
  stop_only   = 仅移动止盈 (无视信号)
  sig_only    = 仅 6 信号 (无视止盈)
疑点: tp_80_div 用"近60日新高+ADX衰竭"代理 (无持仓成本时), 而慢牛池
选出的正是处于高位的强势股 → 选股当天即触发, 保本离场。此脚本验证之。
结果 WORM 落盘 data/_diag_slowbull_exit_causes_<ts>.json.
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
M = 40  # max hold (8 周)


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

    def sim(picks: pd.DataFrame) -> dict:
        causes = {c: {"n": 0, "ret": []} for c in SELL_COLS}
        causes["trail_stop"] = {"n": 0, "ret": []}
        causes["cap"] = {"n": 0, "ret": []}
        stop_only, sig_only = [], []
        for sym, T in zip(picks["symbol"], picks["date"]):
            c = sym_code[str(sym)]
            lo, hi = starts[c], ends[c]
            base = lo + int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
            entry_r = base + 1
            if entry_r >= hi:
                continue
            entry = close[entry_r]
            if not np.isfinite(entry) or entry <= 0:
                continue
            cost = cost_arr[base]
            peak = 0.0
            stop_only_ret = None
            sig_only_ret = None
            combined_ret = None
            cause = None
            for k in range(1, M + 1):
                r = base + k
                if r >= hi:
                    break
                peak = max(peak, high[r] / entry - 1.0)
                stop_px = trailing_stop_price(entry, peak, ma20[r])
                stop_hit = low[r] < stop_px
                sig_hit = any_sell[r]
                if stop_only_ret is None and stop_hit:
                    stop_only_ret = close[r] / entry - 1 - cost
                if sig_only_ret is None and sig_hit:
                    sig_only_ret = close[r] / entry - 1 - cost
                if combined_ret is None and (sig_hit or stop_hit):
                    combined_ret = close[r] / entry - 1 - cost
                    if sig_hit:
                        cause = next(cx for cx in SELL_COLS if sell[cx][r])
                    else:
                        cause = "trail_stop"
                    break
            if combined_ret is None:
                r = base + M
                if r < hi:
                    combined_ret = close[r] / entry - 1 - cost
                    cause = "cap"
                else:
                    continue
            causes[cause]["n"] += 1
            causes[cause]["ret"].append(combined_ret)
            if stop_only_ret is not None:
                stop_only.append(stop_only_ret)
            if sig_only_ret is not None:
                sig_only.append(sig_only_ret)
        rows = {"max_hold": M}
        for cx in SELL_COLS + ("trail_stop", "cap"):
            rr = np.array(causes[cx]["ret"])
            rows[f"cause_{cx}"] = {
                "n": int(causes[cx]["n"]),
                "avg": round(float(rr.mean()), 4) if len(rr) else None,
                "p_win": round(float((rr > 0).mean()), 4) if len(rr) else None,
            }
        for name, rr in (
            ("stop_only", np.array(stop_only)),
            ("sig_only", np.array(sig_only)),
        ):
            rows[name] = {
                "n": int(len(rr)),
                "avg": round(float(rr.mean()), 4) if len(rr) else None,
                "p_win": round(float((rr > 0).mean()), 4) if len(rr) else None,
            }
        return rows

    out = {
        "oos_6m": {
            "start": str(pd.Timestamp(oos_dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
        }
    }
    for board, pk in picks_all.items():
        if pk.empty:
            out[board] = {"n_picks": 0}
            continue
        out[board] = {"n_picks": len(pk), **sim(pk)}
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_exit_causes_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
