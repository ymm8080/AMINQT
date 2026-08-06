# -*- coding: utf-8 -*-
"""慢牛 离场规则变体回测 (2026-08-05, route A 放大幅度).

exit_causes 发现: below_ma20/below_ma5_3d/adx_broken (MA/ADX 破位退出)
占 ~42% 退出且平均 -3~-6% (below_ma20 胜率 0%), 在移动止盈棘轮 (+20%峰值)
启动前就把仓位了结 → realized≈0。本脚本试三种变体, 量化可回收的幅度:
  cur        = 现行规则 (6 信号 + trailing_stop_price)
  g0         = cur + 破位退出仅当 profit<0 (赢家交给止盈/换手/顶背离, 不因破位割肉)
  g0_ratchet = g0 + 棘轮提前: 峰值>=12% 锁 +8% (原 >=20% 锁 +15%);
                峰值<12% 无止损 (去掉 ma20 保底触发, 靠破位守门)
max_hold=40 (8 周), 成本=COST+2×滑点. WORM 落盘.
"""

from __future__ import annotations

import gc
import json
import os

import numpy as np
import pandas as pd

from app.pipeline_parallel.backtest import COST, load_panel, slippage_tier
from app.pipeline_parallel.config import OOS_WINDOWS, SLOW_BULL
from app.pipeline_parallel.signals import daily_slowbull_pool

SELL_COLS = (
    "below_ma20",
    "adx_broken",
    "big_drop",
    "below_ma5_3d",
    "turnover_spike",
    "tp_80_div",
)
TREND_BREAK = ("below_ma20", "adx_broken", "below_ma5_3d")
M = 40


def stop_px(mode: str, entry: float, peak: float, ma20: float) -> float | None:
    """返回止盈线价格; None = 无止盈."""
    if mode == "cur":
        if peak >= 1.0:
            return ma20
        if peak >= 0.5:
            return entry * 1.40
        if peak >= 0.2:
            return entry * 1.15
        return ma20  # 现行: <20% 峰值 → ma20 保底
    if mode == "g0":
        if peak >= 1.0:
            return ma20
        if peak >= 0.5:
            return entry * 1.40
        if peak >= 0.2:
            return entry * 1.15
        return None  # <20% 无止损 (靠破位守门)
    if mode == "g0_ratchet":
        if peak >= 1.0:
            return ma20
        if peak >= 0.5:
            return entry * 1.40
        if peak >= 0.12:
            return entry * 1.08  # 提前棘轮: +12% 锁 +8%
        return None
    raise ValueError(mode)


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
        rets, holds, mfe = [], [], []
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
            exit_ret = None
            for k in range(1, M + 1):
                r = base + k
                if r >= hi:
                    break
                profit = close[r] / entry - 1.0
                peak = max(peak, high[r] / entry - 1.0)
                sp = stop_px(mode, entry, peak, ma20[r])
                stop_hit = sp is not None and low[r] < sp
                trend_break = any(sell[cx][r] for cx in TREND_BREAK) and profit < 0
                other_sig = (
                    sell["big_drop"][r]
                    or sell["turnover_spike"][r]
                    or sell["tp_80_div"][r]
                )
                if stop_hit or trend_break or other_sig:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(k)
                    break
            if exit_ret is None:
                r = base + M
                if r < hi:
                    exit_ret = close[r] / entry - 1 - cost
                    holds.append(M)
                else:
                    continue
            rets.append(exit_ret)
            mcol = f"label_mfe_{M}d_net"
            mfe.append(
                float(w.at[base, mcol]) if not pd.isna(w.at[base, mcol]) else np.nan
            )
        rr = np.array(rets)
        return {
            "n": int(len(rr)),
            "p_win": round(float((rr > 0).mean()), 4),
            "avg": round(float(rr.mean()), 4),
            "median_hold": float(np.median(holds)) if holds else None,
            "avg_mfe": round(float(np.nanmean(mfe)), 4) if mfe else None,
        }

    out = {
        "oos_6m": {
            "start": str(pd.Timestamp(oos_dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
            "max_hold": M,
            "note": "cur=现行; g0=破位退出仅当profit<0; g0_ratchet=g0+棘轮12%锁8%",
        }
    }
    for board, pk in picks_all.items():
        if pk.empty:
            out[board] = {"n_picks": 0}
            continue
        out[board] = {m: sim(pk, m) for m in ("cur", "g0", "g0_ratchet")}
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_exit_variants_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
