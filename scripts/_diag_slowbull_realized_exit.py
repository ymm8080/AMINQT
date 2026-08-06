# -*- coding: utf-8 -*-
"""慢牛 真实离场规则回测 (2026-08-05, 用户路线 A).

把文档 §4.2 卖出信号 + §4.3 移动止盈真正模拟进回测, 测"实际离场收益"。
对 OOS 6m 每日慢牛池 (同一批 picks) 逐股模拟:
  买入 = close_hfq[T+1] (与 MFE/持有到期基准同一买价);
  每日检查 6 个卖出信号 (below_ma20/adx_broken/big_drop/below_ma5_3d/
    turnover_spike/tp_80_div) + 移动止盈 (trailing_stop_price: 峰值盈利
    >=100%→ma20, >=50%→+40%, >=20%→+15%; 峰值用 high_hfq 棘轮, 跌破用 low_hfq);
  任一触发 → 当日收盘卖出; 至 max_hold 无触发 → 到期收盘卖出。
对比三组 (每 max_hold 10/20/40):
  realized   = 卖出规则实际收益 (路线 A 目标)
  baseline   = 无规则持有到 T+cap 收盘
  mfe        = 理论天花板 (窗口内最高价可兑现)
结果 WORM 落盘 data/_diag_slowbull_realized_exit_<ts>.json.
本脚本是交易模拟 (非特征计算), 逐 pick 循环 1400 级 × 40 日, 毫秒级。
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
CAPS = (10, 20, 40)


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    oos_d = OOS_WINDOWS["6m"]
    oos_dates = dates[-oos_d:]

    # 排序 + 按 symbol 分块, 供每 pick O(1) 定位未来交易日
    w = work.sort_values(["symbol", "date"]).reset_index(drop=True)
    uniques, codes = np.unique(w["symbol"].values, return_inverse=True)
    sizes = np.bincount(codes)
    starts = np.zeros(len(uniques), dtype=np.int64)
    starts[1:] = np.cumsum(sizes)[:-1]
    ends = np.cumsum(sizes)
    sym_code = {s: int(c) for c, s in enumerate(uniques)}

    arr = {
        col: w[col].values
        for col in ("close_hfq", "high_hfq", "low_hfq", "ma20", "adv20", "date")
    }
    dates_dt = arr["date"].astype("datetime64[ns]")
    any_sell = np.zeros(len(w), dtype=bool)
    for col in SELL_COLS:
        any_sell |= w[col].values.astype(bool)
    cost_arr = COST + 2 * np.array([slippage_tier(v) for v in arr["adv20"]])
    close, high, low = arr["close_hfq"], arr["high_hfq"], arr["low_hfq"]
    ma20 = arr["ma20"]

    def sim(picks: pd.DataFrame) -> dict:
        rows = {"n_picks": len(picks)}
        for M in CAPS:
            realized, _held, base_hold, mfe, hold_days = [], [], [], [], []
            for sym, T in zip(picks["symbol"], picks["date"]):
                c = sym_code[str(sym)]
                lo, hi = starts[c], ends[c]
                j = int(np.searchsorted(dates_dt[lo:hi], np.datetime64(T)))
                base = lo + j
                entry_r = base + 1
                if entry_r >= hi:  # 无 T+1 买价 → 弃
                    continue
                entry = close[entry_r]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                cost = cost_arr[base]
                peak = 0.0
                exit_ret = None
                for k in range(1, M + 1):
                    r = base + k
                    if r >= hi:  # 未来窗截断 → 弃 (保守)
                        break
                    peak = max(peak, high[r] / entry - 1.0)
                    stop_px = trailing_stop_price(entry, peak, ma20[r])
                    if any_sell[r] or low[r] < stop_px:
                        exit_ret = close[r] / entry - 1 - cost
                        hold_days.append(k)
                        break
                else:
                    r = base + M
                    if r < hi:
                        exit_ret = close[r] / entry - 1 - cost
                        hold_days.append(M)
                if exit_ret is None:
                    continue
                realized.append(exit_ret)
                bh = close[base + M] / entry - 1 - cost if (base + M) < hi else np.nan
                base_hold.append(bh)
                mcol = f"label_mfe_{M}d_net"
                mfe.append(
                    float(w.at[base, mcol]) if not pd.isna(w.at[base, mcol]) else np.nan
                )
            realized = np.array(realized)
            base_hold = np.array(base_hold, dtype=float)
            mfe = np.array(mfe, dtype=float)
            rows[f"max_hold_{M}"] = {
                "n": int(len(realized)),
                "p_win": round(float((realized > 0).mean()), 4)
                if len(realized)
                else None,
                "avg_realized": round(float(realized.mean()), 4)
                if len(realized)
                else None,
                "avg_baseline_hold": round(float(np.nanmean(base_hold)), 4)
                if np.isfinite(base_hold).any()
                else None,
                "avg_mfe": round(float(np.nanmean(mfe)), 4)
                if np.isfinite(mfe).any()
                else None,
                "median_hold_days": float(np.median(hold_days)) if hold_days else None,
                "delta_vs_hold": round(
                    float(np.nanmean(realized) - np.nanmean(base_hold)), 4
                )
                if len(realized) and np.isfinite(base_hold).any()
                else None,
            }
        return rows

    out = {
        "oos_6m": {
            "start": str(pd.Timestamp(oos_dates[0]).date()),
            "end": str(pd.Timestamp(dates[-1]).date()),
            "trading_days": int(oos_d),
        },
        "entry": "close_hfq[T+1] (同 MFE 基准); exit=首个卖出信号或止盈线, "
        "否则 T+cap 收盘; 成本=COST+2×滑点",
    }
    for board in ("main", "dual"):
        picks = []
        for d in oos_dates:
            pool = daily_slowbull_pool(work, d, board, SLOW_BULL, SLOW_BULL.top_n)
            if len(pool):
                picks.append(pool[["symbol", "date"]])
        if not picks:
            out[board] = {"n_picks": 0}
            continue
        pk = pd.concat(picks, ignore_index=True)
        out[board] = sim(pk)
        del pk
        gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_realized_exit_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
