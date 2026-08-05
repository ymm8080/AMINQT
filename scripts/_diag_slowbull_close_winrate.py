# -*- coding: utf-8 -*-
"""慢牛 MFE 胜率 vs 收盘价胜率 对比诊断 (2026-08-05).

质疑: 回测"胜率 91-98%"是否虚高? 根因 = MFE 标签口径
  label_mfe_{h}d_net = max(high_hfq[T+2..T+1+h])/close_hfq[T+1]-1-cost
即"窗口内**最高价**能兑现"的概率, 在 A 股上行市场中人人皆高
(全市场基准本身也 90-96%). 本脚本对同一 OOS 6m 慢牛池计算三组数:
  P(MFE>0)                — 触摸更高价的概率 (回测口径)
  P(close[T+1+h]>entry)   — 持有到 T+k 收盘卖出的真实胜率 (可兑现)
  avg MFE vs avg 收盘收益  — 理想最优 vs 实际可得的幅度差
结果 WORM 落盘 data/_diag_slowbull_close_winrate_<ts>.json.
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

H = (10, 20, 40)


def main() -> int:
    work = load_panel()
    dates = np.sort(work["date"].unique())
    oos_d = OOS_WINDOWS["6m"]
    oos_dates = dates[-oos_d:]

    # 紧凑 forward 收盘价查找表 (T+1 买价 / T+1+h 收盘), 避免宽表 join OOM
    cf = work[["symbol", "date", "close_hfq", "adv20"]].copy()
    cf = cf.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = cf.groupby("symbol", sort=False)
    cf["entry"] = g["close_hfq"].shift(-1)                 # T+1 收盘买价
    for h in H:
        cf[f"exit{h}"] = g["close_hfq"].shift(-(1 + h))    # T+1+h 收盘
    cf["cost"] = COST + 2 * cf["adv20"].map(slippage_tier)
    lut = cf.set_index(["symbol", "date"])
    del cf, g
    gc.collect()

    labels = tuple(f"label_mfe_{h}d_net" for h in H)
    lab = work[["symbol", "date", *labels]].set_index(["symbol", "date"])

    out = {"oos_6m": {"start": str(pd.Timestamp(oos_dates[0]).date()),
                      "end": str(pd.Timestamp(dates[-1]).date()),
                      "trading_days": int(oos_d)},
           "note": "mfe_win = P(窗口内最高价>买入); close_win = P(T+k收盘>买入); "
                   "capture = avg(收盘收益)/avg(MFE) 理想可兑现比例"}
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
        pk = pk.set_index(["symbol", "date"])
        pk = pk.join(lut[["entry", "exit10", "exit20", "exit40", "cost"]])
        pk = pk.join(lab)
        n = len(pk)
        rows = {"n_picks": int(n)}
        for h in H:
            mfe = pk[f"label_mfe_{h}d_net"].dropna()
            close_ret = (pk[f"exit{h}"] / pk["entry"] - 1 - pk["cost"]).dropna()
            row = {
                "n_mfe": int(mfe.size),
                "p_mfe_win": round(float((mfe > 0).mean()), 4) if mfe.size else None,
                "avg_mfe": round(float(mfe.mean()), 4) if mfe.size else None,
                "n_close": int(close_ret.size),
                "p_close_win": round(float((close_ret > 0).mean()), 4) if close_ret.size else None,
                "avg_close_ret": round(float(close_ret.mean()), 4) if close_ret.size else None,
            }
            if row["n_close"] and row["avg_mfe"] and row["avg_mfe"] > 0:
                row["capture"] = round(row["avg_close_ret"] / row["avg_mfe"], 3)
            rows[f"T+{h}"] = row
        out[board] = rows
    del work, lab
    gc.collect()

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_close_winrate_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
