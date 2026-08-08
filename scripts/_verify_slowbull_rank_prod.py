# -*- coding: utf-8 -*-
"""慢牛排名键落地 生产面板验证 (2026-08-08): dual=rps_60+top-10 / main=score+top-20.

复用 rank_ab 诊断的窄面板构建, 用**生产路径** daily_slowbull_pool 逐 OOS 日跑两板,
检查: (a) 每日输出 ≤ 有效 top_n (dual 10 / main 20); (b) score 列单调降 (排序正确);
(c) dual 无 rps_60 < floor 违反 (先过门再排名); (d) 集中度未塌缩 (对比门-only 447 只).
结果并入 verdict.json (WORM). 无前瞻 (当日截面, 入场 T+1).
"""

from __future__ import annotations

import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from app.pipeline_parallel import signals
from app.pipeline_parallel.config import SLOW_BULL, SLOW_BULL_RANK, SLOW_BULL_REGIME, SLOW_BULL_RPS_GATE
from config.settings import BACKTEST_RESULT_DIR
from scripts._diag_slowbull_rank_ab import build_arrays, build_base_panel

OOS_DAYS = 250


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("构建窄面板...", flush=True)
    work = build_base_panel()
    dates = np.sort(work["date"].unique())
    oos_start = dates[-OOS_DAYS]
    A = build_arrays(work)
    floor = float(SLOW_BULL_RPS_GATE.get("floor", 0.0))

    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "objective": "慢牛排名键落地生产面板验证 (daily_slowbull_pool 生产路径)",
        "oos_days": OOS_DAYS,
        "window": {"start": str(pd.Timestamp(oos_start).date()), "end": str(pd.Timestamp(dates[-1]).date())},
        "rank": {"key": SLOW_BULL_RANK["key"], "boards": SLOW_BULL_RANK["boards"], "top_n": SLOW_BULL_RANK["top_n"]},
        "rps_floor": floor,
        "boards": {},
    }
    for b in ("main", "dual"):
        rows = []
        size_viol, order_viol, floor_viol = 0, 0, 0
        n_open = 0
        for d in dates:
            if d < oos_start:
                continue
            if not A["regime_lut"].get(d, False):  # 下降段 no_open
                continue
            n_open += 1
            pool = signals.daily_slowbull_pool(work, d, b, SLOW_BULL, SLOW_BULL.top_n)
            if pool.empty:
                continue
            eff = SLOW_BULL_RANK["top_n"].get(b, SLOW_BULL.top_n)
            if len(pool) > eff:
                size_viol += 1
            if not pool["score"].is_monotonic_decreasing:
                order_viol += 1
            if b == "dual":
                # 门口径 = gate 内日截面 rps_60 百分位 ≥ floor (非全市场原始 rps_60)
                cand = work[(work["date"] == d) & (work["board"] == b) & work["gate_slow_bull"]]
                if len(cand):
                    rk = cand.groupby("date")["rps_60"].rank(pct=True)
                    rk_map = dict(zip(cand["symbol"], rk))
                    if any(rk_map.get(s, 1.0) < floor - 1e-9 for s in pool["symbol"]):
                        floor_viol += 1
            rows.append(pool)
        bres: dict = {"n_open": n_open}
        if rows:
            allp = pd.concat(rows, ignore_index=True)
            uniq = int(allp["symbol"].nunique())
            bres.update(
                {
                    "n_days_with_picks": int(len(rows)),
                    "n_picks": int(len(allp)),
                    "unique_stocks": uniq,
                    "picks_per_stock": round(len(allp) / uniq, 2) if uniq else None,
                    "picks_per_day": round(len(allp) / len(rows), 2),
                    "max_day_size": int(allp.groupby("date").size().max()),
                    "size_violations": size_viol,
                    "order_violations": order_viol,
                    "rps_floor_violations": floor_viol,
                }
            )
        else:
            bres.update({"n_days_with_picks": 0, "n_picks": 0})
        out["boards"][b] = bres
        print(f"  [{b}] {bres}", flush=True)
    del work
    gc.collect()

    ts = out["ts"]
    run_dir = BACKTEST_RESULT_DIR / f"diag_slowbull_rank_ab_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = run_dir / "prod_verify.json"
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
