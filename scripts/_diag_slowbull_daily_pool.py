"""慢牛每日池产出诊断 (2026-08-05).

核对 daily_slowbull_pool 在最近交易日有候选时能产出非空清单 (验证 runner 只跳过真空池,
非逻辑 bug). 结果 WORM 落盘 data/_diag_slowbull_daily_pool_<ts>.json.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from app.pipeline_parallel.backtest import load_panel
from app.pipeline_parallel.config import SLOW_BULL
from app.pipeline_parallel.signals import daily_slowbull_pool


def main() -> int:
    work = load_panel()
    out = {}
    for board in ("main", "dual"):
        bd = work[work["board"] == board]
        dates = sorted(bd["date"].unique())[-10:]
        rows = []
        for d in dates:
            gc = work["gate_slow_bull"] & (work["board"] == board) & (work["date"] == d)
            n_gate = int(gc.sum())
            pool = daily_slowbull_pool(work, d, board, SLOW_BULL, SLOW_BULL.top_n)
            rows.append(
                {
                    "date": str(pd.Timestamp(d).date()),
                    "gate_pass": n_gate,
                    "pool_len": len(pool),
                    "cols": list(pool.columns) if len(pool) else [],
                }
            )
        out[board] = {
            "latest": str(dates[-1].date()) if len(dates) else None,
            "last10": rows,
        }
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    fp = os.path.join("data", f"_diag_slowbull_daily_pool_{ts}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    for b, v in out.items():
        print(f"[{b}] latest={v['latest']}")
        for r in v["last10"]:
            print(
                f"  {r['date']} gate={r['gate_pass']:>3} pool={r['pool_len']:>2} "
                f"{'OK:' + str(len(r['cols'])) + 'cols' if r['pool_len'] else ''}"
            )
    print(f"\n落盘: {fp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
