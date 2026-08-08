"""诊断: 08-07 慢牛池为何为空. 加载 load_panel, 查末 5 交易日 gate_slow_bull 计数 + daily_slowbull_pool 结果."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


from app.pipeline_parallel.backtest import load_panel
from app.pipeline_parallel.config import SLOW_BULL
from app.pipeline_parallel.signals import daily_slowbull_pool


def main() -> int:
    work = load_panel()
    print(f"rows={len(work):,} latest={work['date'].max():%Y-%m-%d}", flush=True)

    last_dates = sorted(work["date"].unique())[-5:]
    for b in ("main", "dual"):
        wb = work[work["board"] == b]
        print(f"\n=== board={b} ===", flush=True)
        for d in last_dates:
            day = wb[wb["date"] == d]
            n_gate = int(day["gate_slow_bull"].sum()) if "gate_slow_bull" in day.columns else -1
            n_up = int((day.loc[day["gate_slow_bull"], "slow_bull_regime"]).sum())
            pool = daily_slowbull_pool(wb, d, b, SLOW_BULL, SLOW_BULL.top_n)
            print(
                f"  {d:%Y-%m-%d}: 行={len(day):,} gate={n_gate} up={n_up} "
                f"pool={len(pool)}{' (空)' if pool.empty else ''}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
