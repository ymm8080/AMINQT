"""_diag_gate_d_phase2_replay_runall.py — Phase 2 四候选 250d OOS replay 串行驱动器 (2026-08-18).

pin50 → candA → candB → neg200, 每候选跑 _diag_legacy_hitrate_topn.py 全量
(--slice 420 --eval 250) 并传 --bundle/--cand, 任一失败即停.
汇总决策表 (总体+4 子窗, 预注册规则) 由阶段脚本 phase2_decide 从 WORM CSV 出.

用法: python scripts/_diag_gate_d_phase2_replay_runall.py
"""

from __future__ import annotations

import subprocess
import sys
import time

JOBS = [
    ("models/pipeline1/dual_phase2_pin50.pkl", "pin50"),
    ("models/pipeline1/dual_phase2_candA.pkl", "candA"),
    ("models/pipeline1/dual_phase2_candB.pkl", "candB"),
    ("models/pipeline1/dual_phase2_neg200.pkl", "neg200"),
]


def main() -> int:
    for bundle, cand in JOBS:
        t0 = time.time()
        print(
            f"\n===== [replay {cand}] {bundle} 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
            flush=True,
        )
        rc = subprocess.call(
            [
                sys.executable,
                "scripts/_diag_legacy_hitrate_topn.py",
                "--bundle",
                bundle,
                "--cand",
                cand,
            ]
        )
        print(
            f"===== [replay {cand}] rc={rc} 耗时 {(time.time() - t0) / 60:.0f}min "
            f"{time.strftime('%H:%M:%S')} =====",
            flush=True,
        )
        if rc != 0:
            print(f"FATAL: replay {cand} 失败 rc={rc}, 停止后续候选", flush=True)
            return rc
    print("\n===== Phase 2 四候选 replay 全部完成 =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
