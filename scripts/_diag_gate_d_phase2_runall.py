"""_diag_gate_d_phase2_runall.py — Phase 2 四候选串行训练驱动器 (2026-08-18).

pin50 基线 → 候选A → 候选B → 200 负控, 串行 (双任务并发必 OOM 须串行),
任一失败即停. 日志由调用方重定向.

用法: python scripts/_diag_gate_d_phase2_runall.py
"""

from __future__ import annotations

import subprocess
import sys
import time

JOBS = [
    ("selected_dual_pinned.json", "pin50"),
    ("selected_dual_candA_20260818.json", "candA"),
    ("selected_dual_candB_20260818.json", "candB"),
    ("selected_dual_20260817T152655.json", "neg200"),
]


def main() -> int:
    for pin, cand in JOBS:
        t0 = time.time()
        print(
            f"\n===== [{cand}] {pin} 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
            flush=True,
        )
        rc = subprocess.call(
            [
                sys.executable,
                "scripts/_diag_gate_d_phase2_train.py",
                "--pin",
                pin,
                "--cand",
                cand,
            ]
        )
        print(
            f"===== [{cand}] rc={rc} 耗时 {(time.time() - t0) / 60:.0f}min "
            f"{time.strftime('%H:%M:%S')} =====",
            flush=True,
        )
        if rc != 0:
            print(f"FATAL: {cand} 训练失败 rc={rc}, 停止后续候选", flush=True)
            return rc
    print("\n===== Phase 2 四候选训练全部完成 =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
