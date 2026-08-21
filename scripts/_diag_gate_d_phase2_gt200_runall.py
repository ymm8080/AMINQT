"""_diag_gate_d_phase2_gt200_runall.py — >200 特征候选串行链: 训练 + replay (2026-08-18).

用户扩展调查: 200 特征 (neg200) 赢下 250d 头对头后, 问"更多特征是否更好".
候选 = 30 窗多窗口扫描 (gate_d_multiwindow_20260818_065249) 频率排名:
  cand300 = Top300 (选中频率降序), candALL = 全量 324 有效特征.
每候选: phase2_train (dual-only 同配方, bundle dual_phase2_<cand>.pkl)
        → hitrate_topn replay (--bundle/--cand, 250d).
任一失败即停. 决策由 _diag_gate_d_phase2_decide.py 对 CSV 出 (基线=neg200).

用法: python scripts/_diag_gate_d_phase2_gt200_runall.py
"""

from __future__ import annotations

import subprocess
import sys
import time

JOBS = [
    ("selected_dual_cand300_20260818.json", "cand300"),
    ("selected_dual_candALL_20260818.json", "candALL"),
]


def main() -> int:
    for pin, cand in JOBS:
        t0 = time.time()
        print(
            f"\n===== [gt200 {cand}] 训练 {pin} 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
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
            f"===== [gt200 {cand}] 训练 rc={rc} 耗时 {(time.time() - t0) / 60:.0f}min =====",
            flush=True,
        )
        if rc != 0:
            print(f"FATAL: {cand} 训练失败 rc={rc}, 停止后续", flush=True)
            return rc
        t1 = time.time()
        print(
            f"\n===== [gt200 {cand}] replay 开始 {time.strftime('%Y-%m-%d %H:%M:%S')} =====",
            flush=True,
        )
        rc = subprocess.call(
            [
                sys.executable,
                "scripts/_diag_legacy_hitrate_topn.py",
                "--bundle",
                f"models/pipeline1/dual_phase2_{cand}.pkl",
                "--cand",
                cand,
            ]
        )
        print(
            f"===== [gt200 {cand}] replay rc={rc} 耗时 {(time.time() - t1) / 60:.0f}min =====",
            flush=True,
        )
        if rc != 0:
            print(f"FATAL: {cand} replay 失败 rc={rc}, 停止后续", flush=True)
            return rc
    print("\n===== >200 候选 训练+replay 全部完成 =====", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
