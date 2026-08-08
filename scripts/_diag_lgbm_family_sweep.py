"""_diag_lgbm_family_sweep.py — quantile/rank/pain 三族 num_leaves 扫描驱动 (2026-08-08).

用户需求: "we just review reg, how about cls, quantile, pank.rank 还能遍历一遍查有更好的结果嘛"
"we can create 10d for those parameters" — 三族都建 10d 视界, 主评 10d.

调用 _diag_lgbm_param_sweep.py 子进程, 串行 (本机 RAM 陷阱, 严禁并发).
每族只扫 num_leaves {15,31,63} (reg/cls 已证明 min_child_samples 无影响, 减少 3x 耗时;
quantile 单组合 ~14min 训 5 个分位, 全 9 组合网格不可行).

用法: python scripts/_diag_lgbm_family_sweep.py [--boards main,dual]
输出: data/_diag_lgbm_leaves_ms_{ts}.json (WORM), 每族每板一份
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP = os.path.join(ROOT, "_diag_lgbm_param_sweep.py")
CAND = "num_leaves=15;num_leaves=31;num_leaves=63"


def run(board: str, kind: str) -> str:
    """跑一个 (board, kind) 扫描, 返回输出 JSON 路径."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, SWEEP, "--board", board, "--kind", kind, "--cand", CAND]
    print(f"\n=== {board} / {kind} @ {time.strftime('%H:%M:%S')} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    # 只保留进度行 + 错误 (过滤 lightgbm 噪音)
    for line in out.splitlines():
        if any(
            skip in line for skip in ("DeprecationWarning", "eval_set =", "  eval_X")
        ):
            continue
        print("  " + line, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"{board}/{kind} failed rc={r.returncode}")
    # 从 stdout 找保存路径
    for line in out.splitlines():
        if "[saved]" in line:
            return line.split("[saved]")[1].strip()
    raise RuntimeError(f"{board}/{kind} 无保存路径")
    print(f"  elapsed {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default="main,dual")
    args = ap.parse_args()
    boards = [b.strip() for b in args.boards.split(",") if b.strip()]
    manifest = {"boards": boards, "cand": CAND, "kinds": {}}
    for board in boards:
        for kind in ("pain", "10d_q", "10d_rank"):
            path = run(board, kind)
            manifest.setdefault("kinds", {}).setdefault(kind, {})[board] = path
    out_path = f"data/_diag_lgbm_family_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f"\n[saved manifest] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
