"""_diag_lgbm_topn_reval.py — TOP-N 金标准复验 num_leaves (2026-08-08).

AI扫参可靠性.docx 教训后的最终定案驱动:
  判据 = TOP-N(15) 实得超额 + 三窗稳定性 (win_prim/win_tn 每窗都赢才算数)
       + min_child_samples=50 对照 (文档头号抗过拟合旋钮, 现默认 20).
  覆盖 10d_cls/10d_reg/pain/10d_rank/10d_q × main/dual.
  顺序: 决策关键且快的先跑 (cls/reg 确认已落地改动, pain/rank), 慢的 10d_q 最后.
  串行 (本机 RAM 陷阱, 严禁并发).

用法: python scripts/_diag_lgbm_topn_reval.py [--boards main,dual]
输出: data/_diag_lgbm_leaves_ms_{ts}.json (每族每板, WORM) +
      data/_diag_lgbm_topn_reval_manifest_{ts}.json
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
# 每族候选: cls/reg 加 min_child_samples=50 对照 (AI扫参可靠性: 关键抗过拟合旋钮)
CAND = {
    "10d_cls": "num_leaves=15;num_leaves=31;num_leaves=63;num_leaves=15,min_child_samples=50",
    "10d_reg": "num_leaves=15;num_leaves=31;num_leaves=63;num_leaves=15,min_child_samples=50",
    "pain": "num_leaves=15;num_leaves=31;num_leaves=63",
    "10d_rank": "num_leaves=15;num_leaves=31;num_leaves=63",
    "10d_q": "num_leaves=15;num_leaves=31;num_leaves=63",
}
ORDER = ("10d_cls", "10d_reg", "pain", "10d_rank", "10d_q")


def run(board: str, kind: str) -> str:
    """跑一个 (board, kind) TOP-N 复验, 返回输出 JSON 路径."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    cmd = [sys.executable, SWEEP, "--board", board, "--kind", kind,
           "--cand", CAND[kind], "--n", "15"]
    print(f"\n=== {board} / {kind} @ {time.strftime('%H:%M:%S')} ===", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if any(skip in line for skip in ("DeprecationWarning", "eval_set =", "  eval_X")):
            continue
        print("  " + line, flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"{board}/{kind} failed rc={r.returncode}")
    for line in out.splitlines():
        if "[saved]" in line:
            return line.split("[saved]")[1].strip()
    raise RuntimeError(f"{board}/{kind} 无保存路径")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", default="main,dual")
    ap.add_argument("--kinds", default=",".join(ORDER),
                    help="逗号分隔的子集, 如 10d_cls,10d_reg (family 扫描已覆盖 pain/q/rank 时可只补 cls/reg)")
    args = ap.parse_args()
    boards = [b.strip() for b in args.boards.split(",") if b.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip() in CAND]
    manifest = {"boards": boards, "n_top": 15, "cand": CAND, "kinds": {}}
    for board in boards:
        for kind in kinds:
            path = run(board, kind)
            manifest.setdefault("kinds", {}).setdefault(kind, {})[board] = path
    out_path = f"data/_diag_lgbm_topn_reval_manifest_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(f"\n[saved manifest] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
