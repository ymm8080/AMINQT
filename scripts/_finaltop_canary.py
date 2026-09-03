"""晋升后 canary: 新 current vs 晋升前 backup 定期重放, 决定性坏签可回退.

(2026-09-02 防坏签任务#7b) 终榜闸判词基于 48 日回放, 但构建间混沌 (±2pp/日)
意味着晋升后的新交易日才是真 OOS. 本脚本读 finaltop_canary_state.json
(retrain 晋升时写入: backup 路径/晋升 tag/窗口天数), 用终榜回放工具重放
backup(A) vs current(B): 判词 FAIL 且 d3_full < -0.005 (明确跌出非劣带) =
决定性坏签 → --revert 时把 backup 恢复为 current 并回滚 current_meta
(镜像 retrain 切换语义). 每板最多重放 days 次, 窗口满 → status=done.

用法: python scripts/_finaltop_canary.py [--revert]
非关键步骤: 无 state / 今日已跑 / 窗口已满 / 回放无判词 (不消窗口) → exit 0.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from config.settings import LEGACY_TOP10_SECOND_VOTE, data_others_path
from scripts._retrain_legacy_full import FINALTOP_TOOL, MODEL_DIR, ROOT

STATE_REL = "diag/finaltop_canary_state.json"
TOL_BAND = 0.005  # 决定性坏签带: d3_full < tol_full - 此值 (明确劣于非劣带)


def _state_path():
    return data_others_path(STATE_REL)


def load_state() -> dict:
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def should_run(entry: dict, today: str) -> bool:
    if entry.get("status", "active") != "active":
        return False
    ran = entry.get("ran", {})
    return today not in ran and len(ran) < int(entry.get("days", 10))


def decisive_fail(verdict: dict, tol_full: float = 0.0) -> bool:
    return (
        bool(verdict.get("ok"))
        and not verdict["pass"]
        and float(verdict["d3_full"]) < tol_full - TOL_BAND
    )


def replay(board: str, backup: str, current: str, eval_days: int) -> dict:
    """终榜回放 backup(A) vs current(B) → verdict_from_payload 判词."""
    from app.pipeline1.finaltop_verdict import verdict_from_payload

    diag_dir = data_others_path("diag")
    before = set(glob.glob(str(diag_dir / "_dual_pkg_finaltop_compare_*.json")))
    flag = "--main-bundles" if board == "main" else "--dual-bundles"
    cmd = [
        sys.executable,
        FINALTOP_TOOL,
        "--boards",
        board,
        "--bundles",
        "a,b",
        flag,
        f"{backup},{current}",
        "--eval-days",
        str(eval_days),
        "--guard-exclude-pid",
        str(os.getpid()),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5400,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout"}
    new_json = sorted(
        set(glob.glob(str(diag_dir / "_dual_pkg_finaltop_compare_*.json"))) - before
    )
    if proc.returncode != 0 or not new_json:
        return {"ok": False, "reason": f"tool_rc{proc.returncode}"}
    payload = json.loads(Path(new_json[-1]).read_text(encoding="utf-8"))
    return verdict_from_payload(
        payload,
        board,
        tol_half=float(LEGACY_TOP10_SECOND_VOTE.get("tol_half", -0.005)),
        win_rate_min=float(LEGACY_TOP10_SECOND_VOTE.get("win_rate_min", 0.5)),
        min_days=int(LEGACY_TOP10_SECOND_VOTE.get("min_days", 10)),
    )


def revert(board: str, entry: dict) -> None:
    """镜像 retrain 切换语义: backup 恢复为 current + current_meta 回滚."""
    from app.pipeline1.model_meta import load_modules, save_modules

    cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
    shutil.copy(entry["backup"], cur)
    mods = load_modules()
    mods[board] = {
        "tag": entry.get("prev_tag", ""),
        "file": os.path.basename(entry["backup"]),
        "updated": time.strftime("%Y-%m-%d %H:%M"),
    }
    save_modules(mods)


def main() -> int:
    ap = argparse.ArgumentParser(description="晋升后 canary 回放 (backup vs current)")
    ap.add_argument(
        "--revert",
        action="store_true",
        help="决定性坏签时执行回退 (缺省只留证不回退)",
    )
    args = ap.parse_args()

    state = load_state()
    if not state:
        print("[canary] 无 state (无在窗口内的晋升), 空过")
        return 0
    today = time.strftime("%Y-%m-%d")
    eval_days = int(LEGACY_TOP10_SECOND_VOTE.get("eval_days", 48))
    for board, entry in state.items():
        if not isinstance(entry, dict) or "backup" not in entry:
            continue
        if not should_run(entry, today):
            continue
        if not os.path.exists(entry["backup"]):
            entry["status"] = "no_backup"
            print(f"[canary:{board}] backup 不存在: {entry['backup']}, canary 终止")
            continue
        cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        v = replay(board, entry["backup"], cur, eval_days)
        if not v.get("ok"):
            print(f"[canary:{board}] 回放无判词 ({v.get('reason')}), 不消窗口")
            continue
        entry.setdefault("ran", {})[today] = {
            "pass": v["pass"],
            "d3_full": v["d3_full"],
            "win_rate": v["win_rate"],
        }
        if v["pass"]:
            print(
                f"[canary:{board}] PASS Δ={v['d3_full']:+.5f}/日 — 晋升维持"
                f" ({len(entry['ran'])}/{entry.get('days', 10)})"
            )
        elif decisive_fail(v):
            print(
                f"[canary:{board}] 决定性坏签 Δ={v['d3_full']:+.5f}/日 "
                f"双半 {v['d3_h1']:+.5f}/{v['d3_h2']:+.5f} 胜率 {v['win_rate']:.3f}"
            )
            if args.revert:
                revert(board, entry)
                entry["status"] = "reverted"
                print(
                    f"[canary:{board}] 已回退 current -> {entry.get('prev_tag')} "
                    f"(meta 同步)"
                )
            else:
                print(f"[canary:{board}] --revert 未给, 仅留证不回退")
        else:
            print(
                f"[canary:{board}] FAIL 非决定性 (Δ={v['d3_full']:+.5f}/日 未跌出"
                f"非劣带), 继续观察 ({len(entry['ran'])}/{entry.get('days', 10)})"
            )
        if entry.get("status", "active") == "active" and len(entry["ran"]) >= int(
            entry.get("days", 10)
        ):
            entry["status"] = "done"
            print(f"[canary:{board}] 窗口满 ({entry['days']} 次), canary 结束")
    save_state(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
