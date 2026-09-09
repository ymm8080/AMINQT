"""周日自进化链 (计划任务 AMINQT-WeeklySelfevolve-Sunday, 周日 09:30, 限时 PT16H).

首版 2026-09-08 落地, 同日被误删且从未 commit (产线工具误删第 4 起);
2026-09-09 重建, 并并入闸准入复审 gate_audit (FADE_GATE 删线 21 清单日证据
直接接线当日翻车的教训 — 闸每周自动复审)。

链序 (复用 run_daily_automation 的 _STEPS / 看门狗 / 超时):
  1. weekly_review  三线质量复盘 (恒 rc=0 告警式) → weekly_review_latest.json
  2. gate_audit     闸准入复审 --gate all (恒 rc=0 告警式, WORM 报告)
  3. 读**新鲜** latest json: 零旗标 → 收工 state=ok reason=no_flags (周五晚链
     已每周重训且同用周五面板, 周日无条件重训 = 同数据重复 7h);
     有旗标 → 进化: retrain (fail-soft, 晋升闸保护旧包, 失败不拦链)
     → legacy_prob_head → legacy (关键) → deliver (关键); 关键步失败 state=failed。

刻意不做 (勿"补全"): THS 推送 (周末用户在机器旁) / parallel 块 (无当日 fresh
检查点) / 影子单 / 漂移监控。

状态: logs/weekly_selfevolve_{tag}.state.json; 日志 logs/weekly_selfevolve_{tag}.log
(WORM, 与每日链前缀隔离)。启动守卫: CHAIN_SENTINELS 冲突 (每日链/重训在跑)
→ 每 2h 复查 ×2 轮, 仍冲突 → state=skipped 收工。

用法:
  python scripts/run_weekly_selfevolve.py            # 正常周日跑
  python scripts/run_weekly_selfevolve.py --dry-run  # 只打印计划 + 当前旗标判定
  python scripts/run_weekly_selfevolve.py --force-evolve  # 无视旗标强制进化
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.settings import DATA_OTHERS_DIR  # noqa: E402
from scripts._run_guard import CHAIN_SENTINELS, find_conflicts  # noqa: E402
from scripts.run_daily_automation import (  # noqa: E402
    _STEP_TIMEOUT_S,
    _STEPS,
    PY,
    _run_step_with_watchdog,
)

LOG_DIR = os.path.join(ROOT, "logs")
_LATEST_REVIEW = DATA_OTHERS_DIR / "weekly_review_latest.json"
_GUARD_TICK_S = 2 * 3600
_GUARD_MAX_TICKS = 2
_REVIEW_STEPS = ["weekly_review", "gate_audit"]  # 恒 rc=0 告警式, 失败不拦链
_EVOLVE_STEPS = ["retrain", "legacy_prob_head", "legacy", "deliver"]
_CRITICAL = {"legacy", "deliver"}


def _state_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"weekly_selfevolve_{tag}.state.json")


def _write_state(tag: str, status: str, **kw) -> None:
    doc = {"tag": tag, "status": status, "ts": time.time(), **kw}
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_state_path(tag), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[weekly] state={status} {kw or ''}", flush=True)


def _read_latest_flags() -> tuple[bool, dict]:
    try:
        rep = json.loads(_LATEST_REVIEW.read_text(encoding="utf-8"))
        return bool(rep.get("any_flag")), rep
    except Exception:
        # 复盘不可读 → 保守不进化 (周五晚链刚重训过, 无进化损失)
        return False, {}


def _wait_for_clearance(tag: str) -> bool:
    for i in range(1, _GUARD_MAX_TICKS + 1):
        time.sleep(_GUARD_TICK_S)
        conflicts = find_conflicts(CHAIN_SENTINELS)
        if not conflicts:
            print(f"[weekly][guard] 第{i}轮复查: 无冲突 → 继续", flush=True)
            return True
        c = conflicts[0]
        print(
            f"[weekly][guard] 第{i}轮复查: 仍冲突 ({c['sentinel']} PID {c['pid']})",
            flush=True,
        )
    print("[weekly][guard] 守候超时 → 收工", flush=True)
    _write_state(tag, "skipped", reason="guard_conflict")
    return False


def _run_step(tag: str, step: str, fh) -> bool:
    argv = [PY, "-u"] + [a.replace("{tag}", tag) for a in _STEPS[step]]
    t0 = time.time()
    msg = f"[{_dt.datetime.now():%H:%M:%S} start] {step}: {' '.join(argv)}"
    print(msg, flush=True)
    print(msg, file=fh, flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    rc, timed_out = _run_step_with_watchdog(argv, fh, env, _STEP_TIMEOUT_S[step])
    if timed_out:
        rc = 124
    status = "ok" if rc == 0 else "FAIL"
    msg = f"[{_dt.datetime.now():%H:%M:%S} {status}] {step} rc={rc} ({time.time() - t0:.0f}s)"
    print(msg, flush=True)
    print(msg, file=fh, flush=True)
    return rc == 0


def main() -> int:
    ap = argparse.ArgumentParser(description="周日自进化链")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划与旗标判定")
    ap.add_argument("--force-evolve", action="store_true", help="无视旗标强制进化")
    args = ap.parse_args()

    tag = _dt.date.today().strftime("%Y%m%d")

    if not args.dry_run:
        conflicts = find_conflicts(CHAIN_SENTINELS)
        if conflicts:
            c = conflicts[0]
            _write_state(tag, "running", reason=f"guard_wait:{c['sentinel']}")
            if not _wait_for_clearance(tag):
                return 0

    if args.dry_run:
        has_flag, rep = _read_latest_flags()
        plan = _REVIEW_STEPS + (_EVOLVE_STEPS if has_flag or args.force_evolve else [])
        print(f"[weekly][dry] tag={tag} 计划={plan} (旗标={has_flag})", flush=True)
        for line, r in rep.get("lines", {}).items():
            print(f"  [dry] {line}: 旗标={r.get('flag') or '无'}", flush=True)
        _write_state(tag, "ok", reason="dry_run")
        return 0

    _write_state(tag, "running")
    log_path = os.path.join(LOG_DIR, f"weekly_selfevolve_{tag}.log")
    with open(log_path, "a", encoding="utf-8") as fh:
        for step in _REVIEW_STEPS:
            _run_step(tag, step, fh)  # 告警式: rc 非零不拦链

        has_flag, _ = _read_latest_flags()  # 读刚写出的新鲜复盘
        evolve = has_flag or args.force_evolve
        print(
            f"[weekly] 复盘旗标={has_flag}"
            f"{'/强制进化' if args.force_evolve else ''} → "
            f"{'进化' if evolve else '收工不进化 (周五晚链已重训, 勿重复)'}",
            flush=True,
        )
        failed = []
        if evolve:
            for step in _EVOLVE_STEPS:
                if not _run_step(tag, step, fh):
                    failed.append(step)

        critical = [s for s in failed if s in _CRITICAL]
        # retrain/legacy_prob_head fail-soft (晋升闸保护旧包); 关键步失败才判链败
        _write_state(
            tag,
            "failed" if critical else "ok",
            failed_steps=failed or None,
            evolved=evolve,
            reason="critical" if critical else None,
        )
    return 1 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
