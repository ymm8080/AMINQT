# -*- coding: utf-8 -*-
"""PARALLEL-only 夜链入口 (2026-09-28).

接替已停用的 AMINQT-DailyAutomation-2330, 在同一 23:30 时间槽只跑 PARALLEL 模块:

  [refresh]          scripts/_refresh_parallel_checkpoints.py   3y 检查点重建 (19:15 fetch 后)
  [parallel]         python -m app.pipeline_parallel.runner     sniper/fusion/slow_bull
  [prob_head]        scripts/_train_parallel_prob_head.py       概率头 (21 交易日到期才重拟)
  [deliver_parallel] scripts/_shortlist_t5_t10.py <tag>         清单落 STOCK_LIST_DIR
  [drift_parallel]   scripts/_monitor_parallel_drift.py         dual 漂移监控 (只读)
  [ths_push]         scripts/_ths_watchlist_push.py <tag>       THS 自选股推送 (UI 驱动)
  [ths_flush_guard]  scripts/_ths_flush_guard.py <tag>          推送结果校验 (不带 --apply)

步骤定义 (_STEPS)、超时 (_STEP_TIMEOUT_S)、依赖 (_DEPENDS)、看门狗杀进程树、防休眠,
全部复用 scripts/run_daily_automation.py 的既有实现; 本文件只决定"跑哪几步 + 记哪个
state/log", 不复制那套机制.

注意: ths_push 是单进程推三个源 (parallel / legacy / genious), 无法只推 parallel —
按用户 2026-09-28 决定一并推.

产物 (与主线分开命名, 避免 state 互相覆盖):
  logs/parallel_automation_<tag>.log         步骤 stdout/stderr (WORM, 追加)
  logs/parallel_automation_<tag>.state.json  running / ok / failed / skipped / interrupted

调用:
  python scripts/run_parallel_automation.py                  # 全链
  python scripts/run_parallel_automation.py --dry-run        # 只打印步骤序列
  python scripts/run_parallel_automation.py --tag YYYYMMDD --force
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts._run_guard import (  # noqa: E402
    CHAIN_SENTINELS,
    find_conflicts,
    skip_reason,
)
from scripts.run_daily_automation import (  # noqa: E402
    _DEPENDS,
    _STEP_TIMEOUT_S,
    _STEPS,
    PY,
    _is_interrupt_rc,
    _prevent_sleep,
    _run_step_with_watchdog,
)

LOG_DIR = os.path.join(ROOT, "logs")

# 执行顺序 (用户令 2026-09-28: "全 PARALLEL 链 + 推 THS").
STEP_ORDER = [
    "refresh",
    "parallel",
    "prob_head",
    "deliver_parallel",
    "drift_parallel",
    "ths_push",
    "ths_flush_guard",
]

# 主线 _DEPENDS 只写了 refresh→parallel→{prob_head, deliver_parallel}; 补三条--
# 清单没出就别推 THS (推的是旧数据), parallel 没成就别跑漂移监控 (比的是旧 run_dir),
# 没推成就不用校验推送.
_DEPENDS_LOCAL = {
    **_DEPENDS,
    "drift_parallel": "parallel",
    "ths_push": "deliver_parallel",
    "ths_flush_guard": "ths_push",
}

# 冲突等待: 23:30 撞上手动重活时, 主链是等 2h 再判; 新链照做 -- 直接 skipped
# 等于当晚没有 PARALLEL 预测也没有 THS 推送 (2026-09-28 审查指出).
_WAIT_TICK_S = 5 * 60
_WAIT_MAX_TICKS = 24  # 5min x 24 = 最多等 2h


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"parallel_automation_{tag}.log")


def _state_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"parallel_automation_{tag}.state.json")


def _write_state(tag: str, status: str, **extra) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    payload = {
        "tag": tag,
        "chain": "parallel",
        "status": status,
        "ts": time.time(),
        **extra,
    }
    with open(_state_path(tag), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _read_state(tag: str) -> dict | None:
    try:
        with open(_state_path(tag), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _emit(fh, msg: str) -> None:
    print(msg, flush=True)
    print(msg, file=fh, flush=True)


def _step_argv(step: str, tag: str) -> list[str]:
    return [PY, "-u"] + [a.replace("{tag}", tag) for a in _STEPS[step]]


def _guard_verdict(tag: str) -> tuple[str, str, list[dict]] | None:
    """启动守卫. 返回 (code, detail, conflicts) 或 None=放行.

    复用主链的 skip_reason: 活进程 > 今日已 ok > 清单已交付. deliverable_exists
    传 False -- parallel 清单文件名不固定, 且 state=ok 已是充分的幂等信号.
    """
    conflicts = find_conflicts(CHAIN_SENTINELS)
    verdict = skip_reason(
        conflicts, today_state=_read_state(tag), deliverable_exists=False
    )
    if verdict is None:
        return None
    code, detail = verdict
    return code, detail, conflicts


def _wait_clear(fh) -> bool:
    """等重活清场 (每 _WAIT_TICK_S 查一次, 最多 _WAIT_MAX_TICKS 次).

    返回 True=可以跑, False=等满仍冲突. 只处理"有人正在跑", 不处理
    state_ok_today/already_delivered -- 那两种等多久都不会变.
    """
    for i in range(1, _WAIT_MAX_TICKS + 1):
        time.sleep(_WAIT_TICK_S)
        conflicts = find_conflicts(CHAIN_SENTINELS)
        if not conflicts:
            _emit(fh, f"[{_stamp()}] guard 第 {i} 次: 已清场, 开跑")
            return True
        c = conflicts[0]
        _emit(
            fh,
            f"[{_stamp()}] guard 第 {i} 次: {c['sentinel']} "
            f"(PID {c['pid']}) 仍在跑, 继续等",
        )
    _emit(fh, f"[{_stamp()}] guard 等满 {_WAIT_MAX_TICKS} 轮仍冲突, 放弃")
    return False


def _panel_stale_gate(tag: str, fh) -> int | None:
    """V3 面板新鲜度闸 (与主链同口径). 返回退出码或 None=放行.

    23:30 链依赖 19:15 fetch 落盘; 面板陈旧说明 fetch 没跑成, 此时跑出的预测是
    旧数据, 必须大声失败而不是静默交付.
    """
    from app.pipeline1 import freshness_guard
    from config.settings import PANEL_V3_PATH

    pmax = freshness_guard.file_max_date(str(PANEL_V3_PATH), "date")
    if pmax is None:
        _emit(
            fh,
            "[FATAL] V3 面板不可读 (file_max_date=None), 链终止. (看 D:/AMINQT/PARQUET)",
        )
        _write_state(tag, "failed", reason="panel_unreadable")
        return 2
    allow, reason = freshness_guard.panel_stale_gate(
        pmax, _dt.date.today(), freshness_guard.load_trade_cal()
    )
    if not allow:
        _emit(fh, f"[FATAL] {reason}, 疑似 fetch 未跑. 链终止. (看 _daily_fetch.py)")
        _write_state(tag, "failed", reason="panel_stale")
        return 2
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="PARALLEL-only 每日自动化链 (23:30)")
    ap.add_argument("--tag", default=None, help="清单 tag YYYYMMDD (缺省=今天)")
    ap.add_argument("--dry-run", action="store_true", help="只打印步骤序列, 不执行")
    ap.add_argument(
        "--force", action="store_true", help="跳过启动守卫 (活进程/今日已完成)"
    )
    args = ap.parse_args()

    tag = args.tag or _dt.date.today().strftime("%Y%m%d")

    # 长链期间不让机器休眠 (08-27 事故: 12:52 休眠吞掉整链)
    _prevent_sleep()

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_log_path(tag), "a", encoding="utf-8") as fh:
        _emit(fh, f"[{_stamp()}] PARALLEL 链启动 tag={tag} 步骤={STEP_ORDER}")

        if args.dry_run:
            for step in STEP_ORDER:
                _emit(fh, f"  [dry] {step}: {' '.join(_step_argv(step, tag))}")
            _write_state(tag, "ok", reason="dry_run")
            return 0

        if not args.force:
            hit = _guard_verdict(tag)
            if hit is not None:
                code, detail, conflicts = hit
                for c in conflicts:
                    _emit(
                        fh,
                        f"[{_stamp()}] guard 冲突: {c['sentinel']} "
                        f"(PID {c['pid']}) {c['cmdline']}",
                    )
                _emit(fh, f"[{_stamp()}] guard 命中 ({code}): {detail}")
                # state_ok_today / already_delivered 等不来; 只有活进程值得等.
                if code != "live_process" or not _wait_clear(fh):
                    _emit(fh, f"[{_stamp()}] guard 放弃启动 ({code})")
                    _write_state(tag, "skipped", reason=code)
                    return 0

        rc_gate = _panel_stale_gate(tag, fh)
        if rc_gate is not None:
            return rc_gate

        _write_state(tag, "running")
        current = {"step": None}

        def _on_sigint(_signum, _frame):
            _write_state(tag, "interrupted", step=current["step"])
            raise SystemExit(130)

        signal.signal(signal.SIGINT, _on_sigint)

        failures: list[str] = []
        for step in STEP_ORDER:
            dep = _DEPENDS_LOCAL.get(step)
            if dep in failures:
                _emit(fh, f"[{_stamp()} skip] {step} (依赖 {dep} 失败, 不跑)")
                continue
            current["step"] = step
            argv = _step_argv(step, tag)
            _emit(fh, f"[{_stamp()} start] {step}: {' '.join(argv)}")
            t0 = time.time()
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            rc, timed_out = _run_step_with_watchdog(
                argv, fh, env, _STEP_TIMEOUT_S[step]
            )
            if timed_out:
                _emit(
                    fh,
                    f"[{_stamp()} TIMEOUT] {step} 超 {_STEP_TIMEOUT_S[step]}s, "
                    f"已杀进程树",
                )
                rc = 124
            dt = time.time() - t0
            if _is_interrupt_rc(rc):
                _emit(fh, f"[{_stamp()} interrupt] {step} 被中断 (rc={rc})")
                _write_state(tag, "interrupted", step=step, rc=rc)
                return 130
            _emit(
                fh,
                f"[{_stamp()} {'ok' if rc == 0 else 'FAIL'}] {step} rc={rc} ({dt:.0f}s)",
            )
            if rc != 0:
                failures.append(step)

        _emit(fh, f"[done] 失败步骤={failures or '无'}")
        _write_state(tag, "ok" if not failures else "failed", failed_steps=failures)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
