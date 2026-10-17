# -*- coding: utf-8 -*-
"""LEGACY-only 夜链入口 (2026-10-17).

用户令 "RESTORE LEGACY DAILY PIPELINE AND SCHEDULE DAILY RUN IN PROPER TIME
SLOT": 2026-09-28 起 23:30 槽位只跑 PARALLEL, LEGACY 侧 (清单/密度上游/周重训)
随之断供 (最后交付 20260922)。本链恢复 LEGACY 日常管道, 挂 21:30 时间槽
(AMINQT-LegacyAutomation-2130), 与 20:30 GENIOUS / 23:30 PARALLEL 错峰:

  20:30  GENIOUS (纯规则, 实测 ~10-25min 收工)
  21:30  LEGACY  (本链, 实测节奏 ~2.5h; 周五含 retrain ~+2h)
  23:30  PARALLEL (guard 见本链哨兵会等本链退出再跑, 周五 ~02:00 清场)

四模块 (LEGACY/PARALLEL/GENIOUS/密度) 全部在次日 08:00 前落地:
密度 (prob10dens) 吃 legacy 概率头带 + parallel 原始预测列, 属跨链步骤,
挂在 PARALLEL 链尾 (run_parallel_automation.py), 不在本链。

  [cyq]             scripts/_backfill_cyq_panel.py --workers 6  幂等回填
  [sw_history]      scripts/fetch_sw_daily_history.py --incremental
  [freshness]       scripts/_freshness_check.py                 列水位闸 (非阻断)
  [canary]          scripts/_finaltop_canary.py                 backup vs current 回放
  [retrain]         scripts/_retrain_legacy_full.py <tag>       仅周五 (RETRAIN_WEEKDAY)
  [legacy_prob_head] scripts/_train_legacy_prob_head.py         21 交易日到期才重拟
  [legacy]          scripts/_gen_legacy_list.py <tag>           清单生成
  [deliver]         scripts/_deliver_legacy_list.py <tag>       落 STOCK_LIST_DIR
  [drift]           scripts/_monitor_legacy_drift.py            漂移监控 (只读)
  [gate_audit]      scripts/_gate_admission.py --gate all       闸准入日复审

步骤定义 (_STEPS)、超时 (_STEP_TIMEOUT_S)、看门狗杀进程树、防休眠, 全部复用
scripts/run_daily_automation.py 的既有实现 (与 run_parallel_automation.py 同模式)。

产物:
  logs/legacy_automation_<tag>.log         步骤 stdout/stderr (WORM, 追加)
  logs/legacy_automation_<tag>.state.json  running / ok / failed / skipped / interrupted

调用:
  python scripts/run_legacy_automation.py                  # 全链 (缺省今天)
  python scripts/run_legacy_automation.py --dry-run        # 只打印步骤序列
  python scripts/run_legacy_automation.py --tag 20261016   # 补跑指定日
  python scripts/run_legacy_automation.py --force-retrain  # 无视星期几强制重训
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
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
    _STEP_TIMEOUT_S,
    _STEPS,
    PY,
    RETRAIN_WEEKDAY,
    _is_interrupt_rc,
    _prevent_sleep,
    _run_step_with_watchdog,
)

LOG_DIR = os.path.join(ROOT, "logs")

# 执行顺序与 run_daily_automation.plan_steps 的 legacy 段一致 (cyq→sw→freshness→
# canary→[retrain 周五]→legacy_prob_head→legacy→deliver); 尾部只保留 legacy 侧
# 监控 (drift/gate_audit)。跨链交付步 (密度/combined/ths_push/final_stocklist)
# 归 PARALLEL 链尾, 不在此重复。
STEP_ORDER = [
    "cyq",
    "sw_history",
    "freshness",
    "canary",
    "legacy_prob_head",
    "legacy",
    "deliver",
    "drift",
    "gate_audit",
]

# 主链 _DEPENDS 只覆盖 parallel 段; 此处补 legacy 侧: 清单生成要概率头 bundle,
# 交付要清单, 两个监控读交付后的清单。retrain 不挂依赖 — 失败 fail-soft 落回
# current 模型继续出单 (run_daily_automation 既有设计)。
_DEPENDS_LOCAL = {
    "legacy": "legacy_prob_head",
    "deliver": "legacy",
    "drift": "deliver",
    "gate_audit": "deliver",
}

# 链成败只看模块交付 (与 run_daily_automation._CRITICAL 的 legacy 段对齐):
# 预检/回放/重训/监控失败记 FAIL 但不判整链失败 (retrain fail-soft 是显式设计,
# canary 自身几乎恒 exit 0, drift/gate_audit 是只读监控)。
_NON_CRITICAL_STEPS = {
    "cyq",
    "sw_history",
    "canary",
    "retrain",
    "drift",
    "gate_audit",
}

# 冲突等待: 21:30 撞上手动重活最多等 2h (23:30 前必须开跑, 否则会挤压
# PARALLEL 链的守候窗口)。与 run_parallel_automation 同节奏。
_WAIT_TICK_S = 5 * 60
_WAIT_MAX_TICKS = 24  # 5min x 24 = 最多等 2h


def _stamp() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"legacy_automation_{tag}.log")


def _state_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"legacy_automation_{tag}.state.json")


def _write_state(tag: str, status: str, **extra) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    payload = {
        "tag": tag,
        "chain": "legacy",
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


def _list_delivered(tag: str) -> bool:
    """legacy 清单当日已交付 → 幂等信号 (与主链 _today_list_delivered 同口径)."""
    from config.settings import STOCK_LIST_DIR

    return bool(
        glob.glob(os.path.join(str(STOCK_LIST_DIR), f"legacy_stocklist_{tag}__*.csv"))
    )


def plan_steps(today: _dt.date, *, force_retrain: bool = False) -> list[str]:
    """STEP_ORDER + 周五 retrain 插入 (与主链 RETRAIN_WEEKDAY 同规则)."""
    steps = list(STEP_ORDER)
    if force_retrain or today.weekday() == RETRAIN_WEEKDAY:
        steps.insert(steps.index("canary") + 1, "retrain")
    return steps


def _guard_verdict(tag: str) -> tuple[str, str, list[dict]] | None:
    conflicts = find_conflicts(CHAIN_SENTINELS)
    verdict = skip_reason(
        conflicts, today_state=_read_state(tag), deliverable_exists=_list_delivered(tag)
    )
    if verdict is None:
        return None
    code, detail = verdict
    return code, detail, conflicts


def _wait_clear(fh) -> bool:
    """等重活清场 (每 5min 查一次, 最多 2h). True=可以跑, False=等满仍冲突."""
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
    """V3 面板新鲜度闸 (与主链/parallel 链同口径). 返回退出码或 None=放行."""
    from app.pipeline1 import freshness_guard
    from config.settings import PANEL_V3_PATH

    pmax = freshness_guard.file_max_date(str(PANEL_V3_PATH), "date")
    if pmax is None:
        _emit(fh, "[FATAL] V3 面板不可读 (file_max_date=None), 链终止. (看 D:/AMINQT/PARQUET)")
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
    ap = argparse.ArgumentParser(description="LEGACY-only 每日自动化链 (21:30)")
    ap.add_argument("--tag", default=None, help="清单 tag YYYYMMDD (缺省=今天)")
    ap.add_argument("--dry-run", action="store_true", help="只打印步骤序列, 不执行")
    ap.add_argument("--force-retrain", action="store_true", help="无视星期几强制重训")
    ap.add_argument(
        "--force", action="store_true", help="跳过启动守卫 (活进程/今日已完成)"
    )
    args = ap.parse_args()

    tag = args.tag or _dt.date.today().strftime("%Y%m%d")
    steps = plan_steps(_dt.date.today(), force_retrain=args.force_retrain)

    # 长链期间不让机器休眠 (08-27 事故: 12:52 休眠吞掉整链)
    _prevent_sleep()

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(_log_path(tag), "a", encoding="utf-8") as fh:
        _emit(fh, f"[{_stamp()}] LEGACY 链启动 tag={tag} 步骤={steps}")

        if args.dry_run:
            for step in steps:
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
        for step in steps:
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
            if rc != 0 and step in _NON_CRITICAL_STEPS:
                _emit(fh, f"[{_stamp()} note] {step} 非关键失败, 不判链失败")
            elif rc != 0:
                failures.append(step)

        _emit(fh, f"[done] 失败步骤={failures or '无'}")
        _write_state(tag, "ok" if not failures else "failed", failed_steps=failures)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
