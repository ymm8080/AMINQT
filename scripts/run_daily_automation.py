"""四模块每日自动化编排 (2026-08-06) — 重训 → 预测出清单 → 落盘 → 看板可见.

覆盖"四个模块":
  1. legacy main   — 周频全量重训 (_retrain_legacy_full.py, OOS 过闸才切换 current)
  2. legacy dual   — 同上 (同一脚本双板)
  3. 并行 sniper   — 每日重生成 (app.pipeline_parallel.runner 回测 + 短名单)
  4. 并行 fusion   — 同上 (同一 runner, 内含 slow_bull 一并输出)

步骤按"交付保底优先"编排 (2026-08-27 重排: 当日 refresh 卡死 9h 全链无清单, 故
legacy 预测链 ~35min 前置到一切重活之前, 重活卡死/超时被杀都不影响当日清单落盘;
每步独立子进程隔离执行释放内存, 避免 44GB commit 上限 OOM):
  [cyq]      scripts/_backfill_cyq_panel.py            cyq_panel 增量回填 (legacy 慢牛列)
  [legacy_prob_head] scripts/_train_legacy_prob_head.py legacy 并行式概率头 (21 交易日自判断重训)
  [legacy]   scripts/_gen_legacy_list.py <tag>          legacy 预测出清单 (用最新 current)
  [deliver]  scripts/_deliver_legacy_list.py <tag>      legacy 清单交付 STOCK_LIST_DIR
  [refresh]  scripts/_refresh_parallel_checkpoints.py  并行行集 3y 检查点 (需 19:15 fetch 后)
  [retrain]  scripts/_retrain_legacy_full.py <tag>     legacy 周频重训 (仅 RETRAIN_WEEKDAY; 重排后当日清单先用现有模型, 新模型自次一清单生效)
  [parallel] python -m app.pipeline_parallel.runner     并行回测 + 短名单 (sniper/fusion/slow_bull)
  [deliver_parallel] scripts/_shortlist_t5_t10.py <tag> 并行短名单交付 STOCK_LIST_DIR
  [drift]    scripts/_monitor_legacy_drift.py           幅度漂移监控 (全池 pred vs 实现偏差)
  [drift_parallel] scripts/_monitor_parallel_drift.py   parallel dual 漂移监控 (短名单 vs 检查点标签)
  [shadow_xmodule] scripts/_shadow_xmodule_blend.py     跨模块影子排名 (legacy×parallel 合池混排, 只记录不交付)

"推送看板" = 各步落盘到看板只读目录, 看板渲染时自动展示:
  模型 → models/pipeline1/current_meta.json + *.pkl (档案页·模型档案)
  回测 → BACKTEST_RESULT_DIR/<ts>/ (档案页·回测历史, 含 hv 胜率图)
  清单 → STOCK_LIST_DIR (档案页·落盘清单) + PredictionDB (每日预测)

失败策略 (失败要大声): refresh 失败 → 跳过 parallel (无新鲜行集); retrain 失败 →
继续当日清单 (沿用现有模型); parallel 失败 → 跳过 deliver_parallel (否则交付旧 run_dir);
任一关键步骤 (legacy/deliver/deliver_parallel) 失败 → 非零退出.
中断策略 (08-21 事故): 任何步骤返回 0xC013A (STATUS_CONTROL_C_EXIT, 控制台
Ctrl+C/进程组被杀) → 立即终止整条链, 不启动后续重活步骤; 收到 SIGINT 同理.
终态判据: logs/daily_automation_<tag>.state.json (running → ok/failed/interrupted),
监督方 (scripts/_babysit_daily_automation.py) 见终态即退出, 不再无限等待耗 token.
每步 rc/耗时 + 全部子进程输出 → logs/daily_automation_<tag>.log (WORM, 不覆盖).

用法:
  python scripts/run_daily_automation.py                      # 完整跑 (推荐: 定时任务)
  python scripts/run_daily_automation.py --dry-run            # 只打印计划不执行
  python scripts/run_daily_automation.py --skip-checkpoints --skip-retrain --skip-parallel
                                                              # 只跑 legacy 预测+交付 (轻量验证)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG_DIR = os.path.join(ROOT, "logs")

# 0=Mon .. 6=Sun. 周频重训落在周五晚 (周末分析用周五收盘后最新模型).
RETRAIN_WEEKDAY = 4

# 每步超时兜底 (2026-08-17): 08-17 run1 legacy 预测卡死 7h (与手动重训并发页交换),
# 无超时则僵尸进程占内存影响后续所有步骤. 取值=正常耗时的 4-6 倍 (refresh 08-17 实测
# 70min / legacy 18-30min / retrain 5-7h), 只兜底卡死不误杀慢跑. 超时按步骤记为 rc=124.
_STEP_TIMEOUT_S = {
    "refresh": 3 * 3600,
    "cyq": 40 * 60,
    "retrain": 12 * 3600,
    "parallel": 4 * 3600,
    "prob_head": 1 * 3600,
    "legacy_prob_head": 1 * 3600,
    "legacy": 3 * 3600,
    "deliver": 30 * 60,
    "deliver_parallel": 30 * 60,
    "drift": 30 * 60,
    "drift_parallel": 30 * 60,
    "shadow_xmodule": 15 * 60,
}


def _kill_tree(pid: int) -> None:
    """整树强杀 — 含步骤脚本派生的 worker 孙进程 (它们继承 stdout 管道,
    只杀直接子进程会漏).
    Windows: taskkill /T /F; POSIX: kill -9 -- -pgid (杀进程组).
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                timeout=30,
            )
        except Exception:  # noqa: BLE001 — 看门狗杀不掉只能放弃, 不影响主流程
            pass
    else:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # 进程已退出或无权限; 降级杀单进程
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass


def _run_step_with_watchdog(
    argv: list[str], fh, env: dict, timeout_s: int
) -> tuple[int, bool]:
    """Popen + 外部看门狗线程, 超时 taskkill 整树强杀. 返回 (rc, timed_out).

    不再用 subprocess.run(timeout=): 08-27 事故中 refresh 爬行 8h+, 其 3h 内部
    超时始终未触发 (日志无 TIMEOUT 记录, 机器全程未休眠), 内部计时器不可信;
    看门狗线程 sleep 到点后 poll + 整树强杀, 与主线程等待互为冗余.
    """
    # Windows: CREATE_NEW_PROCESS_GROUP 使其成为进程组头, taskkill /T 可杀整树.
    # POSIX: start_new_session=True 跑在新会话/进程组, os.killpg 整树强杀不影响父进程.
    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, **popen_kwargs
    )
    state = {"timed_out": False}

    def _watch() -> None:
        time.sleep(timeout_s)
        if proc.poll() is None:
            state["timed_out"] = True
            _kill_tree(proc.pid)

    threading.Thread(target=_watch, daemon=True, name="step-watchdog").start()
    rc = proc.wait()
    return rc, state["timed_out"]


def _prevent_sleep() -> None:
    """链运行期间请求系统不睡眠 (08-27: 12:52 电池睡眠恰好落在链运行中)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadExecutionState(
                0x80000000 | 0x00000001  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            )
        except Exception:  # noqa: BLE001 — 拿不到也不拦链启动
            pass


# (步骤名, argv) — argv 不含解释器, run_step 负责拼 [PY, "-u", ...]
_STEPS = {
    "refresh": ["scripts/_refresh_parallel_checkpoints.py"],
    "cyq": [
        "scripts/_backfill_cyq_panel.py",
        "--workers",
        "6",
    ],  # cyq_panel 增量 (2026-08-19)
    "retrain": ["scripts/_retrain_legacy_full.py", "{tag}"],
    "parallel": ["-m", "app.pipeline_parallel.runner"],
    "prob_head": ["scripts/_train_parallel_prob_head.py"],
    "legacy_prob_head": ["scripts/_train_legacy_prob_head.py"],
    "legacy": ["scripts/_gen_legacy_list.py", "{tag}"],
    "deliver": ["scripts/_deliver_legacy_list.py", "{tag}"],
    "deliver_parallel": ["scripts/_shortlist_t5_t10.py", "{tag}"],
    "drift": ["scripts/_monitor_legacy_drift.py"],
    "drift_parallel": ["scripts/_monitor_parallel_drift.py"],
    "shadow_xmodule": ["scripts/_shadow_xmodule_blend.py"],
}
# refresh 失败后应跳过的后续步骤 (parallel 需要新鲜检查点);
# deliver_parallel 需要当日 fresh parallel run_dir (短名单), 否则会交付旧 run_dir 脏数据;
# prob_head 读 parallel 检查点 (面板), 需当日 fresh 面板.
_DEPENDS = {
    "parallel": "refresh",
    "prob_head": "parallel",
    "deliver_parallel": "parallel",
}
# 关键步骤: 失败 → 整个任务非零退出 (看板当日清单缺失)
_CRITICAL = {"legacy", "deliver", "deliver_parallel"}


def plan_steps(
    today: _dt.date,
    *,
    skip_checkpoints: bool = False,
    skip_retrain: bool = False,
    skip_parallel: bool = False,
    force_retrain: bool = False,
) -> list[str]:
    """按星期 + skip 标志选出当日步骤序列 (纯函数, 可单测)."""
    steps: list[str] = []
    # [2026-08-27] 交付保底优先: legacy 预测链 (cyq→lph→legacy→deliver, ~35min) 前置到
    # 一切重活 (refresh/retrain) 之前 — 08-27 事故 refresh 卡死 9h, 全链一条清单都没出.
    # 代价: 重训日当日清单先用现有模型, 新模型自次一清单生效.
    # cyq_panel 增量回填 (2026-08-19): 读 V3 面板补 cache 缺失日期, 非关键步骤 —
    # 失败只损失当日 pct_70_con (慢牛 0.05 权重列跳过), 清单不受影响; 恒前置 (轻量)
    steps.append("cyq")
    # legacy 并行式概率头: 读面板+特征现场构建 (不依赖 parallel 检查点, 无前置依赖);
    # 自判断新鲜度 (21 交易日重训一次), 未到期开销小 — 放 legacy 预测前 (概率闸依赖 bundle)
    steps.append("legacy_prob_head")
    steps.append("legacy")
    steps.append("deliver")
    if not skip_checkpoints:
        steps.append("refresh")
    if not skip_retrain and (force_retrain or today.weekday() == RETRAIN_WEEKDAY):
        steps.append("retrain")
    if not skip_parallel:
        steps.append("parallel")
        # 概率头训练自判断新鲜度 (21 交易日重训一次); 仅并行交付启用时才有消费者
        steps.append("prob_head")
        steps.append(
            "deliver_parallel"
        )  # 并行清单交付依赖当日 fresh parallel 重生成, 跳过则同步丢弃
    steps.append("drift")  # 幅度漂移监控 (读历史 candidates, 非关键步骤)
    steps.append(
        "drift_parallel"
    )  # parallel dual 漂移监控 (读历史短名单+检查点, 非关键步骤)
    steps.append(
        "shadow_xmodule"
    )  # 跨模块影子排名 (读两侧已交付清单纯记录, 非关键步骤, 2026-08-26)
    return steps


def _log_fh(tag: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    return open(
        os.path.join(LOG_DIR, f"daily_automation_{tag}.log"), "a", encoding="utf-8"
    )


# 0xC000013A STATUS_CONTROL_C_EXIT — 控制台 Ctrl+C / 进程组被杀. 此类中断 ≠ 普通
# 步骤失败: 必须立即终止整条链, 不得继续启动下一个重活步骤 (08-21 事故: cyq 被
# Ctrl+C 杀后仍启动 6h retrain; 监督方也靠终态 state 文件判定停止, 否则永远等待).
_INTERRUPT_RC = 3221225786


def _state_path(tag: str) -> str:
    return os.path.join(LOG_DIR, f"daily_automation_{tag}.state.json")


def _write_state(tag: str, status: str, **extra) -> None:
    """写运行状态文件 — 监督方 (babysitter) 的终态判据.

    status: running → 启动; ok/failed/interrupted → 终态 (监督方见此即退出,
    不再无限等待耗 token). 同一 tag 重跑时覆盖 (该 tag 当前运行的真实状态).
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    payload = {"tag": tag, "status": status, "ts": time.time(), **extra}
    with open(_state_path(tag), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)


def _is_interrupt_rc(rc: int) -> bool:
    """0xC013A STATUS_CONTROL_C_EXIT — 控制台中断/进程组被杀, 须终止整条链."""
    return rc == _INTERRUPT_RC


def _exit_status(failures: list[str]) -> str:
    """失败步骤列表 → 终态 status."""
    return "ok" if not failures else "failed"


def main() -> int:
    ap = argparse.ArgumentParser(description="四模块每日自动化")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    ap.add_argument(
        "--skip-checkpoints", action="store_true", help="跳过并行检查点刷新"
    )
    ap.add_argument("--skip-retrain", action="store_true", help="跳过 legacy 周频重训")
    ap.add_argument(
        "--force-retrain", action="store_true", help="强制 legacy 周频重训 (不限于周五)"
    )
    ap.add_argument("--skip-parallel", action="store_true", help="跳过并行系统重生成")
    ap.add_argument("--tag", default=None, help="清单交易日 YYYYMMDD (默认今天)")
    args = ap.parse_args()

    today = _dt.date.today()
    tag = args.tag or today.strftime("%Y%m%d")

    # 运行状态文件 (监督方终态判据): 启动先写 running, 结束/中断覆盖为终态.
    _write_state(tag, "running")
    _prevent_sleep()
    current_step: str | None = None

    # Ctrl+C/控制台关闭 → 立即写 interrupted 终态并退出, 不留"无标记裸退出"
    # (那会让监督方永远等不到终态而一直耗 token).
    def _on_sigint(_signum, _frame):
        _write_state(tag, "interrupted", step=current_step)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    # 数据新鲜度护栏: V3 面板超过 3 个自然日未更新 → 拒绝空跑数小时重活 (边界校验).
    try:
        import pandas as pd  # 惰性导入, 保持 --dry-run 轻量

        from config.settings import PANEL_V3_PATH

        pmax = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].max().date()
    except Exception:
        pmax = None
    if pmax is not None and (today - pmax).days > 3:
        print(
            f"[FATAL] V3 面板最新日期 {pmax} 早于今天 3 天以上, 数据可能未 fetch. "
            f"终止, 不跑重活. (先跑 _daily_fetch.py)",
            flush=True,
        )
        _write_state(tag, "failed", reason="panel_stale")
        return 2

    steps = plan_steps(
        today,
        skip_checkpoints=args.skip_checkpoints,
        skip_retrain=args.skip_retrain,
        skip_parallel=args.skip_parallel,
        force_retrain=args.force_retrain,
    )
    print(
        f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] 四模块自动化 tag={tag} "
        f"星期={today.strftime('%A')} 步骤={steps}",
        flush=True,
    )
    if args.dry_run:
        for s in steps:
            print(
                f"  [dry] {s}: python {' '.join(_STEPS[s])}".replace("{tag}", tag),
                flush=True,
            )
        _write_state(tag, "ok", reason="dry_run")
        return 0

    failures: list[str] = []
    with _log_fh(tag) as fh:
        print(
            f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] 四模块自动化 tag={tag} "
            f"星期={today.strftime('%A')} 步骤={steps}",
            file=fh,
            flush=True,
        )
        for step in steps:
            dep = _DEPENDS.get(step)
            if dep in failures:
                print(f"[skip] {step} (前置 {dep} 失败, 无新鲜输入)", flush=True)
                print(f"[skip] {step} (前置 {dep} 失败)", file=fh, flush=True)
                continue
            current_step = step
            argv = [PY, "-u"] + [a.replace("{tag}", tag) for a in _STEPS[step]]
            print(
                f"[{_dt.datetime.now():%H:%M:%S} start] {step}: {' '.join(argv)}",
                flush=True,
            )
            print(
                f"[{_dt.datetime.now():%H:%M:%S} start] {step}: {' '.join(argv)}",
                file=fh,
                flush=True,
            )
            t0 = time.time()
            # 统一子进程 stdout 为 UTF-8: 有的脚本 reconfigure 有的不, 混合编码会污染日志文件.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            rc, timed_out = _run_step_with_watchdog(
                argv, fh, env, _STEP_TIMEOUT_S[step]
            )
            if timed_out:
                msg = (
                    f"[{_dt.datetime.now():%H:%M:%S} TIMEOUT] {step} 超过 "
                    f"{_STEP_TIMEOUT_S[step]}s 未完成, 已终止 (疑似卡死, 见上日志)"
                )
                print(msg, flush=True)
                print(msg, file=fh, flush=True)
                rc = 124
            dt = time.time() - t0
            if _is_interrupt_rc(rc):
                msg = (
                    f"[{_dt.datetime.now():%H:%M:%S} interrupt] {step} 被控制台中断 "
                    f"(rc={rc}=0xC013A), 终止整条链, 不启动后续重活步骤"
                )
                print(msg, flush=True)
                print(msg, file=fh, flush=True)
                _write_state(tag, "interrupted", step=step, rc=rc)
                return 130
            ok = rc == 0
            status = "ok" if ok else "FAIL"
            print(
                f"[{_dt.datetime.now():%H:%M:%S} {status}] {step} rc={rc} ({dt:.0f}s)",
                flush=True,
            )
            print(
                f"[{_dt.datetime.now():%H:%M:%S} {status}] {step} rc={rc} ({dt:.0f}s)",
                file=fh,
                flush=True,
            )
            if not ok:
                failures.append(step)

        print(
            f"[done] 失败步骤={failures or '无'} → "
            f"{'非零退出 (看板当日清单缺失)' if failures else '全部成功'}",
            flush=True,
        )
        print(
            f"[done] 失败步骤={failures or '无'} → "
            f"{'非零退出 (看板当日清单缺失)' if failures else '全部成功'}",
            file=fh,
            flush=True,
        )
        if failures:
            print(
                f"[hint] 看日志 logs/daily_automation_{tag}.log 定位失败步骤; "
                f"重跑可用 --skip-* 跳过已成功步骤",
                flush=True,
            )
        # 终态 state 文件 — 监督方 (babysitter) 见 ok/failed 即退出
        _write_state(tag, _exit_status(failures), failed_steps=failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
