"""重训/预测并发守卫 — 手动任务与每日自动化链互斥 (2026-09-01).

背景: 手动重训/预测 (5-7h, 吃满内存) 与 20:15 自动化链并发 → 页交换卡死
(08-17 legacy 预测卡 7h) / 双建 OOM 整链被杀 (08-24 refresh 双建 16GB OOM).
refresh 已有自己的 PID 锁 (_refresh_parallel_checkpoints.py), 其余重训/预测
脚本之间、手动 vs 自动之间零互斥.

两层防护:
  链级 (run_daily_automation.py 启动): 活进程 / 今日链已 ok / 今日清单已交付
    三闸任一命中 → 不启动; 活进程冲突专属 2h 守候循环, 冲突清空且清单仍缺才启动
    (--force 绕过全部闸).
  脚本级 (_retrain_legacy_full.py / _gen_legacy_list.py main 开头): 只查活进程,
    防反向冲突 (手动启动撞上正在跑的链). 哨兵不含 run_daily_automation.py —
    链作为父进程存活是合法场景, 否则子步骤会被自己的链误杀.
"""

from __future__ import annotations

import os

# 重训+预测重活哨兵 — 进程 cmdline 含任一子串即视为冲突 (与自动化重活步骤对齐;
# deliver/drift/shadow/cyq 等轻量步骤不在内).
HEAVY_SENTINELS = (
    "_retrain_legacy_full.py",
    "_gen_legacy_list.py",
    "_refresh_parallel_checkpoints.py",
    "app.pipeline_parallel.runner",
    "_train_legacy_prob_head.py",
    "_train_parallel_prob_head.py",
    "_rankkey_multiseed_sweep.py",
    "_diag_q90_slot_replay.py",
    "_diag_parallel_fullpool_replay.py",
    "_diag_vp_family_ab.py",
    "_diag_rank_head_replay.py",
    "_diag_q50_ensemble_ab.py",
    "_dual_pkg_finaltop_compare.py",
    "_diag_preinfo_audit.py",
    "_diag_time_decay_ab.py",
    "_diag_fakeleg_event.py",
    "_diag_prerise_detector.py",
    "_diag_volflush_study.py",
    "_diag_preflush_detector.py",
    "_diag_flushfilter_check.py",
    "_diag_pool2stage_check.py",
    "_diag_parallel_parity_audit.py",
    "_diag_reg_decay_ab.py",
)
ORCHESTRATOR_SENTINEL = "run_daily_automation.py"
# 链级守卫额外把另一条链实例视为冲突; 脚本级不含 (见模块 docstring).
CHAIN_SENTINELS = HEAVY_SENTINELS + (ORCHESTRATOR_SENTINEL,)


def find_conflicts(
    sentinels: tuple[str, ...] = HEAVY_SENTINELS,
    exclude_pids: set[int] | None = None,
    procs: list[dict] | None = None,
) -> list[dict]:
    """返回 cmdline 命中哨兵的存活进程 [{pid, name, cmdline, sentinel}].

    只匹配 python 进程 (name 含 "python"): 编辑器/终端等非 python 进程的 cmdline
    可能恰好含脚本路径 (打开的文件标签), 误命中会永久错杀链. 自身 PID 恒排除
    (哨兵子串总出现在自己的 argv 里). procs 供单测注入伪进程表.
    """
    exclude = {os.getpid(), *(exclude_pids or set())}
    if procs is None:
        import psutil

        procs = [
            {
                "pid": p.info["pid"],
                "name": p.info["name"] or "",
                "cmdline": p.info["cmdline"] or [],
            }
            for p in psutil.process_iter(["pid", "name", "cmdline"])
        ]
    hits: list[dict] = []
    for p in procs:
        pid = p.get("pid")
        if pid is None or pid in exclude:
            continue
        if "python" not in (p.get("name") or "").lower():
            continue
        cmdline = " ".join(p.get("cmdline") or [])
        for s in sentinels:
            if s in cmdline:
                hits.append(
                    {
                        "pid": pid,
                        "name": p.get("name"),
                        "cmdline": cmdline,
                        "sentinel": s,
                    }
                )
                break
    return hits


def skip_reason(
    conflicts: list[dict],
    *,
    today_state: dict | None,
    deliverable_exists: bool,
) -> tuple[str, str] | None:
    """链级三闸判定 (纯函数). 返回 (code, detail) 或 None=放行.

    优先级: 活进程 > 今日链已 ok/已取消 > 今日清单已交付. state 为 failed/
    interrupted/running 时不拦 — 允许失败重试; running 且无活进程 = 上条链死了,
    应继续跑. 今日 ok 但 reason="dry_run" 的不拦 — dry-run 只打印计划 (会写 ok
    终态), 若算"已完成"会让当晚真实链被静默跳过 (2026-09-01).
    """
    if conflicts:
        c = conflicts[0]
        more = f" 等 {len(conflicts)} 个" if len(conflicts) > 1 else ""
        return ("live_process", f"{c['sentinel']} (PID {c['pid']}){more} 在跑")
    if today_state:
        st = today_state.get("status")
        if st == "ok" and today_state.get("reason") != "dry_run":
            return ("state_ok_today", "今日链已完成 (state=ok), 不重复跑")
        if st == "cancelled":
            return (
                "cancelled_today",
                f"今日已取消 (reason={today_state.get('reason')})",
            )
    if deliverable_exists:
        return ("already_delivered", "今日 legacy 清单已交付, 预测已实现")
    return None
