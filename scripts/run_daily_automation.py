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
  [canary]   scripts/_finaltop_canary.py <tag>        晋升后 canary 回放 (backup vs current,
                                            非关键证据步恒 exit 0; 决定性坏签只留证,
                                            回退需人工 --revert)
  [retrain]  scripts/_retrain_legacy_full.py <tag>     legacy 周频重训 (仅 RETRAIN_WEEKDAY; 重排后当日清单先用现有模型, 新模型自次一清单生效)
  [parallel] python -m app.pipeline_parallel.runner     并行回测 + 短名单 (sniper/fusion/slow_bull)
  [deliver_parallel] scripts/_shortlist_t5_t10.py <tag> 并行短名单交付 STOCK_LIST_DIR
  [ths_push] scripts/_ths_watchlist_push.py <tag>       当日 TOP10 推同花顺自选股 (UI 自动化,
                                            非关键; parallel 跳过时自动回退 legacy 清单)
  [ths_flush_guard] scripts/_ths_flush_guard.py <tag>   当日放量下跌标记→自选股剔除文档 (非关键,
                                            判断只用日频 OHLCV/动量/量能; 09-03)
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
终态判据: logs/daily_automation_<tag>.state.json (running → ok/failed/interrupted/
skipped), 监督方 (scripts/_babysit_daily_automation.py) 见终态即退出, 不再无限等待
耗 token. 每步 rc/耗时 + 全部子进程输出 → logs/daily_automation_<tag>.log (WORM, 不覆盖).

启动守卫 (2026-09-01): 手动重训/预测与链并发 → 页交换卡死 (08-17) / OOM 整链被杀
(08-24). 三闸任一命中不启动: ①活的重训/预测进程 (含另一条链) → 2h 守候循环,
冲突清空且当日清单仍缺才启动; ②今日链 state=ok; ③今日 legacy 清单已交付.
--force 绕过. 守卫逻辑见 scripts/_run_guard.py, 日志与步骤日志同文件.

用法:
  python scripts/run_daily_automation.py                      # 完整跑 (推荐: 定时任务)
  python scripts/run_daily_automation.py --dry-run            # 只打印计划不执行
  python scripts/run_daily_automation.py --skip-checkpoints --skip-retrain --skip-parallel
                                                              # 只跑 legacy 预测+交付 (轻量验证)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
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

# 脚本直跑时 sys.path[0]=scripts/, 补 ROOT 使 scripts.* 包导入在两种模式下都成立
sys.path.insert(0, ROOT)

from config.settings import STOCK_LIST_DIR  # noqa: E402
from scripts._run_guard import (  # noqa: E402
    CHAIN_SENTINELS,
    find_conflicts,
    skip_reason,
)

# 0=Mon .. 6=Sun. 周频重训落在周五晚 (周末分析用周五收盘后最新模型).
RETRAIN_WEEKDAY = 4

# 每步超时兜底 (2026-08-17): 08-17 run1 legacy 预测卡死 7h (与手动重训并发页交换),
# 无超时则僵尸进程占内存影响后续所有步骤. 取值=正常耗时的 4-6 倍 (refresh 08-17 实测
# 70min / legacy 18-30min / retrain 5-7h), 只兜底卡死不误杀慢跑. 超时按步骤记为 rc=124.
_STEP_TIMEOUT_S = {
    "refresh": 3 * 3600,
    "cyq": 40 * 60,
    # sw_history 正常 3-4min (约 400 指数 × 0.15s 延迟 + API 延迟); 09-03 实测限流日
    # 500 指数 ~16min (100 只/100s), 15min 超时被杀 → 上调 30min 留 2x 余量
    "sw_history": 30 * 60,
    # freshness 只读 schema/尾列 (实际约 1-2min); 15min 守 "每步 ≥15min" 下限惯例
    "freshness": 15 * 60,
    "retrain": 12 * 3600,
    "parallel": 4 * 3600,
    # prob_head 半衰期集成后 = 2 板 × len(half_lives) 桡 bundle 训练 (09-03 起 6 次)
    "prob_head": 3 * 3600,
    "legacy_prob_head": 1 * 3600,
    "legacy": 3 * 3600,
    "deliver": 30 * 60,
    # canary 每板回放工具内部超时 5400s, 双板合法最坏 3h; 4h 只兜卡死
    "canary": 4 * 3600,
    "deliver_parallel": 30 * 60,
    "ths_push": 15
    * 60,  # 客户端已开 ~20s; 冷启动拉起+登录最长 ~2.5min, 下限 15min 只兜卡死
    "ths_flush_guard": 10 * 60,  # 面板单日切片+秩计算 ~1min, 下限只兜卡死
    "drift": 30 * 60,
    "drift_parallel": 30 * 60,
    "shadow_xmodule": 15 * 60,
}

# 启动守卫守候循环 (2026-09-01): 活进程冲突时每 2h 复查一次, 最多 3 轮 (6h).
# 上限的硬约束是计划任务 ExecutionTimeLimit=PT16H — 周五最坏 20:15+6h 守候+7h 重训
# 仍留有余量; 更长的守候会让链跑不完被任务限时强杀.
_GUARD_TICK_S = 2 * 3600
_GUARD_MAX_TICKS = 3


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
    "sw_history": ["scripts/fetch_sw_daily_history.py", "--incremental"],
    # 全族特征新鲜度守卫 (2026-09-02): 注册表 config/freshness_registry.yaml, 告警式
    # 恒 exit 0 — 判定逻辑见 scripts/_freshness_check.py 模块 docstring
    "freshness": ["scripts/_freshness_check.py"],
    "retrain": ["scripts/_retrain_legacy_full.py", "{tag}"],
    "parallel": ["-m", "app.pipeline_parallel.runner"],
    "prob_head": ["scripts/_train_parallel_prob_head.py"],
    "legacy_prob_head": ["scripts/_train_legacy_prob_head.py"],
    "legacy": ["scripts/_gen_legacy_list.py", "{tag}"],
    "deliver": ["scripts/_deliver_legacy_list.py", "{tag}"],
    "canary": ["scripts/_finaltop_canary.py"],
    "deliver_parallel": ["scripts/_shortlist_t5_t10.py", "{tag}"],
    "ths_push": ["scripts/_ths_watchlist_push.py", "{tag}"],
    "ths_flush_guard": ["scripts/_ths_flush_guard.py", "{tag}"],
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
    # 申万指数日线增量 (2026-09-02): 面板冻结@07-31 事故 (end 写死 + 无人挂链) 断供
    # dim28 特征族 39 列上游; 恒前置轻量非关键步骤 — 失败只损失当日行业指数特征新鲜度
    steps.append("sw_history")
    # 全族特征新鲜度守卫 (2026-09-02): 四起静默停更事故 (cyq@07-17/sw冻结@07-31/
    # announce_date@08-14/fina列冻结) 后建的系统级闸, 告警式不阻断 — 08-27 零清单
    # 教训: 链对失败一视同仁, 告警绝不能拦交付 (恒 exit 0)
    steps.append("freshness")
    # legacy 并行式概率头: 读面板+特征现场构建 (不依赖 parallel 检查点, 无前置依赖);
    # 自判断新鲜度 (21 交易日重训一次), 未到期开销小 — 放 legacy 预测前 (概率闸依赖 bundle)
    steps.append("legacy_prob_head")
    steps.append("legacy")
    steps.append("deliver")
    if not skip_checkpoints:
        steps.append("refresh")
    # 晋升后 canary (2026-09-02 防坏签): 新 current vs 晋升前 backup 定期重放,
    # 决定性坏签留证 (回退需人工 --revert, 链上不带). 放 retrain 前 — retrain
    # 晋升会覆盖 canary state, 先跑让在窗晋升按自然日推进窗口; state 空/窗口满
    # 时秒过恒 exit 0, 非关键步骤
    steps.append("canary")
    if not skip_retrain and (force_retrain or today.weekday() == RETRAIN_WEEKDAY):
        steps.append("retrain")
    if not skip_parallel:
        steps.append("parallel")
        # 概率头训练自判断新鲜度 (21 交易日重训一次); 仅并行交付启用时才有消费者
        steps.append("prob_head")
        steps.append(
            "deliver_parallel"
        )  # 并行清单交付依赖当日 fresh parallel 重生成, 跳过则同步丢弃
    # 同花顺自选股推送 (2026-09-01): TOP10 主源并行短名单 rank 序, parallel 跳过时
    # collect_codes 自动回退 legacy 清单 — 故放在 parallel 块之外恒执行, 非关键步骤
    steps.append("ths_push")
    # 放量下跌自选股守卫 (2026-09-03): 当日放量下跌标记 (日频 OHLCV/动量/量能) →
    # 自选股剔除 + 当日删除文档; UI 删除待 Del 键流程探针验证后经 --apply 启用, 非关键步骤
    steps.append("ths_flush_guard")
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


# ── 启动守卫 (2026-09-01): 手动重训/预测与晚上链并发 → 页交换卡死 (08-17) /
#    OOM 整链被杀 (08-24). 三闸 + 活进程冲突守候循环, 判定逻辑见 scripts/_run_guard.py ──


def _read_state(tag: str) -> dict | None:
    try:
        with open(_state_path(tag), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _today_list_delivered(tag: str) -> bool:
    """今日 legacy 清单是否已交付 (任一板块) — 守卫"预测已实现"闸."""
    return bool(
        glob.glob(os.path.join(str(STOCK_LIST_DIR), f"legacy_stocklist_{tag}__*.csv"))
    )


def _guard_log(tag: str, msg: str) -> None:
    """守卫日志同时进 stdout 与当日链日志 — 计划任务的 stdout 无人看见, 文件才是真相."""
    print(msg, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(
        os.path.join(LOG_DIR, f"daily_automation_{tag}.log"), "a", encoding="utf-8"
    ) as fh:
        print(msg, file=fh, flush=True)


def _startup_guard_conflicts() -> list[dict]:
    return find_conflicts(CHAIN_SENTINELS)


def _run_startup_guard(tag: str) -> str:
    """启动三闸. 返回 "go" 放行 / "skip" 放弃 / "wait" 进入守候循环."""
    conflicts = _startup_guard_conflicts()
    verdict = skip_reason(
        conflicts,
        today_state=_read_state(tag),
        deliverable_exists=_today_list_delivered(tag),
    )
    if verdict is None:
        return "go"
    code, detail = verdict
    ts = f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S} guard]"
    if code == "live_process":
        for c in conflicts:  # 全量列出留证, 便于排查是谁挡的
            _guard_log(
                tag, f"{ts} 冲突进程: {c['sentinel']} (PID {c['pid']}) {c['cmdline']}"
            )
        _guard_log(tag, f"{ts} 守卫拦截 ({code}): {detail} → 进入守候循环, 每 2h 复查")
        return "wait"
    _guard_log(tag, f"{ts} 守卫跳过 ({code}): {detail}")
    return "skip"


def _wait_for_clearance(
    tag: str,
    *,
    tick_s: int = _GUARD_TICK_S,
    max_ticks: int = _GUARD_MAX_TICKS,
) -> bool:
    """守候循环 (活进程冲突专属): 每 2h 复查 — 今日清单已出 → 不必启动 (False);
    冲突清空且清单仍缺 → 启动 (True); 超过 max_ticks 轮仍冲突 → 放弃 (False).
    """
    for i in range(1, max_ticks + 1):
        try:
            time.sleep(tick_s)
        except KeyboardInterrupt:
            _guard_log(tag, "[guard] 守候中被 Ctrl+C 中断, 放弃")
            return False
        ts = f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S} guard]"
        if _today_list_delivered(tag):
            _guard_log(tag, f"{ts} 第{i}次复查: 今日清单已出, 守候结束不启动")
            return False
        conflicts = _startup_guard_conflicts()
        if not conflicts:
            _guard_log(tag, f"{ts} 第{i}次复查: 冲突清空且当日清单仍缺 → 启动链")
            return True
        c0 = conflicts[0]
        more = f" 等 {len(conflicts)} 个" if len(conflicts) > 1 else ""
        _guard_log(
            tag,
            f"{ts} 第{i}次复查: 仍冲突 ({c0['sentinel']} PID {c0['pid']}{more}) → 继续守候",
        )
    _guard_log(
        tag,
        f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S} guard] 守候 {max_ticks} 轮仍冲突 → "
        f"放弃, 今日不启动 (冲突清空后可手动触发或用 --force)",
    )
    return False


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
    ap.add_argument(
        "--force",
        action="store_true",
        help="绕过启动守卫 (并发进程/今日链已 ok/今日清单已交付三闸)",
    )
    args = ap.parse_args()

    today = _dt.date.today()
    tag = args.tag or today.strftime("%Y%m%d")

    # 守候/链运行期间机器不睡眠 (08-27 事故), 守卫守候循环同样要覆盖
    _prevent_sleep()

    # 启动守卫 (2026-09-01) — dry-run 只看计划不执行, 不受守卫约束
    if not args.dry_run and not args.force:
        verdict = _run_startup_guard(tag)
        if verdict == "skip":
            _write_state(tag, "skipped", reason="guard")
            return 0
        if verdict == "wait":
            # 立刻写 skipped 终态: babysitter 见此即退出, 不陪守候循环空等
            _write_state(tag, "skipped", reason="guard_live_process_waiting")
            if not _wait_for_clearance(tag):
                return 0
            # 复查通过 → 走正常链, state 下方覆盖为 running

    # 运行状态文件 (监督方终态判据): 启动先写 running, 结束/中断覆盖为终态.
    _write_state(tag, "running")
    current_step: str | None = None

    # Ctrl+C/控制台关闭 → 立即写 interrupted 终态并退出, 不留"无标记裸退出"
    # (那会让监督方永远等不到终态而一直耗 token).
    def _on_sigint(_signum, _frame):
        _write_state(tag, "interrupted", step=current_step)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, _on_sigint)

    # 数据新鲜度护栏 (2026-09-02 加固): V3 面板停更 → 拒绝空跑数小时重活.
    # 修复旧闸三漏洞 (此前: except 静默放行 / 自然日阈值 / 阈值硬编码):
    #   1. 读失败 → [FATAL] + state=panel_unreadable, 不再静默放行
    #      (读失败 ≠ 数据新鲜, except-pass 正是旧闸漏洞);
    #   2. 判定改调 freshness_guard.panel_stale_gate: 有交易日历时按交易日 lag 判
    #      — 旧 "(today-pmax).days > 3" 自然日口径下, 周一跑链面板停周五 = 自然日 3
    #      恰好放行, 周二才拦 (滞后 1-2 天); 交易日口径周一 lag=1 正常放行;
    #   3. 阈值常量集中在 freshness_guard (_PANEL_MAX_LAG_TRADING/_NATURAL).
    from app.pipeline1 import freshness_guard
    from config.settings import PANEL_V3_PATH  # 惰性导入, 保持 --dry-run 轻量

    pmax = freshness_guard.file_max_date(str(PANEL_V3_PATH), "date")
    if pmax is None:
        print(
            "[FATAL] V3 面板不可读 (file_max_date=None), 数据可能损坏或被外部移动. "
            "终止, 不跑重活. (先查 D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet)",
            flush=True,
        )
        _write_state(tag, "failed", reason="panel_unreadable")
        return 2
    allow, reason = freshness_guard.panel_stale_gate(
        pmax, today, freshness_guard.load_trade_cal()
    )
    if not allow:
        print(
            f"[FATAL] {reason}, 数据可能未 fetch. 终止, 不跑重活. (先跑 _daily_fetch.py)",
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
