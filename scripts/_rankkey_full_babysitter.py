"""_rankkey_full_babysitter.py — 等今晚自动化链跑完后自动发射 125d 排名键扫描 (2026-09-01).

背景: 125d 多 seed 排名键扫描 (~3h 重活) 不能与自动化链并发 (RAM 串行铁律;
链的 2h 守候后强启会撞车 → 08-17/08-24 页交换/OOM 事故同型). 链今晚必跑
(tag 20260902 state=failed 允许重试), 故扫描排链后发射. 本脚本以独立进程运行
(Start-Process), 会话退出不影响.

发射条件 (每 POLL_S 秒轮询, 先到先发):
  A. 链完成: logs/daily_automation_*.state.json 在今晚 23:00 后重写 且 status
     终态 (ok/failed/interrupted/cancelled/skipped) 且 CHAIN_SENTINELS 无冲突.
  B. 兜底: --fallback-mins N (默认次日 02:00) 仍无今晚 state 重写 (链未触发/
     被跳过) 且连续 3 次无冲突 → 窗口空闲, 直接发射.
发射后: sweep 自身 find_conflicts 是末道闸; rc=2 (撞车) 退避 10 min 重试 ×3;
其他非零 rc → 大声记日志退出 (fail-fast, 检查点 data/_diag_rankkey_wf_*_e125
支持手动续跑). 硬超时: 次日 08:00 未发射 → exit 1.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts._run_guard import CHAIN_SENTINELS, HEAVY_SENTINELS, find_conflicts

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

REPO = Path(__file__).resolve().parent.parent
LOGS = REPO / "logs"
SWEEP = REPO / "scripts" / "_rankkey_multiseed_sweep.py"
SWEEP_LOG = LOGS / "_rankkey_multiseed_full.log"
SMOKE2_LOG = LOGS / "_rankkey_smoke2.log"
POLL_S = 300
LAUNCH_RETRY = 3
RETRY_BACKOFF_S = 600
FALLBACK_HOUR = 2  # 次日 02:00 链仍无终态 state → 视为当晚空闲
HARD_HOUR = 8  # 次日 08:00 未发射 → 放弃
TERMINAL = {"ok", "failed", "interrupted", "cancelled", "skipped"}


def log(msg: str) -> None:
    logger.info("[%s] %s", f"{datetime.now():%m-%d %H:%M:%S}", msg)


def _state_fresh_terminal(tonight: datetime) -> str | None:
    """今晚 23:00 后重写且 status 终态的 state 文件名; 无则 None."""
    for st in LOGS.glob("daily_automation_*.state.json"):
        if datetime.fromtimestamp(st.stat().st_mtime) < tonight:
            continue
        try:
            import json

            data = json.loads(st.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("status") in TERMINAL:
            return f"{st.name}(status={data.get('status')})"
    return None


def _smoke2_state() -> str:
    """冒烟终态闸: "done" | "failed" | "pending". 全量与冒烟同一代码路径,
    冒烟 Traceback → 全量 3h 必同样死在半路, 不发射 (WORM 检查点设计不变)."""
    try:
        txt = SMOKE2_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "done"  # 无冒烟日志 (历史清理过) → 不拦
    if "=== DONE ===" in txt:
        return "done"
    if "Traceback" in txt:
        return "failed"
    return "pending"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fallback-mins",
        type=int,
        default=None,
        help="兜底发射: 启动 N 分钟后链仍无今晚终态 state 即发射 "
        "(默认: 次日 02:00 内置档)",
    )
    args = ap.parse_args()

    now = datetime.now()
    tonight = now.replace(hour=23, minute=0, second=0, microsecond=0)
    tomorrow = tonight + timedelta(days=1)
    fallback_t = (
        now + timedelta(minutes=args.fallback_mins)
        if args.fallback_mins is not None
        else tomorrow.replace(hour=FALLBACK_HOUR)
    )
    hard_t = tomorrow.replace(hour=HARD_HOUR)
    log(
        f"babysitter 启动 pid={os.getpid()} sweep={SWEEP.name} "
        f"发射主路径=今晚{tonight:%H:%M}后链终态, 兜底={fallback_t:%m-%d %H:%M}, "
        f"硬超时={hard_t:%m-%d %H:%M}"
    )
    quiet_streak = 0
    launched = False
    while not launched:
        now = datetime.now()
        if now >= hard_t:
            log("HARD TIMEOUT 次日 08:00 未发射, 放弃")
            return 1
        done = _state_fresh_terminal(tonight)
        conflicts = find_conflicts(sentinels=CHAIN_SENTINELS)
        quiet = not conflicts
        quiet_streak = quiet_streak + 1 if quiet else 0
        go = (done is not None and quiet) or (now >= fallback_t and quiet_streak >= 3)
        if go:
            why = f"链终态 {done}" if done else f"兜底 (quiet_streak={quiet_streak})"
            log(f"发射条件满足: {why}, 启动 sweep")
            break
        if int(now.timestamp()) % 1800 < POLL_S:
            log(
                f"等待: state={'无' if done is None else done} "
                f"conflicts={len(conflicts)} quiet_streak={quiet_streak}"
            )
        time.sleep(POLL_S)

    rc = 0
    # 冒烟终态闸 (独立于发射重试, 不占 attempt): 全量与冒烟同一代码路径
    while True:
        sm = _smoke2_state()
        if sm == "done":
            log("smoke2 终态 ok, 冒烟闸放行")
            break
        if sm == "failed":
            log("smoke2 冒烟失败 (Traceback) — 全量同代码路径, 中止发射")
            return 5
        if not find_conflicts(sentinels=HEAVY_SENTINELS):
            log("smoke2 无终态标记且进程已死 — 异常终止, 中止发射")
            return 5
        if datetime.now() >= hard_t:
            log("HARD TIMEOUT 等待冒烟终态, 放弃")
            return 1
        log(f"smoke2 仍在跑 ({sm}), 等待冒烟终态")
        time.sleep(POLL_S)
    for attempt in range(1, LAUNCH_RETRY + 1):
        hits = find_conflicts(sentinels=CHAIN_SENTINELS)
        if hits:
            log(f"发射前守卫命中 (尝试 {attempt}/{LAUNCH_RETRY}), 退避重试: {hits}")
            time.sleep(RETRY_BACKOFF_S)
            continue
        try:
            fh = open(SWEEP_LOG, "a", encoding="utf-8")
        except OSError as exc:
            log(f"打开日志失败: {SWEEP_LOG.name}: {exc}")
            return 3
        with fh:
            log(f"启动 sweep (尝试 {attempt}): {SWEEP} → {SWEEP_LOG.name}")
            try:
                rc = subprocess.call(
                    [sys.executable, "-u", str(SWEEP)],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd=str(REPO),
                )
            except Exception as exc:
                log(f"sweep 子进程异常 (尝试 {attempt}/{LAUNCH_RETRY}): {exc}")
                return 3
        if rc == 0:
            log("sweep rc=0 完成")
            return 0
        if rc == 2:
            log(f"sweep rc=2 守卫拦截 (尝试 {attempt}/{LAUNCH_RETRY}), 退避重试")
            time.sleep(RETRY_BACKOFF_S)
            continue
        log(
            f"SWEEP FAILED rc={rc} — fail-fast 不重试; 检查点 _diag_rankkey_wf_*_e125 可续跑"
        )
        return 3
    log(f"LAUNCH RETRIES EXHAUSTED rc={rc}")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
