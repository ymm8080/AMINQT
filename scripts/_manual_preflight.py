"""_manual_preflight.py — 手工 pipeline 数据前置 (2026-09-08).

用户指令 "手动 PIPELINE 需要有所有功能": 链 (run_daily_automation.plan_steps) 在
重训/预测前恒跑三个数据前置 — cyq 回填 / sw_history 增量 / freshness 守卫 —
手工跑重训/预测/清单此前不跑它们 (09-08 实发: 手工重出清单时 cyq_panel 缺当日,
派发闸只能看 T-1 获利盘). 本模块把三前置打进手工入口 main 开头:
train_predict_main / _retrain_legacy_full / _gen_legacy_list / run_daily.

argv/超时复用 run_daily_automation 的 _STEPS/_STEP_TIMEOUT_S (单一真相源, 链改
步骤名/超时这里自动跟随). 语义与链一致: 三步全非关键 — 单步失败大声告警不拦
主流程 (cyq→派发闸 fail-open, sw→行业特征滞后一日, freshness→告警式恒 exit 0).
被链调起时 (父进程=run_daily_automation) 自动跳过 — 链在前面已跑过, 不重复.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_daily_automation import (  # noqa: E402
    _STEP_TIMEOUT_S,
    _STEPS,
    _run_step_with_watchdog,
)

LOG_DIR = os.path.join(ROOT, "logs")
# 链内恒前置的数据步骤 (plan_steps 顺序: cyq → sw_history → freshness)
PREFLIGHT_STEPS = ("cyq", "sw_history", "freshness")


def _under_chain() -> bool:
    """父进程是否为自动化链 — 链已跑过数据前置, 子步骤内无需重复."""
    try:
        import psutil

        parent = psutil.Process(os.getppid())
        return "run_daily_automation.py" in " ".join(parent.cmdline() or [])
    except Exception:  # noqa: BLE001 — 判不出当手工跑 (宁重复勿跳过)
        return False


def run_preflight(caller: str) -> list[tuple[str, int]] | None:
    """跑数据前置三步; 返回 [(step, rc)] (链内跳过 → None).

    任一步失败/超时只告警不抛错 — 与链的非关键步骤语义一致.
    """
    if _under_chain():
        print(
            f"[preflight:{caller}] 运行于自动化链内, 链已跑数据前置 — 跳过",
            flush=True,
        )
        return None
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"manual_preflight_{time.strftime('%Y%m%d_%H%M%S')}.log")
    results: list[tuple[str, int]] = []
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    print(
        f"[preflight:{caller}] 数据前置: {' → '.join(PREFLIGHT_STEPS)} "
        f"(log={log_path})",
        flush=True,
    )
    with open(log_path, "a", encoding="utf-8") as fh:
        for step in PREFLIGHT_STEPS:
            argv = [sys.executable, "-u", *_STEPS[step]]
            t0 = time.time()
            rc, timed_out = _run_step_with_watchdog(
                argv, fh, env, _STEP_TIMEOUT_S[step]
            )
            if timed_out:
                rc = 124
            results.append((step, rc))
            status = "ok" if rc == 0 else f"FAIL rc={rc}"
            print(
                f"[preflight] {step} {status} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    bad = [(s, rc) for s, rc in results if rc != 0]
    if bad:
        print(
            f"[preflight:{caller}] 完成, {len(bad)} 步失败 (非关键, 继续): {bad}",
            flush=True,
        )
    else:
        print(f"[preflight:{caller}] 数据前置全部 ok", flush=True)
    return results


if __name__ == "__main__":
    run_preflight("standalone")
