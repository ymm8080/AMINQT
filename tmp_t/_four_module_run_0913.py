# -*- coding: utf-8 -*-
"""[0913] 四模块手动生产链: LEGACY/PARALLEL/density/slowbull 重训+预测+推送 THS.

用户令 0913 晨: 池重扫完成后重训+预测四模块并推送同花顺。
发射条件: ① tmp_t/_rescan_mag_pool_0913.state.json phase=DONE ② 池判词已
接线 (赢家改 config.py SNIPER/FUSION + WORM 记录, 或判零改动) — 本脚本只硬闸
①, ② 由人保证 (勿跳过判词直接发射, 会用旧池白扫)。

序列 = run_daily_automation.plan_steps 全量日链序 − 未令步 (drift/drift_
parallel/shadow_xmodule/gate_audit/a1/gappocket), retrain 恒上 (手动令, 不走
RETRAIN_WEEKDAY 分支)。复用 _STEPS/_run_step_with_watchdog/超时; fail-fast:
关键步失败停链写 state; 全程 WORM 日志 logs/four_module_run_{TAG}.log。

注意: prob_head 现带 purged_v2_wf2 影子 (2 板×3 档×2 折全史拟合) — 原超时
3h 上探 5h (e94f47eb 遗留盯梢项)。

用法: python tmp_t/_four_module_run_0913.py [--force]
  --force = 跳过重扫 DONE 硬闸 (仅限重扫 FAIL 后用户明令续跑四模块时用)。
"""

import datetime as _dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from scripts.run_daily_automation import (  # noqa: E402
    PY,
    _STEPS,
    _STEP_TIMEOUT_S,
    _prevent_sleep,
    _run_step_with_watchdog,
)

TAG = "20260913"
LOG_DIR = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(LOG_DIR, f"four_module_run_{TAG}.log")
STATE_PATH = os.path.join(LOG_DIR, f"four_module_run_{TAG}.state.json")
RESCAN_STATE = os.path.join(HERE, "_rescan_mag_pool_0913.state.json")

_SEQ = [
    "cyq", "sw_history", "freshness", "canary",
    "retrain", "legacy_prob_head", "legacy", "deliver",
    "refresh", "parallel", "prob_head", "deliver_parallel",
    "ths_push", "prob10dens_push", "slowbull_shadow",
    "ths_flush_guard", "final_stocklist", "stocklist_combined",
]
# 告警式, 失败不拦链: 数据前置三步+canary 恒软; THS 推送两步软 (ensure_idle 用户
# 在场即返 False → rc=1, 0910 实证; 失败后单独重试, 勿拦其后 slowbull/清单装配五步)
_SOFT = {"cyq", "sw_history", "freshness", "canary", "ths_push", "prob10dens_push"}
_TIMEOUTS = {**_STEP_TIMEOUT_S, "prob_head": 5 * 3600}


def _state(status: str, **kw) -> None:
    doc = {"tag": TAG, "status": status, "ts": time.time(), **kw}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"[fourmod] state={status} {kw or ''}", flush=True)


def main() -> int:
    force = "--force" in sys.argv
    _prevent_sleep()

    if not force:
        try:
            st = json.loads(open(RESCAN_STATE, encoding="utf-8").read()).get("phase")
        except Exception:
            st = None
        if st != "DONE":
            _state("refused", reason=f"rescan_phase={st}")
            print("[fourmod] 池重扫未 DONE, 拒启 (先读判词接线; 明令续跑加 --force)", flush=True)
            return 2

    _state("running", step="(start)")
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        for step in _SEQ:
            argv = [PY, "-u"] + [a.replace("{tag}", TAG) for a in _STEPS[step]]
            t0 = time.time()
            msg = f"[{_dt.datetime.now():%H:%M:%S} start] {step}: {' '.join(argv)}"
            print(msg, flush=True)
            print(msg, file=fh, flush=True)
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            rc, timed_out = _run_step_with_watchdog(argv, fh, env, _TIMEOUTS[step])
            if timed_out:
                rc = 124
            ok = rc == 0
            msg = f"[{_dt.datetime.now():%H:%M:%S} {'ok' if ok else 'FAIL'}] {step} rc={rc} ({time.time() - t0:.0f}s)"
            print(msg, flush=True)
            print(msg, file=fh, flush=True)
            _state("running", step=step, last_rc=rc)
            if not ok and step not in _SOFT:
                _state("failed", failed_step=step, rc=rc)
                print(f"[fourmod] 关键步 {step} 失败, fail-fast 停链", flush=True)
                return 1
    _state("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
