# -*- coding: utf-8 -*-
"""09-09 现有模块先出清单 (用户指令: 重训全链完成前, 先用现有模块出今日预测).

夜链编排器已于 preflight 阶段让位 (retrain 未启动, taskkill 树杀);
本器串行跑 legacy_prob_head → legacy(gen) → deliver, tag=20260909,
fail-fast, 终态写 tmp_t/_predict_first_0909_state.json.
全部退出后由主会话重启 tmp_t/_nightp_orch_0909.py 全链.
子步骤进程名自带 HEAVY_SENTINELS; 本器自身不注册哨兵 (与夜链编排器同设计).
"""
import json
import subprocess
import sys
import time

TAG = "20260909"
PY = sys.executable
STEPS = [
    ("legacy_prob_head", [PY, "-u", "scripts/_train_legacy_prob_head.py"]),
    ("legacy", [PY, "-u", "scripts/_gen_legacy_list.py", TAG]),
    ("deliver", [PY, "-u", "scripts/_deliver_legacy_list.py", TAG]),
]
STATE = "tmp_t/_predict_first_0909_state.json"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    results = {}
    for name, argv in STEPS:
        log(f"start {name}")
        t0 = time.time()
        rc = subprocess.run(argv, cwd=".").returncode
        results[name] = rc
        log(f"{'ok' if rc == 0 else 'FAIL'} {name} rc={rc} ({time.time() - t0:.0f}s)")
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump({"status": "ok" if rc == 0 else "failed", "steps": results,
                       "ts": time.time()}, f)
        if rc != 0:
            log(f"PREDICT_FIRST_DONE status=failed failed={name}")
            return 1
    log("PREDICT_FIRST_DONE status=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
