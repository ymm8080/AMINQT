# -*- coding: utf-8 -*-
"""修改管线后延迟重训+预测 (2026-09-16 用户令: "AFTER RETRAIN AND PREDICT
PIPELINE IS MODIFY, RUN RETRAIN AND PREDICT 2 HOURS LATER").

通道: 检测管线代码 mtime 变化 → 定 2h 后的一次性触发 (schtasks /ONETIME,
独立进程不死守会话)。幂等: 同 tag 已挂任务则跳过。

用法:
  python scripts/_delayed_retrain_predict_0916.py            # 若管线有新改动, 挂 2h 延迟任务
  python scripts/_delayed_retrain_predict_0916.py --run-now  # 立即跑 (跳过延迟)
"""

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_FILES = [
    "app/pipeline1/engine.py",
    "app/pipeline1/features.py",
    "app/pipeline1/selection.py",
    "app/pipeline_parallel/runner.py",
    "app/pipeline_parallel/scoring.py",
    "scripts/_retrain_legacy_full.py",
    "scripts/run_daily_automation.py",
]
STAMP_PATH = ROOT / "logs" / "pipeline_last_modified.stamp"
TASK_NAME = "AMINQT-Delayed-Retrain-Predict"
PY = sys.executable

TRAIN_CMD = [PY, "-u", "scripts/train_predict_main.py"]


def _latest_pipeline_mtime() -> float | None:
    ts = [os.path.getmtime(ROOT / f) for f in PIPELINE_FILES if (ROOT / f).exists()]
    return max(ts) if ts else None


def _deferred_needed() -> bool:
    """管线有比上次 stamp 更新的改动 → 需要延迟触发."""
    m = _latest_pipeline_mtime()
    if m is None:
        return False
    try:
        last = float(STAMP_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return True
    return m > last


def _run_schtasks(args: list[str]) -> subprocess.CompletedProcess:
    """schtasks 输出是系统代码页 (GBK/CP936), 显式解码防 UnicodeDecodeError."""
    return subprocess.run(
        args, capture_output=True, text=True, encoding="gbk", errors="replace"
    )


def _schedule_delayed_run(delay_hours: float = 2.0) -> bool:
    """schtasks ONETIME +delay. 幂等: 已挂同名任务 → 返回 False (不重挂)."""
    q = _run_schtasks(["schtasks", "/query", "/tn", TASK_NAME])
    if q.returncode == 0 and TASK_NAME in (q.stdout or ""):
        print(f"[delay] {TASK_NAME} 已挂, 不重挂")
        return False
    run_at = (_dt.datetime.now() + _dt.timedelta(hours=delay_hours)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    cmd = [
        "schtasks",
        "/create",
        "/tn",
        TASK_NAME,
        "/tr",
        f'"{PY}" "{ROOT / "scripts" / "train_predict_main.py"}"',
        "/sc",
        "once",
        "/st",
        run_at[11:19],
        "/sd",
        run_at[:10],
        "/f",
    ]
    r = _run_schtasks(cmd)
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()
    print(f"[delay] 挂 {TASK_NAME} @{run_at}: rc={r.returncode} {out} {err}")
    return r.returncode == 0


def _mark_stamp() -> None:
    m = _latest_pipeline_mtime()
    if m is not None:
        STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        STAMP_PATH.write_text(f"{m:.6f}", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-now", action="store_true", help="立即重训+预测, 跳过延迟闸")
    ap.add_argument(
        "--delay-hours", type=float, default=2.0, help="延迟小时数 (默认 2)"
    )
    args = ap.parse_args()
    if not args.run_now:
        if not _deferred_needed():
            print("[delay] 管线无新改动, 不挂延迟任务")
            return 0
        if _schedule_delayed_run(args.delay_hours):
            _mark_stamp()
            return 0
        return 1
    rc = subprocess.call(TRAIN_CMD, cwd=ROOT)
    if rc == 0:
        _mark_stamp()
    return rc


if __name__ == "__main__":
    sys.exit(main())
