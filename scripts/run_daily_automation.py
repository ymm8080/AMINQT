"""四模块每日自动化编排 (2026-08-06) — 重训 → 预测出清单 → 落盘 → 看板可见.

覆盖"四个模块":
  1. legacy main   — 周频全量重训 (_retrain_legacy_full.py, OOS 过闸才切换 current)
  2. legacy dual   — 同上 (同一脚本双板)
  3. 并行 sniper   — 每日重生成 (app.pipeline_parallel.runner 回测 + 短名单)
  4. 并行 fusion   — 同上 (同一 runner, 内含 slow_bull 一并输出)

步骤按依赖顺序子进程隔离执行 (每步独立进程释放内存, 避免 44GB commit 上限 OOM):
  [refresh]  scripts/_refresh_parallel_checkpoints.py  并行行集 3y 检查点 (需 22:00 fetch 后)
  [retrain]  scripts/_retrain_legacy_full.py <tag>     legacy 周频重训 (仅 RETRAIN_WEEKDAY)
  [parallel] python -m app.pipeline_parallel.runner     并行回测 + 短名单 (sniper/fusion/slow_bull)
  [legacy]   scripts/_gen_legacy_list.py <tag>          legacy 预测出清单 (用最新 current)
  [deliver]  scripts/_deliver_legacy_list.py <tag>      legacy 清单交付 STOCK_LIST_DIR

"推送看板" = 各步落盘到看板只读目录, 看板渲染时自动展示:
  模型 → models/pipeline1/current_meta.json + *.pkl (档案页·模型档案)
  回测 → BACKTEST_RESULT_DIR/<ts>/ (档案页·回测历史, 含 hv 胜率图)
  清单 → STOCK_LIST_DIR (档案页·落盘清单) + PredictionDB (每日预测)

失败策略 (失败要大声): refresh 失败 → 跳过 parallel (无新鲜行集); retrain 失败 →
继续当日清单 (沿用现有模型); 任一关键步骤 (legacy/deliver) 失败 → 非零退出.
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
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
LOG_DIR = os.path.join(ROOT, "logs")

# 0=Mon .. 6=Sun. 周频重训落在周五晚 (周末分析用周五收盘后最新模型).
RETRAIN_WEEKDAY = 4

# (步骤名, argv) — argv 不含解释器, run_step 负责拼 [PY, "-u", ...]
_STEPS = {
    "refresh": ["scripts/_refresh_parallel_checkpoints.py"],
    "retrain": ["scripts/_retrain_legacy_full.py", "{tag}"],
    "parallel": ["-m", "app.pipeline_parallel.runner"],
    "legacy": ["scripts/_gen_legacy_list.py", "{tag}"],
    "deliver": ["scripts/_deliver_legacy_list.py", "{tag}"],
}
# refresh 失败后应跳过的后续步骤 (parallel 需要新鲜检查点)
_DEPENDS = {"parallel": "refresh"}
# 关键步骤: 失败 → 整个任务非零退出 (看板当日清单缺失)
_CRITICAL = {"legacy", "deliver"}


def plan_steps(
    today: _dt.date,
    *,
    skip_checkpoints: bool = False,
    skip_retrain: bool = False,
    skip_parallel: bool = False,
) -> list[str]:
    """按星期 + skip 标志选出当日步骤序列 (纯函数, 可单测)."""
    steps: list[str] = []
    if not skip_checkpoints:
        steps.append("refresh")
    if not skip_retrain and today.weekday() == RETRAIN_WEEKDAY:
        steps.append("retrain")
    if not skip_parallel:
        steps.append("parallel")
    steps.append("legacy")
    steps.append("deliver")
    return steps


def _log_fh(tag: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    return open(os.path.join(LOG_DIR, f"daily_automation_{tag}.log"), "a", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="四模块每日自动化")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    ap.add_argument("--skip-checkpoints", action="store_true", help="跳过并行检查点刷新")
    ap.add_argument("--skip-retrain", action="store_true", help="跳过 legacy 周频重训")
    ap.add_argument("--skip-parallel", action="store_true", help="跳过并行系统重生成")
    ap.add_argument("--tag", default=None, help="清单交易日 YYYYMMDD (默认今天)")
    args = ap.parse_args()

    today = _dt.date.today()
    tag = args.tag or today.strftime("%Y%m%d")

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
        return 2

    steps = plan_steps(
        today,
        skip_checkpoints=args.skip_checkpoints,
        skip_retrain=args.skip_retrain,
        skip_parallel=args.skip_parallel,
    )
    print(
        f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] 四模块自动化 tag={tag} "
        f"星期={today.strftime('%A')} 步骤={steps}",
        flush=True,
    )
    if args.dry_run:
        for s in steps:
            print(f"  [dry] {s}: python {' '.join(_STEPS[s])}".replace("{tag}", tag), flush=True)
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
            rc = subprocess.call(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
            dt = time.time() - t0
            ok = rc == 0
            status = "ok" if ok else "FAIL"
            print(f"[{_dt.datetime.now():%H:%M:%S} {status}] {step} rc={rc} ({dt:.0f}s)", flush=True)
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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
