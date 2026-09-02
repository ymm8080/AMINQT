"""_freshness_check.py — 特征族新鲜度守卫 CLI (链步骤, 告警式不阻断, 2026-09-02).

加载 config/freshness_registry.yaml → expected_trading_date → run_checks →
逐条打印健康/违规 + WORM 写 logs/freshness_<YYYYMMDD>.json 报告.

设计决定: 退出码恒 0 (仅 --strict 下 critical 违规 exit 1). 为什么不阻断:
  1. 链对任何步骤失败一视同仁标 failed (run_daily_automation 的 _CRITICAL 集合
     实际未被引用, 非零退出 = 拉低终态), 一次新鲜度误报就会把当日交付打没;
  2. 08-27 事故: refresh 卡死 9h 全天零清单 — 新鲜度告警绝不能变成第二个阻断点;
  3. 告警靠大声日志 + 报告文件; 真正的强拦截只有两处专门设计过: 链级面板 A1 闸
     (读失败/严重停更才拦) 与 refresh 后置断言 (检查点落后拦), 其余一律告警.

用法:
  python scripts/_freshness_check.py                  # 链步骤 / 日常巡检, 恒 exit 0
  python scripts/_freshness_check.py --strict         # 手动: critical 违规 exit 1
  python scripts/_freshness_check.py --registry PATH  # 自定义注册表
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.pipeline1 import freshness_guard as fg  # noqa: E402

DEFAULT_REGISTRY = os.path.join(ROOT, "config", "freshness_registry.yaml")
LOG_DIR = os.path.join(ROOT, "logs")


def _report_path(today: datetime.date) -> str:
    """WORM: 带日期后缀不覆盖旧文件; 同日重跑加时间戳后缀."""
    base = os.path.join(LOG_DIR, f"freshness_{today:%Y%m%d}.json")
    if not os.path.exists(base):
        return base
    return os.path.join(
        LOG_DIR, f"freshness_{today:%Y%m%d}_{datetime.datetime.now():%H%M%S}.json"
    )


def _print_line(prefix: str, name: str, obs: dict) -> None:
    """逐条打印: 健康显示观测值, 违规显示 lag/阈值 + 细节."""
    crit = "[CRITICAL] " if obs.get("critical") else ""
    if prefix == "[ok]":
        print(f"  {prefix} {name} (max={obs.get('observed')}, threshold={obs.get('threshold')})")
    else:
        lag = obs.get("lag")
        lag_s = f"lag={lag}" if lag is not None else "lag=?"
        print(
            f"  {prefix} {crit}{name} (observed={obs.get('observed')}, "
            f"expected={obs.get('expected')}, {lag_s}, threshold={obs.get('threshold')})"
        )
        print(f"           -> {obs.get('detail')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="特征族新鲜度守卫 (告警式)")
    ap.add_argument(
        "--strict", action="store_true", help="手动巡检: critical 违规 exit 1"
    )
    ap.add_argument("--registry", default=DEFAULT_REGISTRY, help="注册表路径")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    registry = fg.load_registry(args.registry)
    today = datetime.date.today()
    cal = fg.load_trade_cal()  # 失败返回 None, expected/check 自动回退自然日
    expected, cal_source = fg.expected_trading_date(today, cal)
    result = fg.run_checks(registry, expected, cal)

    ts = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
    print(f"[{ts}] 特征族新鲜度检查: expected={expected} cal_source={cal_source}")
    for obs in result.observations:
        _print_line("[ok]", obs["name"], obs)
    for v in result.violations:
        _print_line("[STALE]", v["name"], v)
    for s in result.skipped:
        print(f"  [skip] {s['name']} ({s['reason']})")

    report = {
        "generated_at": ts,
        "expected": expected.isoformat(),
        "cal_source": cal_source,
        "registry": args.registry,
        "ok": [o["name"] for o in result.observations],
        "violations": result.violations,
        "skipped": result.skipped,
        "observations": result.observations,
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    path = _report_path(today)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    n_crit = sum(1 for v in result.violations if v["critical"])
    print(
        f"[{ts}] 汇总: ok={len(result.observations)} stale={len(result.violations)} "
        f"(critical={n_crit}) skipped={len(result.skipped)} → 报告 {path}"
    )
    # 退出码恒 0 — 告警式不阻断, 设计决定见模块 docstring (08-27 零清单教训)
    if args.strict and result.has_critical_violation:
        print("[strict] 存在 critical 违规 → exit 1 (仅手动巡检模式)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
