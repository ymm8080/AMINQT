"""stocklist_combined 缺文件看门狗 — 扫近窗交易日, 缺谁补谁 (LEGACY 源在才补)。

背景: 2026-10-30 起 PARALLEL 夜链 refresh 步骤 OOM (主板 3y 检查点 1.67 GiB
分配失败) 连锁 deliver_parallel 失败, combined 作为其依赖步骤每晚被跳过 —
2026-10-29 是最后一次产出, 2026-11-11 用户令 "fix to have daily creation"。
两道修复:
  1) run_parallel_automation._DEPENDS_LOCAL 里 combined 解耦 deliver_parallel:
     它的硬依赖是 LEGACY 清单 (LEGACY 链独立健康), PARALLEL/密度/SLOW_BULL
     缺源本来就跳页 — parallel 侧任何失败不再拖垮 combined;
  2) 本看门狗 (AMINQT-Combined-Watchdog-0830, 周一~五 08:30): 夜链整链没跑 /
     中途死 / 当晚仍失败的日期, 次晨按 LEGACY 清单补产 (缺源跳页口径,
     PARALLEL 页缺失留待 parallel 侧自愈后再重跑到 __vN 变体)。

规则 (与 _genious_backfill_missing.py 同族约定):
- 窗口 = 最近 --lookback 个交易日, 含今日; 交易日历不可得 → fail-open 退出:
  宁可不补, 不盲补节假日。
- 欠交付 = 该日有 legacy_stocklist_{date}__*.csv 且无 stocklist_combined_{date}*.xlsx
  (glob 认 __v2/__v3 变体, 与 _stocklist_combined._next_path 的 WORM 取号口径一致)。
  LEGACY 清单不在 → 不补 (LEGACY/PARALLEL 双缺才退出是 builder 自己的契约)。
- 任一夜链哨兵或 _stocklist_combined.py 存活 → 整轮让路 (2026-10-23 双链互等
  死锁的教训; 并发重跑只会互出 __vN 副本)。
- 每次运行必留心跳 (含无事可做); 单轮最多补 --max-runs 个, 余量顺延下轮。
- 退出码: 0 = 无缺/全补成/让路; 1 = 有补跑失败 (输出留在心跳日志)。

用法:
    python scripts/_stocklist_combined_backfill.py --dry-run    # 只报缺哪些
    python scripts/_stocklist_combined_backfill.py --lookback 8  # 补更长缺口
"""

from __future__ import annotations

import argparse
import datetime
import glob
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline1.freshness_guard import load_trade_cal  # noqa: E402
from config.settings import STOCK_LIST_DIR  # noqa: E402
from scripts._run_guard import CHAIN_SENTINELS, find_conflicts  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).resolve().parent / "_stocklist_combined.py"
LOG_PATH = PROJECT_ROOT / "logs" / "combined_watchdog.log"
LIST_DIR = Path(STOCK_LIST_DIR)
RUN_TIMEOUT_S = 20 * 60  # 正常 ~90s; BIGDROP 重建 ~5min, 留 4 倍余量后杀
# 让路哨兵: 夜链全哨兵 + builder 自身 (补跑子进程 cmdline 含它)。
# 注意 "_stocklist_combined.py" 不是 "_stocklist_combined_backfill.py" 的子串,
# 本看门狗进程不会自命中 (find_conflicts 另外恒排除自身 PID)。
_SELF_SENTINEL = "_stocklist_combined.py"

log = logging.getLogger("combined-watchdog")


def _setup_logging() -> None:
    log.setLevel(logging.INFO)
    if log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [combined-watchdog] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)


def _window_tags(lookback: int, cal) -> list[str]:
    idx = pd.DatetimeIndex(cal)
    past = idx[idx <= pd.Timestamp(datetime.date.today())]
    return [d.strftime("%Y%m%d") for d in past[-lookback:]]


def _has_combined(tag: str) -> bool:
    return bool(glob.glob(str(LIST_DIR / f"stocklist_combined_{tag}*.xlsx")))


def _has_legacy(tag: str) -> bool:
    return bool(glob.glob(str(LIST_DIR / f"legacy_stocklist_{tag}__*.csv")))


def main() -> int:
    ap = argparse.ArgumentParser(description="stocklist_combined 缺文件看门狗")
    ap.add_argument("--lookback", type=int, default=5, help="扫最近 N 个交易日 (含今日)")
    ap.add_argument("--max-runs", type=int, default=5, help="单轮最多补跑数, 余量顺延下轮")
    ap.add_argument("--dry-run", action="store_true", help="只报缺哪些, 不补")
    args = ap.parse_args()

    _setup_logging()

    cal = load_trade_cal()
    if cal is None or len(cal) == 0:
        log.error("交易日历不可得, fail-open 退出 (宁可不补, 不盲补节假日)")
        return 0

    tags = _window_tags(args.lookback, cal)
    due = [t for t in tags if _has_legacy(t) and not _has_combined(t)]
    log.info("窗口 %s: 欠交付 %s", tags, due or "无")

    if not due:
        return 0
    if args.dry_run:
        log.info("dry-run, 不补")
        return 0

    conflicts = find_conflicts(CHAIN_SENTINELS + (_SELF_SENTINEL,))
    if conflicts:
        for c in conflicts:
            log.warning(
                "让路: %s (PID %s) %s", c["sentinel"], c["pid"], c["cmdline"][:120]
            )
        log.warning("夜链/combined 存活, 本轮让路退出")
        return 0

    failures: list[str] = []
    batch = due[: max(args.max_runs, 0)]
    for tag in batch:
        log.info("补跑 %s: %s", tag, BUILDER.name)
        try:
            r = subprocess.run(
                [sys.executable, "-u", str(BUILDER), tag],
                cwd=str(PROJECT_ROOT),
                timeout=RUN_TIMEOUT_S,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            log.error("%s 超时 %ss, 已杀 (WORM 会把半截文件视为已交付, 需人工核)",
                      tag, RUN_TIMEOUT_S)
            failures.append(tag)
            continue
        tail = "\n".join((r.stdout or "").splitlines()[-6:])
        if r.returncode != 0:
            log.error("%s rc=%d\n%s\nSTDERR: %s",
                      tag, r.returncode, tail, (r.stderr or "")[-600:])
            failures.append(tag)
        else:
            log.info("%s rc=0\n%s", tag, tail)

    if len(due) > len(batch):
        log.info("余 %d 日顺延下轮 (--max-runs=%d)", len(due) - len(batch), args.max_runs)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
