"""GENIOUS 缺文件看门狗 — 扫近窗交易日, 缺谁补谁 (顺序, --no-push)。

背景: 2026-10-22 与 10-26 两次漏交付 — 20:30 计划任务未触发或失败后, 错过日期
无人补 (StartWhenAvailable 补跑用的是*当天* tag, 错过日永缺)。本脚本注册为
AMINQT-Genious-Watchdog-2110 (周一~五 21:10, StartWhenAvailable): 晚于
GENIOUS (20:30, 实测 ~10min) 早于 LEGACY (21:30); 开机补跑时也能补历史缺日。

规则:
- 窗口 = 最近 --lookback 个交易日, 含今日 (20:30 主任务当晚失败时由本任务自愈);
  存活 _genious_excel.py 进程 → 整轮让路 (WORM 重跑只会产出 __HHMMSS 副本)。
- 任一夜链哨兵 (HEAVY/三条链) 存活 → 让路 (2026-10-23 双链互等死锁的教训)。
- 交易日历不可得 → fail-open 退出: 宁可不补, 不盲补节假日。
- 每次运行必留心跳日志 (含无事可做); 单轮最多补 --max-runs 个, 余量顺延下轮。
- 退出码: 0 = 无缺/全补成/让路; 1 = 有补跑失败 (证据在 logs/genious_{date}.state.json)。

用法:
    python scripts/_genious_backfill_missing.py --dry-run   # 只报缺哪些
    python scripts/_genious_backfill_missing.py
"""

from __future__ import annotations

import argparse
import datetime
import json
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
GENIOUS_SCRIPT = Path(__file__).resolve().parent / "_genious_excel.py"
LOG_PATH = PROJECT_ROOT / "logs" / "genious_watchdog.log"
RUN_TIMEOUT_S = 40 * 60  # 正常 ~10min; bigdrop 重建 ~5min, 留 4 倍余量后杀

log = logging.getLogger("genious-watchdog")


def _setup_logging() -> None:
    log.setLevel(logging.INFO)
    if log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [watchdog] %(message)s")
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)


def _state_status(tag: str) -> str | None:
    p = PROJECT_ROOT / "logs" / f"genious_{tag}.state.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("status")
    except Exception as e:  # noqa: BLE001 — 状态读不到按未知处理, 不阻断
        log.warning("读 %s 失败: %s", p.name, e)
        return None


def _window_tags(lookback: int, cal) -> list[str]:
    idx = pd.DatetimeIndex(cal)
    past = idx[idx <= pd.Timestamp(datetime.date.today())]
    return [d.date().strftime("%Y%m%d") for d in past[-lookback:]]


def _delivered(tag: str) -> bool:
    return any(STOCK_LIST_DIR.glob(f"GENIOUS_{tag}*.xlsx"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lookback", type=int, default=5, help="回看交易日数 (含今日)")
    ap.add_argument("--max-runs", type=int, default=3, help="单轮最多补几个 (余量顺延)")
    ap.add_argument("--dry-run", action="store_true", help="只报缺哪些, 不补")
    args = ap.parse_args()

    _setup_logging()
    LOG_PATH.parent.mkdir(exist_ok=True)

    if live := find_conflicts(("_genious_excel.py",)):
        log.info("让路: GENIOUS 主进程存活 (pid=%s), 本轮不补", live[0]["pid"])
        return 0
    if chain := find_conflicts(CHAIN_SENTINELS):
        log.info(
            "让路: 夜链存活 (sentinel=%s, pid=%s), 避免互等死锁",
            chain[0]["sentinel"],
            chain[0]["pid"],
        )
        return 0

    cal = load_trade_cal()
    if cal is None or len(cal) == 0:
        log.warning("交易日历不可得 → fail-open 退出 (不盲补节假日)")
        return 0

    window = _window_tags(args.lookback, cal)
    missing = [t for t in window if not _delivered(t)]
    log.info(
        "窗口 %s..%s (%d 个交易日), 缺 %d: %s",
        window[0] if window else "-",
        window[-1] if window else "-",
        len(window),
        len(missing),
        missing or "无",
    )
    if args.dry_run or not missing:
        return 0

    failed: list[str] = []
    for tag in missing[: args.max_runs]:
        log.info("补跑 %s 开始 (--no-push)", tag)
        try:
            r = subprocess.run(
                [
                    sys.executable,
                    str(GENIOUS_SCRIPT),
                    tag,
                    "--wait-min",
                    "0",
                    "--no-push",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=RUN_TIMEOUT_S,
            )
            status = _state_status(tag)
            ok = r.returncode == 0 and status == "ok"
        except subprocess.TimeoutExpired:
            ok, status, r = False, "timeout", None
        if ok:
            log.info("补跑 %s 成功 (state=ok)", tag)
        else:
            failed.append(tag)
            log.error(
                "补跑 %s 失败: rc=%s state=%s%s",
                tag,
                getattr(r, "returncode", "-"),
                status,
                "" if r is None else f" stderr_tail={r.stderr[-500:]!r}",
            )
    if len(missing) > args.max_runs:
        log.warning("缺 %d > max-runs %d, 余 %s 顺延下轮",
                    len(missing), args.max_runs, missing[args.max_runs:])

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
