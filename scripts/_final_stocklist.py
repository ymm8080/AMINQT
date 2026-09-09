"""终版清单 (2026-09-06 用户): 全闸之后产出的单文件 Excel.

一文件合并三源当日已过闸清单 (legacy / parallel / prob10dens), 每行带:
- module 列: legacy / prob10dens / parallel 的 systems 标记
  (sniper / fusion / fusion+sniper / 空 = parallel 无系统标记)
- win_rate 列 = 当夜死区闸该线滚动赢率 (_deadzone_guard.win_rate 单源;
  样本不足留空)
- status 列 = landed (当日任一次实推已落袋) / blocked;
  reason 优先级: flush(守卫已删) > deadzone(停推) > manual/ui_fail/not_pushed
ths_push_result 文件链末归档进 ths_push_archive/ 子目录 (2026-09-08 用户: STOCK LIST
只留清单/终表): 看板推送状态卡数据源 + 本脚本 landed 合并依据不变 (归档兼读)。
输出: STOCK_LIST_DIR/stocklist_final_{date}__{HH}.xlsx (HH 戳 WORM, 同日重跑各留各的)。
链路: run_daily_automation "final_stocklist" 步骤, 置 ths_flush_guard 后 (非关键)。
"""

import argparse
import datetime
import glob
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import STOCK_LIST_DIR  # noqa: E402
from scripts._deadzone_guard import is_alarm, win_rate  # noqa: E402
from scripts._ths_watchlist_push import (  # noqa: E402
    THS_PUSH_ARCHIVE,
    read_push_results,
)

# (源名, 文件 glob, 死区闸线名) — 顺序 = 终表行序 (生产在前)
_SOURCES = [
    ("legacy", "legacy_stocklist_{date}__*.csv", "top10"),
    ("parallel", "parallel_shortlist_{date}__*.csv", "parallel"),
    ("prob10dens", "prob10dens_{date}__*.csv", "prob10dens"),
]


def _newest(pattern: str, list_dir: Path) -> Path | None:
    hits = glob.glob(str(list_dir / pattern))
    if not hits:
        return None
    return Path(max(hits, key=os.path.getmtime))


def _flush_removed(date: str, list_dir: Path) -> set[str]:
    """当日放量下跌守卫已删的自选股 (现不在自选 = blocked/flush)。"""
    fp = list_dir / f"ths_flush_removed_{date}__flushguard.csv"
    if not fp.exists():
        return set()
    df = pd.read_csv(fp, dtype={"symbol": str})
    return set(df["symbol"].dropna().astype(str).str.zfill(6))


def build(date: str, list_dir=STOCK_LIST_DIR) -> pd.DataFrame:
    """三源合并 + module/win_rate/status/reason 列 (纯读盘, 可单测)。"""
    list_dir = Path(list_dir)
    res = read_push_results(date, list_dir=list_dir)
    if res.empty:
        landed: set[str] = set()
        result_status: dict[str, str] = {}
    else:
        syms = res["symbol"].astype(str).str.zfill(6)
        landed = set(syms[res["status"] == "landed"])
        result_status = dict(zip(syms, res["status"]))
    removed = _flush_removed(date, list_dir)

    frames: list[pd.DataFrame] = []
    for name, pat, line in _SOURCES:
        fp = _newest(pat.format(date=date), list_dir)
        if fp is None:
            print(f"[final] {name}: 无当日清单, 跳过")
            continue
        df = pd.read_csv(fp, dtype={"symbol": str})
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        if "systems" in df.columns:  # parallel: sniper/fusion 标记, 空=无
            df.insert(0, "module", df["systems"].fillna("").astype(str).str.strip())
        else:
            df.insert(0, "module", name)
        wr = win_rate(line, date)
        df["win_rate"] = f"{wr:.1%}" if wr is not None else ""
        alarm, _why = is_alarm(line, date)
        reasons = []
        statuses = []
        for sym in df["symbol"]:
            if sym in removed:
                statuses.append("blocked")
                reasons.append("flush")
            elif sym in landed:
                statuses.append("landed")
                reasons.append("")
            elif alarm:
                statuses.append("blocked")
                reasons.append("deadzone")
            else:
                statuses.append("blocked")
                reasons.append(result_status.get(sym, "not_pushed"))
        df["status"] = statuses
        df["reason"] = reasons
        frames.append(df)
        print(
            f"[final] {name}: {fp.name} → {len(df)} 行 "
            f"(win_rate={'-' if wr is None else f'{wr:.1%}'}, alarm={alarm})"
        )
    if not frames:
        raise SystemExit(f"[final] {date} 三源清单均无, 不产出")
    return pd.concat(frames, ignore_index=True)


def archive_push_artifacts(date: str, list_dir=STOCK_LIST_DIR) -> int:
    """当日推送中间产物 (ths_watchlist txt + ths_push_result csv) 移入归档子目录.

    移动非删除 (WORM): read_push_results 兼读归档目录, 看板推送状态卡/本脚本
    landed 合并不受影响. 只能在链末 (本步骤) 调用 — ths_flush_guard 同夜还要
    glob 主目录取推送成员并集, 归档提前会漏成员.
    """
    list_dir = Path(list_dir)
    dest = list_dir / THS_PUSH_ARCHIVE
    moved = 0
    for pat in (f"ths_watchlist_{date}__*.txt", f"ths_push_result_{date}__*.csv"):
        for fp in list_dir.glob(pat):
            dest.mkdir(exist_ok=True)
            shutil.move(str(fp), dest / fp.name)
            moved += 1
    return moved


def write(df: pd.DataFrame, date: str, list_dir=STOCK_LIST_DIR) -> Path:
    hh = datetime.datetime.now().strftime("%H")
    fp = Path(list_dir) / f"stocklist_final_{date}__{hh}.xlsx"
    with pd.ExcelWriter(fp, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="stocklist_final")
    return fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "date", nargs="?", default=None, help="YYYYMMDD (缺省取最新 legacy 清单日)"
    )
    ap.add_argument("--list-dir", default=str(STOCK_LIST_DIR))
    args = ap.parse_args()
    date = args.date
    if date is None:
        fp = _newest("legacy_stocklist_????????__*.csv", Path(args.list_dir))
        if fp is None:
            raise SystemExit("STOCK LIST 目录无任何 legacy 清单")
        date = re.search(r"legacy_stocklist_(\d{8})__", fp.name).group(1)
    df = build(date, list_dir=Path(args.list_dir))
    out = write(df, date, list_dir=Path(args.list_dir))
    n_landed = int((df["status"] == "landed").sum())
    print(
        f"[final] {out} ({len(df)} 行, landed={n_landed}/blocked="
        f"{int((df['status'] == 'blocked').sum())})"
    )
    n_arch = archive_push_artifacts(date, list_dir=Path(args.list_dir))
    if n_arch:
        print(f"[final] 推送中间产物归档 {n_arch} 件 → {THS_PUSH_ARCHIVE}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
