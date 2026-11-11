"""GENIOUS「冠军四段」页 → 同花顺自选股 每日推送 (2026-11-11 用户令
"make sure there is auto push to ths for genious file 冠军四段 ON DAILY BASIS")。

源 = 交付层 GENIOUS_{date}*.xlsx 的「冠军四段」页 (同日重跑取 mtime 最新,
与 _ths_watchlist_push.collect_lists 同口径) — 该页本身已是裁窄冠军单, 整页
推送 (不做 次数双非零 过滤, 那是板前哨页的口径)。

闸 (2026-11-11): config GENIOUS["push_champ4_to_ths"]=True 单独放行
genious__champ4 推送单, 不受 2026-11-03 "stop pushing to ths" 总闸
(push_to_ths=False) 影响 — 板前哨/影子单等其余源仍全停 (见
_ths_watchlist_push._push_allowed)。

幂等: 该日已有 ths_push_result_{date}__*__genious__champ4.csv 且全部行
status=landed → 视为已推送, rc=0 跳过; 有 blocked/manual 行 → 重推
(新 HH 槽位 txt, WORM 各留各的)。双触发: GENIOUS 交付后当晚 spawn
(_genious_excel._spawn_ths_push(champ4=True)) + 晨间计划任务
AMINQT-Champ4-Push-0745 (周一~五 07:45) 重推兜底 — 用户在场时 UI 空闲闸
fail-closed (结果单全 blocked), 次晨任务自动补推。

每日节奏: GENIOUS_{D}.xlsx 晚间交付 → 当晚或次晨推入自选 (表内「明日开盘」
口径本来就是次日执行)。

用法:
    python scripts/_ths_champ4_push.py               # 最新交付日, 幂等
    python scripts/_ths_champ4_push.py 20261111      # 指定交付日
    python scripts/_ths_champ4_push.py --dry-run     # 只列码单不写不推
"""

from __future__ import annotations

import argparse
import datetime
import glob
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import GENIOUS, STOCK_LIST_DIR  # noqa: E402
from scripts._ths_watchlist_push import (  # noqa: E402
    CODE_RE,
    _newest,
    push_via_ths,
    write_ths_txt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIST_DIR = Path(STOCK_LIST_DIR)
LOG_PATH = PROJECT_ROOT / "logs" / "champ4_push.log"
SHEET = "冠军四段"
TAG = "genious__champ4"

log = logging.getLogger("champ4-push")


def _setup_logging() -> None:
    log.setLevel(logging.INFO)
    if log.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s [champ4-push] %(message)s")
    try:
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError:  # 心跳日志不可写不拦推送
        pass
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)


def _cell(v) -> str:
    """冠军四段 symbol 单元格归一: 数值 1331 → '001331'; 其余去空白."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, bool):
        return ""
    if isinstance(v, (int, float)):
        try:
            return str(int(v)).zfill(6)
        except (ValueError, OverflowError):
            return ""
    return str(v).strip()


def champ4_codes(xlsx, sheet: str = SHEET) -> list[str]:
    """交付层 xlsx → 冠军四段页 symbol 序 (保序去重, 非法码滤除).

    表头行动态定位 (banner/图例行布局变更不脆): 前 6 行内找含 "symbol" 的行.
    """
    raw = pd.read_excel(xlsx, sheet_name=sheet, header=None, dtype=object)
    hdr_pos = None
    for i in range(min(6, len(raw))):
        vals = [str(v).strip() for v in raw.iloc[i].tolist()]
        if "symbol" in vals:
            hdr_pos = i
            break
    if hdr_pos is None:
        raise ValueError(f"{os.path.basename(xlsx)} 「{sheet}」页找不到 symbol 表头行")
    col_idx = raw.iloc[hdr_pos].tolist().index("symbol")
    col = raw.iloc[hdr_pos + 1 :, col_idx]
    out: list[str] = []
    for v in col:
        c = _cell(v)
        if CODE_RE.match(c) and c not in out:
            out.append(c)
    return out


def latest_genious_xlsx() -> tuple[str | None, Path | None]:
    prefix = GENIOUS["filename_prefix"]
    fp = _newest(f"{prefix}_*.xlsx", LIST_DIR)
    if fp is None:
        return None, None
    date = re.search(rf"{prefix}_(\d{{8}})", os.path.basename(fp)).group(1)
    return date, Path(fp)


def _result_files(date: str, list_dir=None) -> list[Path]:
    # 晚绑定: 缺省取当期模块 LIST_DIR (单测 monkeypatch 生效的前提).
    d = Path(LIST_DIR if list_dir is None else list_dir)
    pats = [
        d / f"ths_push_result_{date}__*__{TAG}.csv",
        d / "ths_push_archive" / f"ths_push_result_{date}__*__{TAG}.csv",
    ]
    hits: list[Path] = []
    for pat in pats:
        hits.extend(Path(x) for x in glob.glob(str(pat)))
    return sorted(set(hits))


def already_landed(date: str, list_dir=None) -> bool:
    """该日 champ4 结果单存在且全部行 landed → 已推送 (True); 缺/有 blocked·manual → False."""
    if list_dir is None:
        list_dir = LIST_DIR
    for fp in _result_files(date, list_dir):
        df = pd.read_csv(fp, dtype=str)
        if df.empty or "symbol" not in df.columns or "status" not in df.columns:
            continue
        if (df["status"].fillna("") == "landed").all():
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GENIOUS 冠军四段 → THS 自选股 每日推送")
    ap.add_argument("date", nargs="?", default=None, help="YYYYMMDD (缺省=最新交付日)")
    ap.add_argument("--dry-run", action="store_true", help="只列码单, 不写 txt 不进 UI")
    ap.add_argument("--force", action="store_true", help="忽略已 landed 结果单, 强制重推")
    args = ap.parse_args(argv)

    _setup_logging()

    if not GENIOUS.get("push_champ4_to_ths"):
        log.info("[no-push] push_champ4_to_ths=False: champ4 推送关闭 (11-11 令回退口径)")
        return 0

    date, xlsx = (args.date, None) if args.date else (None, None)
    if date is None:
        date, xlsx = latest_genious_xlsx()
        if date is None:
            log.error("STOCK LIST 目录无任何 GENIOUS 交付层, 无法推送")
            return 1
    if xlsx is None:
        xlsx = Path(_newest(f"{GENIOUS['filename_prefix']}_{date}*.xlsx", LIST_DIR) or "")
        if not xlsx.exists():
            log.error("%s 无 GENIOUS 交付层 xlsx", date)
            return 1

    codes = champ4_codes(xlsx)
    if not codes:
        log.info("%s 冠军四段为空, 无可推 (约 26%% 交易日空榜属正常)", date)
        return 0

    if not args.force and already_landed(date):
        log.info("%s champ4 已全 landed (%d 票), 幂等跳过", date, len(codes))
        return 0

    log.info("%s champ4 %d 票 (源 %s): %s", date, len(codes), xlsx.name, ",".join(codes))
    if args.dry_run:
        log.info("dry-run: 只列码单, 不写 txt 不进 UI")
        return 0

    slot = datetime.datetime.now().strftime("%H")
    txt = LIST_DIR / f"ths_watchlist_{date}__{slot}__{TAG}.txt"
    write_ths_txt(codes, txt)
    log.info("推送单 %s 已写, 进入 UI 推送 (用户在场会 fail-closed 写 blocked)", txt.name)
    ok = push_via_ths(txt)
    log.info("push_via_ths → %s (逐码终态看 %s)", ok, txt.name.replace("ths_watchlist_", "ths_push_result_").replace(".txt", ".csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
