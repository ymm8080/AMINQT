"""同花顺自选股放量下跌守卫: 当日放量下跌标记 → 剔除自选股成员 + 落当日删除文档.

标记口径 (2026-09-03 用户: 判断只用日维度数据, OHLCV/动量/量能):
  event_flag  当日放量下跌 = pct_chg <= -5% 且 turnover_rate 截面秩 >= 0.80 (⑤同口径)
  f3_risk     F3 热度风险 = mean(动量秩 bias_20/60/120/250, 量能秩 amount/turnover_rate/
              volume_ratio/ma_vol_ratio_5_20) top10% (⑥同口径, 仅作文档警示列, 不触发删除)

成员来源: STOCK_LIST_DIR 近 10 日 ths_watchlist_*__*.txt 并集 (ths_push 推送记录代理;
未来接 THS 实读后替换). 删除动作: event_flag 且在自选股 → 记 action=删除.

文档 (WORM): STOCK_LIST_DIR/ths_flush_removed_{date}__flushguard.csv
  全部当日 event_flag 股票: symbol/pct_chg/turnover_rank/f3_risk/in_watchlist/action/asof

用法: python scripts/_ths_flush_guard.py [YYYYMMDD] [--apply]
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH, STOCK_LIST_DIR

FLUSH_PCT = -5.0
VOL_Q = 0.80
F3_TOP_Q = 0.10
MEMBERSHIP_DAYS = 10

MOM_COLS = ("bias_20", "bias_60", "bias_120", "bias_250")
HOTVOL_COLS = ("amount", "turnover_rate", "volume_ratio", "ma_vol_ratio_5_20")


def newest_panel_date() -> str:
    dts = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].unique()
    return str(pd.Timestamp(np.sort(pd.to_datetime(pd.Series(dts)).unique())[-1]).date())


def load_membership(list_dir, ref_date: str) -> set[str]:
    """近 N 日 ths_push 推送 txt 并集 = 自选股成员代理."""
    cutoff = datetime.strptime(ref_date, "%Y%m%d") - timedelta(days=MEMBERSHIP_DAYS)
    out: set[str] = set()
    for fp in list_dir.glob("ths_watchlist_*__*.txt"):
        m = re.search(r"ths_watchlist_(\d{8})__", fp.name)
        if m is None or datetime.strptime(m.group(1), "%Y%m%d") < cutoff:
            continue
        for line in fp.read_text(encoding="utf-8").split():
            if re.fullmatch(r"\d{6}", line.strip()):
                out.add(line.strip())
    return out


def compute_flags(date: str) -> pd.DataFrame | None:
    read_cols = ["symbol", "date", "pctChg", *MOM_COLS, *HOTVOL_COLS]
    df = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=read_cols,
        filters=[("date", "==", pd.Timestamp(date).normalize())],
    )
    if df.empty:
        return None
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["turnover_rank"] = df["turnover_rate"].rank(pct=True)
    f1 = df[list(MOM_COLS)].rank(pct=True).mean(axis=1)
    f2 = df[list(HOTVOL_COLS)].rank(pct=True).mean(axis=1)
    df["f3"] = pd.concat([f1, f2], axis=1).mean(axis=1)
    f3_thr = df["f3"].quantile(1 - F3_TOP_Q)
    df["event_flag"] = (df["pctChg"] <= FLUSH_PCT) & (df["turnover_rank"] >= VOL_Q)
    df["f3_risk"] = df["f3"] >= f3_thr
    return df


def delete_codes_from_watchlist(codes: list[str]) -> dict[str, str]:
    """UI 自动化逐码删除自选股成员 (截图读码定位→点击选中→Del→复读复验).

    09-03 定案: 键入定位判死 (数字键即便网格有焦点也被全局代码入口截走,
    Enter=跳分时图), 定位只允许截图读码. Del 无确认立即删除当前选中行,
    删后光标自动落下一行 (盲发第二下 Del 会连环误删). 安全设计:
      1. 入口空闲闸: 用户在场整体跳过 (非关键步, 文档已落盘)
      2. 定位走 ui.delete_code_flow: 读码找行 → 点击后紫色选中态验证,
         验证失败绝不发 Del; 全程无数字键入
      3. Del 后复读同码: 码消失 = deleted; 仍在 = delete_failed
      4. 每次按键前断言前台 ∈ hexin*, 用户中途回来抛 ForegroundLostError,
         剩余码标记 user_returned_abort 整体中止
    单码失败不中断后续码 (ForegroundLostError 除外).
    """
    import time

    from scripts import _ths_ui as ui

    def log(msg: str) -> None:
        print(f"[guard] {msg}", flush=True)

    if not ui.ensure_idle(what="自选股删除"):
        return {c: "user_active_skip" for c in codes}

    results: dict[str, str] = {}
    remaining = list(codes)
    try:
        win = ui.ensure_watchlist_window()
    except Exception as exc:
        log(f"自选股窗口不可用: {exc}")
        return {c: "window_lost" for c in codes}

    try:
        while remaining:
            code = remaining[0]
            try:
                win = ui.activate_watchlist()  # 每码重激活, 防前码弹窗抢焦点
                results[code] = ui.delete_code_flow(win, code, log)
                log(f"{code} {results[code]}")
                remaining.pop(0)
                time.sleep(1.0)  # 码间 settle
            except ui.ForegroundLostError:
                raise
            except Exception as exc:
                log(f"{code} 失败: {exc}")
                results[code] = "not_found_or_failed"
                remaining.pop(0)
                time.sleep(1.0)
    except ui.ForegroundLostError as exc:
        log(f"前台丢失, 中止剩余码: {exc}")
        for c in remaining:
            results.setdefault(c, "user_returned_abort")
    finally:
        d = ui.find_window("复制识别")
        if d is not None:
            try:
                ui.close_x(d)
            except Exception:
                pass
    return results


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else None
    if date is None:
        date = newest_panel_date().replace("-", "")
    date_iso = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    df = compute_flags(date_iso)
    if df is None:
        print(f"[guard] {date_iso} 面板无数据, 跳过")
        return 0

    members = load_membership(STOCK_LIST_DIR, date)
    ev = df[df["event_flag"]].copy()
    ev["in_watchlist"] = ev["symbol"].isin(members)
    ev["action"] = np.where(ev["in_watchlist"], "删除", "不在自选股")

    out = STOCK_LIST_DIR / f"ths_flush_removed_{date}__flushguard.csv"
    doc = ev[
        ["symbol", "pctChg", "turnover_rank", "f3_risk", "in_watchlist", "action"]
    ].copy()
    doc.insert(0, "asof", date_iso)
    doc.to_csv(out, index=False, encoding="utf-8-sig")

    to_remove = sorted(ev.loc[ev["in_watchlist"], "symbol"])
    print(f"[guard] {date_iso} 放量下跌 {len(ev)} 只, 自选股成员 {len(members)} 只, "
          f"待删除 {len(to_remove)} 只: {' '.join(to_remove) if to_remove else '-'}")
    print(f"[guard] 文档: {out}")
    if not to_remove:
        return 0

    # UI 删除 (--apply: Del 键流程执行器; 流程待 _diag_ths_del_probe.py 探针证实后
    # 翻开夜间链, 链上步骤保持不带 --apply). 文档已落盘, 删除尽力而为, 失败不非零退出.
    if "--apply" in sys.argv:
        results = delete_codes_from_watchlist(to_remove)
        n_ok = sum(1 for s in results.values() if s == "deleted")
        n_fail = len(results) - n_ok
        for code, status in results.items():
            print(f"[guard] 删除 {code}: {status}")
        if n_fail:
            print(f"[guard] 删除汇总: 成功 {n_ok}/{len(results)}, 未确认/失败 {n_fail} "
                  f"(尽力而为, 不中断夜间链)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
