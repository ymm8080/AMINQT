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
    """UI 自动化逐码删除自选股成员 (Del 键流程, 来源 _diag_ths_del_probe.py).

    状态: deleted = 确认对话框探到并回车 / dialog_timeout = {Delete} 后未探到
    确认对话框, 删除是否生效未知 / not_found_or_failed = 码不在自选股 (被当成
    新增弹了加自选股对话框, 已关闭) 或单码流程异常. 单码失败不中断后续码.
    """
    import ctypes
    import time

    import psutil
    import uiautomation as auto

    from scripts._ths_watchlist_push import THS_HEXIN_PATH

    def log(msg: str) -> None:
        print(f"[guard] {msg}", flush=True)

    def hexin_pids() -> set[int]:
        # 全量 hexin* 进程: 首个匹配常是 hexinhelper 辅助进程而非主程序
        return {
            p.info["pid"]
            for p in psutil.process_iter(["name", "pid"])
            if (p.info["name"] or "").lower().startswith("hexin")
        }

    def find_window(substr: str):
        pids = hexin_pids()
        for c in auto.GetRootControl().GetChildren():
            try:
                if c.ProcessId in pids and c.Name and substr in c.Name:
                    return c
            except Exception:
                pass
        return None

    def close_x(dlg) -> None:
        r = dlg.BoundingRectangle
        auto.Click(r.right - 19, r.top + 21)  # ✕
        time.sleep(0.8)

    def find_confirm():
        # 假设(未验证): {Delete} 后确认对话框的标题与默认按钮行为以 11:33 探针截图为准
        for nm in ("删除", "提示", "确认", "警告"):
            d = find_window(nm)
            if d is not None:
                return d
        return None

    results: dict[str, str] = {}

    win = find_window("自选股")
    if win is None:
        if not THS_HEXIN_PATH.exists():
            log(f"客户端不存在: {THS_HEXIN_PATH}")
            return {c: "not_found_or_failed" for c in codes}
        log(f"拉起客户端: {THS_HEXIN_PATH}")
        os.startfile(THS_HEXIN_PATH)
        for _ in range(60):
            time.sleep(2)
            win = find_window("自选股")
            if win is not None:
                break
        if win is None:
            log("120s 内未出现自选股窗口 (可能停在登录页)")
            return {c: "not_found_or_failed" for c in codes}

    ctypes.windll.user32.ShowWindow(int(win.NativeWindowHandle), 9)  # SW_RESTORE
    time.sleep(0.5)

    for code in codes:
        try:
            win = find_window("自选股")  # 每码重找激活, 防前码弹窗抢焦点
            if win is None:
                raise RuntimeError("自选股窗口丢失")
            win.SetActive()
            time.sleep(0.5)
            auto.SendKeys(code, interval=0.05)
            time.sleep(1.2)
            add = find_window("复制识别")  # 防御: 码不在列表被当成新增
            if add is not None:
                log(f"{code} 不在自选股 (键入后弹加自选股对话框), 关闭")
                close_x(add)
                results[code] = "not_found_or_failed"
                continue
            auto.SendKeys("{Enter}")  # 定位行
            time.sleep(1.5)
            add = find_window("复制识别")
            if add is not None:
                log(f"{code} 不在自选股 (定位后弹加自选股对话框), 关闭")
                close_x(add)
                results[code] = "not_found_or_failed"
                continue
            auto.SendKeys("{Delete}")
            time.sleep(1.5)
            confirm = find_confirm()
            if confirm is not None:
                auto.SendKeys("{Enter}")  # 默认按钮=确认删除 (待探针证实)
                time.sleep(1.5)
                log(f"{code} deleted (确认对话框: {confirm.Name})")
                results[code] = "deleted"
            else:
                log(f"{code} dialog_timeout (删除键后未探到确认对话框, 生效未知)")
                results[code] = "dialog_timeout"
        except Exception as exc:
            log(f"{code} 失败: {exc}")
            results[code] = "not_found_or_failed"
        time.sleep(1.0)  # 码间 settle

    # 统一清理残留对话框 (右上角 ✕: right-19, top+21)
    for nm in ("复制识别", "删除", "提示"):
        d = find_window(nm)
        if d is not None:
            try:
                close_x(d)
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
