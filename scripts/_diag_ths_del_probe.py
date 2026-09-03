"""_diag_ths_del_probe.py — 同花顺自选股删除流程探针 (牺牲码 000001, 2026-09-03).

目的: 为「放量下跌标记→删自选股」管线步摸清 Del 键删除流程.
流程 (全程截图留证, 不碰现有名单成员):
  1. 激活自选股窗口, 截图基线
  2. 经 复制识别 对话框加入牺牲码 000001 (复用 ths_push 已验证加路径)
  3. 键入 000001 + Enter 定位 (自选股内已存在, 安全)
  4. 发 Del 键 → 截图: 有确认对话框? 还是立即删除?
  5. 若有对话框 → 截图后按确认 (牺牲码本就该删); 若立即删除 → 直接验证
  6. 终态截图 + 找残留对话框并关闭

用法: python scripts/_diag_ths_del_probe.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SHOT_DIR = Path(__file__).resolve().parent.parent / "tmp_t"
SACRIFICIAL = "000001"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    import ctypes

    import psutil
    import uiautomation as auto

    from scripts._ths_watchlist_push import THS_HEXIN_PATH, push_via_ths

    def log(msg: str) -> None:
        print(f"[probe] {msg}", flush=True)

    def hexin_pids() -> set[int]:
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

    def snap(name: str) -> None:
        from PIL import ImageGrab

        p = SHOT_DIR / f"ths_del_probe_{name}.png"
        ImageGrab.grab().save(str(p))
        log(f"截图 {p.name}")

    # 1. 拉起/激活客户端
    win = find_window("自选股")
    if win is None:
        # 客户端已运行但自选股窗口未开 (主窗口停在行情页) → 激活主窗口按 F6 呼出
        pids = hexin_pids()
        main_win = None
        for c in auto.GetRootControl().GetChildren():
            try:
                if c.ProcessId in pids and c.Name and "同花顺" in c.Name:
                    main_win = c
                    break
            except Exception:
                pass
        if main_win is not None:
            log("客户端已运行, 自选股窗口未开 → 主窗口 F6 呼出")
            ctypes.windll.user32.ShowWindow(int(main_win.NativeWindowHandle), 9)
            time.sleep(0.5)
            main_win.SetActive()
            time.sleep(0.5)
            auto.SendKeys("{F6}")
            for _ in range(15):
                time.sleep(2)
                win = find_window("自选股")
                if win is not None:
                    break
    if win is None:
        if not THS_HEXIN_PATH.exists():
            log(f"客户端不存在: {THS_HEXIN_PATH}")
            return 1
        os.startfile(THS_HEXIN_PATH)
        for _ in range(60):
            time.sleep(2)
            win = find_window("自选股")
            if win is not None:
                break
        if win is None:
            log("120s 未出现自选股窗口")
            return 1
    ctypes.windll.user32.ShowWindow(int(win.NativeWindowHandle), 9)
    time.sleep(0.5)
    win.SetActive()
    time.sleep(0.5)
    snap("01_baseline")

    # 2. 加入牺牲码 (复用已验证加路径)
    tmp_txt = SHOT_DIR / "_ths_del_probe_sacrificial.txt"
    tmp_txt.write_text(SACRIFICIAL + "\n", encoding="utf-8")
    ok = push_via_ths(tmp_txt)
    log(f"加牺牲码 exit={ok}")
    time.sleep(1.5)
    snap("02_added")

    # 3. 键入定位
    win = find_window("自选股")
    if win is None:
        log("窗口丢失")
        return 1
    win.SetActive()
    time.sleep(0.5)
    auto.SendKeys(SACRIFICIAL, interval=0.05)
    time.sleep(1.2)
    snap("03_typed")
    auto.SendKeys("{Enter}")
    time.sleep(1.5)
    snap("04_located")

    # 4. Del 键
    auto.SendKeys("{Delete}")
    time.sleep(1.5)
    snap("05_after_del")

    # 5. 确认对话框探测: 找新弹窗
    dlg_names = ["删除", "提示", "确认", "警告"]
    confirm = None
    for nm in dlg_names:
        confirm = find_window(nm)
        if confirm is not None:
            break
    if confirm is not None:
        log(f"发现确认对话框: {confirm.Name}")
        snap("06_dialog")
        auto.SendKeys("{Enter}")  # 牺牲码, 确认删除
        time.sleep(1.5)
        snap("07_confirmed")
    else:
        log("无确认对话框 (Del 立即删除或未生效)")

    # 6. 终态: 关残留对话框
    for nm in ("复制识别", "删除", "提示"):
        d = find_window(nm)
        if d is not None:
            try:
                r = d.BoundingRectangle
                auto.Click(r.right - 19, r.top + 21)
                time.sleep(0.5)
            except Exception:
                pass
    snap("08_final")
    log("探针完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
