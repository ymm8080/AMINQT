"""实测: 自选股页 Ctrl+V 批量粘贴 — 观察弹窗行为 (一次性探查脚本)."""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import uiautomation as auto

TXT = max(
    __import__("glob").glob(r"D:\AMINQT\DAILY OPERATION\STOCK LIST\ths_watchlist_*.txt"),
    key=os.path.getmtime,
)
codes = [c for c in open(TXT, encoding="utf-8").read().split() if c]
print(f"txt={os.path.basename(TXT)} codes={len(codes)}")


def hexin_pids() -> set[int]:
    import psutil

    return {
        p.info["pid"]
        for p in psutil.process_iter(["name", "pid"])
        if (p.info["name"] or "").lower().startswith("hexin")
    }


pids = hexin_pids()
win = None
for c in auto.GetRootControl().GetChildren():
    try:
        if c.ProcessId in pids and "自选股" in (c.Name or ""):
            win = c
    except Exception:
        pass
if win is None:
    print("ABORT: 找不到自选股窗口 (客户端可能未开或不在自选股页)")
    sys.exit(1)

print(f"window: {win.Name!r}")
import ctypes

ctypes.windll.user32.ShowWindow(int(win.NativeWindowHandle), 9)  # SW_RESTORE
time.sleep(0.5)
win.SetActive()
time.sleep(0.5)
print(f"activated: {win.Name!r}")


def snap(tag):
    kids = []
    for c in auto.GetRootControl().GetChildren():
        try:
            if c.ProcessId in pids:
                kids.append(f"{c.ControlTypeName} {c.Name!r} {c.ClassName!r}")
        except Exception:
            pass
    print(f"[{tag}] hexin顶层: {kids}")


snap("before")
auto.SetClipboardText("\n".join(codes))
auto.SendKeys("{Ctrl}v")
time.sleep(2.0)
snap("after+2s")

# 若出现对话框, 深挖一层看结构
for c in auto.GetRootControl().GetChildren():
    try:
        if c.ProcessId not in pids or c.ClassName in ("#32770",) or "自选股" not in (c.Name or ""):
            if c.ProcessId in pids and c.Name and "同花顺(9" not in c.Name:
                print(f"=== 对话框候选: {c.ControlTypeName} {c.Name!r} {c.ClassName!r}")
                for ch in c.GetChildren():
                    print(f"    {ch.ControlTypeName} {ch.Name!r} {ch.ClassName!r}")
    except Exception:
        pass
print("done")
