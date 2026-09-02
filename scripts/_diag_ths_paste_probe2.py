"""实测2: 点进自选股表格 → Ctrl+V → End 到底部 → 截图验证."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import uiautomation as auto

TXT = max(
    __import__("glob").glob(
        r"D:\AMINQT\DAILY OPERATION\STOCK LIST\ths_watchlist_*.txt"
    ),
    key=os.path.getmtime,
)
codes = [c for c in open(TXT, encoding="utf-8").read().split() if c]
print(f"codes={codes}")


def hexin_pids():
    import psutil

    return {
        p.info["pid"]
        for p in psutil.process_iter(["name", "pid"])
        if (p.info["name"] or "").lower().startswith("hexin")
    }


pids = hexin_pids()
win = next(
    c
    for c in auto.GetRootControl().GetChildren()
    if c.ProcessId in pids and c.Name and "自选股" in c.Name
)
win.SetActive()
time.sleep(0.5)

r = win.BoundingRectangle
cx = (r.left + r.right) // 2
cy = (r.top + r.bottom) // 2
print(f"rect={r} click=({cx},{cy})")
auto.Click(cx, cy)  # 单击选中行, 不打开个股页
time.sleep(0.5)

auto.SetClipboardText("\n".join(codes))
auto.SendKeys("{Ctrl}v")
time.sleep(2.5)

auto.SendKeys("{End}")
time.sleep(1.0)

from PIL import ImageGrab

img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom))
img.save(r"D:\AMINQT\AMINQT CODES\tmp_t\ths_paste_check2.png")
print("saved check2", img.size)
