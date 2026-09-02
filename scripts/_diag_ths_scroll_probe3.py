"""实测3: 滚轮滚到自选股底部 → 截图."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import uiautomation as auto


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
time.sleep(0.3)
r = win.BoundingRectangle
auto.SetCursorPos((r.left + r.right) // 2, (r.top + r.bottom) // 2)
time.sleep(0.2)
for _ in range(15):
    auto.WheelDown()
    time.sleep(0.08)
time.sleep(0.5)

from PIL import ImageGrab

img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom))
img.save(r"D:\AMINQT\AMINQT CODES\tmp_t\ths_scroll_bottom.png")
print("saved", img.size)
