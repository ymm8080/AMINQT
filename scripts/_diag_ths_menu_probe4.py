"""实测4: 右键自选股表格 → 截图看上下文菜单 → ESC 关闭."""
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
cx, cy = (r.left + r.right) // 2, (r.top + r.bottom) // 2
auto.SetCursorPos(cx, cy)
time.sleep(0.2)
auto.RightClick(cx, cy)
time.sleep(1.0)

from PIL import ImageGrab

ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom)).save(
    r"D:\AMINQT\AMINQT CODES\tmp_t\ths_context_menu.png"
)
print("menu screenshot saved")
auto.SendKeys("{Esc}")
