"""同花顺 hexin.exe UIA 树探查 (被动只读, 维护驱动脚本时用).

用法: python scripts/_diag_ths_uia_probe.py [深度, 默认4]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import uiautomation as auto


def main() -> int:
    max_depth = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    tops = []
    for c in auto.GetRootControl().GetChildren():
        try:
            if c.ProcessId in _hexin_pid_set():
                tops.append(c)
        except Exception:
            pass
    print(f"hexin 顶层窗口 {len(tops)} 个")
    for w in tops:
        print(f"\n=== top: type={w.ControlTypeName} name={w.Name!r} "
              f"class={w.ClassName!r} pid={w.ProcessId} rect={w.BoundingRectangle}")
        _walk(w, 1, max_depth)
    return 0


def _hexin_pid_set() -> set[int]:
    import psutil

    return {p.info["pid"] for p in psutil.process_iter(["name", "pid"])
            if p.info["name"] and p.info["name"].lower().startswith("hexin")}


def _walk(ctrl, depth: int, max_depth: int) -> None:
    if depth > max_depth:
        return
    try:
        children = ctrl.GetChildren()
    except Exception:
        return
    for ch in children:
        try:
            print("  " * depth +
                  f"{ch.ControlTypeName} name={ch.Name!r} class={ch.ClassName!r} "
                  f"auto_id={ch.AutomationId!r}")
        except Exception:
            continue
        _walk(ch, depth + 1, max_depth)


if __name__ == "__main__":
    sys.exit(main())
