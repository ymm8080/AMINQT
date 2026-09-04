"""同花顺 UI 自动化共享助手: 空闲闸/前台断言/窗口管理/UIA 菜单点击/高亮行检测.

2026-09-03 误删事故教训: 用户在场时 Windows 前台锁让 SetActive 失效, SendKeys
会串进用户正在用的窗口 (实测键入进了 Chrome, Del 删掉真实自选股成员 688118).
因此任何键序动作必须过两道闸:
  1. 空闲闸 ensure_idle(): GetLastInputInfo 系统级输入空闲 >= IDLE_MIN_S
  2. 前台断言 assert_foreground_hexin(): 每次按键/点击前复查, 用户中途回来
     立即抛 ForegroundLostError, 调用方必须整体中止, 不得续发剩余按键

自选股网格是 MFC 自绘 (UIA 读不到行文本), 行级验证走截图. **定位 = 截图读码**:
网格代码列渲染有 6 位代码 (09-03 实证), 模板匹配读出每行代码后直接点击目标行.
选中行背景为纯紫 RGB(64,0,128) (09-03 探针实测, 未选中行是灰底 RGB(45,45,45)).

键入定位已判死 (09-03 定案): 即便网格有键盘焦点 ({Down} 能移动高亮行), 数字键
仍被主窗全局代码入口截走, Enter 直接跳转该股分时图 — 任何"键入代码定位"路线
都不可用, 删除流程只允许读码+点击.

菜单栏是真实 MenuBarControl 但 UIA 矩形错位约两项 (点"工具"实命中"机会"), 真实
菜单点击只能走窗口相对校准坐标 (open_copy_recognition_dialog, 09-03 实证).

hexin 窗口模型 (09-03 实证): 只有一个主窗 (F6/页面切换都是同 hwnd 换标题,
"同花顺 - 自选股" ↔ "同花顺(9.60.20) - A股分时走势").
"""

from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

IDLE_MIN_S = 90.0  # 用户离开键盘/鼠标至少 90s 才允许键序动作

THS_HEXIN_PATH = Path(os.getenv("THS_HEXIN_PATH", r"C:\同花顺软件\同花顺\hexin.exe"))

# 选中行高亮色 RGB(64,0,128) 容差 (09-03 截图校准)
_HL_R = (40, 100)
_HL_G = (0, 40)
_HL_B = (95, 170)


class ForegroundLostError(RuntimeError):
    """前台窗口不属于 hexin* — 用户可能回来了, 必须立即停止一切按键."""


def user_idle_seconds() -> float:
    """系统级键盘/鼠标空闲秒数 (GetLastInputInfo, GetTickCount64 防回绕)."""

    class _LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    lii = _LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        return 0.0
    now64 = ctypes.windll.kernel32.GetTickCount64()
    return max(0.0, ((now64 - lii.dwTime) & 0xFFFFFFFF) / 1000.0)


def ensure_idle(min_idle_s: float = IDLE_MIN_S, what: str = "UI 自动化") -> bool:
    """空闲闸: 不满足返回 False (调用方跳过本次动作, 打日志)."""
    idle = user_idle_seconds()
    if idle >= min_idle_s:
        return True
    print(
        f"[ths-ui] {what} 跳过: 用户在场 (输入空闲 {idle:.0f}s < {min_idle_s:.0f}s)",
        flush=True,
    )
    return False


def hexin_pids() -> set[int]:
    """全量 hexin* 进程 pid (首个匹配常是 hexinhelper 辅助进程, 必须用全集)."""
    import psutil

    return {
        p.info["pid"]
        for p in psutil.process_iter(["name", "pid"])
        if (p.info["name"] or "").lower().startswith("hexin")
    }


def find_window(substr: str):
    """按标题子串找 hexin 顶层窗口, 无则 None."""
    import uiautomation as auto

    pids = hexin_pids()
    for c in auto.GetRootControl().GetChildren():
        try:
            if c.ProcessId in pids and c.Name and substr in c.Name:
                return c
        except Exception:
            pass
    return None


def foreground_pid() -> int:
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return 0
    pid = wintypes.DWORD(0)
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def assert_foreground_hexin(what: str = "按键") -> None:
    pid = foreground_pid()
    if pid not in hexin_pids():
        raise ForegroundLostError(f"{what} 前台 pid={pid} 不属于 hexin*")


def activate_window(win) -> None:
    """激活窗口并断言前台已属于 hexin* (SetActive 被前台锁拒绝时此处即失败)."""
    ctypes.windll.user32.ShowWindow(int(win.NativeWindowHandle), 9)  # SW_RESTORE
    time.sleep(0.4)
    win.SetActive()
    time.sleep(0.4)
    assert_foreground_hexin("激活自选股窗口")


def ensure_watchlist_window():
    """找自选股窗口: 现存 → 主窗 F6 呼出 → 冷启动. 全程断言前台, 失败 raise.

    调用方必须先过空闲闸 (F6/冷启动会发键或抢前台).
    """
    win = find_window("自选股")
    if win is None:
        main_win = None
        for c in _hexin_top_windows():
            if c.Name and "自选股" not in c.Name:
                main_win = c
                break
        if main_win is not None:
            activate_window(main_win)
            import uiautomation as auto

            auto.SendKeys("{F6}")
            for _ in range(15):
                time.sleep(2)
                win = find_window("自选股")
                if win is not None:
                    break
    if win is None:
        if not THS_HEXIN_PATH.exists():
            raise RuntimeError(f"客户端不存在: {THS_HEXIN_PATH}")
        os.startfile(THS_HEXIN_PATH)
        for _ in range(60):
            time.sleep(2)
            win = find_window("自选股")
            if win is not None:
                break
    if win is None:
        raise RuntimeError("120s 内未出现自选股窗口 (可能停在登录页)")
    activate_window(win)
    return win


def activate_watchlist():
    """轻量版: 只找现存自选股窗口并激活 (不 F6 不冷启动), 供逐码循环用."""
    win = find_window("自选股")
    if win is None:
        raise RuntimeError("自选股窗口不存在")
    activate_window(win)
    return win


def _hexin_top_windows() -> list:
    import uiautomation as auto

    pids = hexin_pids()
    out = []
    for c in auto.GetRootControl().GetChildren():
        try:
            if c.ProcessId in pids and c.Name:
                out.append(c)
        except Exception:
            pass
    return out


def click_menu_bar_item(win, name: str) -> None:
    """按名点击主窗菜单栏项 — 已废弃的 UIA 点击 (矩形错位, 见下) 保留签名防误用.

    09-03 实证: UIA MenuBarControl 项矩形比可视菜单左移约两项, item.Click()
    点"工具"实际命中"机会". 禁止用此函数做真实点击; 复制识别路径用
    open_copy_recognition_dialog() (窗口相对校准坐标+截图锚定).
    """
    raise RuntimeError(
        "click_menu_bar_item 已废弃: UIA 菜单矩形错位 (09-03 实证), "
        "用 open_copy_recognition_dialog() 校准路径"
    )


# 09-03 校准 (窗 rect left=33, top=33 时实测): 工具菜单打开点 = 窗相对 (357,9),
# 弹出的下拉里 复制识别 = 窗相对 (417,258). 窗口移动时相对偏移不变.
_MENU_TOOL_REL = (357, 9)
_MENU_COPY_RECOG_REL = (417, 258)
# 下拉展开确认: 菜单体浅色底 (~RGB 240) 在网格深底上, 取 (450,200) 邻域亮像素占比
_MENU_OPEN_CHECK_REL = (450, 200)


def open_copy_recognition_dialog(win) -> bool:
    """点 工具→复制识别 打开对话框 (校准坐标+截图锚定). True=对话框已弹出.

    前置: 调用方已过空闲闸且 win 已激活前台. 每次点击前断言前台 hexin.
    """
    import uiautomation as auto
    from PIL import ImageGrab

    r = win.BoundingRectangle

    def rel_click(rx: int, ry: int) -> None:
        ui_x = r.left + rx
        ui_y = r.top + ry
        assert_foreground_hexin(f"点击窗相对 ({rx},{ry})")
        auto.Click(ui_x, ui_y)

    rel_click(*_MENU_TOOL_REL)
    time.sleep(1.2)

    # 下拉展开确认: (450,200) 邻域 40x20 亮像素占比 (菜单浅底 vs 网格深底)
    cx = r.left + _MENU_OPEN_CHECK_REL[0]
    cy = r.top + _MENU_OPEN_CHECK_REL[1]
    patch = np.asarray(ImageGrab.grab(bbox=(cx - 20, cy - 10, cx + 20, cy + 10)))
    light_frac = float((patch.mean(axis=2) > 180).mean())
    if light_frac < 0.5:
        print(f"[ths-ui] 工具下拉未展开 (亮像素占比 {light_frac:.0%}), 不盲点菜单项")
        auto.SendKeys("{Esc}")
        return False

    rel_click(*_MENU_COPY_RECOG_REL)
    time.sleep(1.5)
    dlg = find_window("复制识别")
    if dlg is None:
        print("[ths-ui] 复制识别对话框未出现 (下拉点击后未找到对话框)")
        return False
    return True


def close_stray_windows() -> list[str]:
    """关掉误开的 THS 杂窗 (形态选股 教程窗/方案侧栏 等), WM_CLOSE 不死者 SW_HIDE.

    必须走 Win32 EnumWindows 而非 UIA: 09-03 实证这些 MFC 弹窗对 UIA 根遍历
    不可见 (close_stray_windows 的 UIA 版每次空转, 教程窗连开三轮盖住网格).
    不需要前台/焦点, 用户在场也安全. 返回处理过的窗口标题.
    """
    user32 = ctypes.windll.user32
    rows: list[tuple[int, int, str]] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            rows.append((int(hwnd), int(pid.value), buf.value))
        return True

    user32.EnumWindows(EnumWindowsProc(_cb), 0)
    pids = hexin_pids()

    handled: list[str] = []
    for hwnd, pid, title in rows:
        if pid not in pids or not title or "自选股" in title:
            continue
        if "形态选股" not in title and title != "同花顺":
            continue
        user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        handled.append(title)
    if handled:
        time.sleep(1.0)
        # WM_CLOSE 存活者 (如 形态选股方案 侧栏, 09-03 实证不理 WM_CLOSE) → SW_HIDE
        for hwnd, _pid, title in rows:
            if title in handled and user32.IsWindowVisible(hwnd):
                user32.ShowWindow(hwnd, 0)  # SW_HIDE
                print(f"[ths-ui] WM_CLOSE 无效, 已 SW_HIDE: {title!r}")
    return handled


def popup_menu_item(name: str, timeout_s: float = 5.0) -> bool:
    """已废弃: MFC 自绘下拉对 UIA 不可见 (无 #32768 顶层窗), 此路恒 False."""
    raise RuntimeError(
        "popup_menu_item 已废弃: 自绘下拉 UIA 不可见 (09-03 实证), "
        "用 open_copy_recognition_dialog() 校准路径"
    )


def close_x(dlg) -> None:
    """点对话框右上角 ✕ (right-19, top+21, 09-01 实测锚点)."""
    import uiautomation as auto

    r = dlg.BoundingRectangle
    auto.Click(r.right - 19, r.top + 21)
    time.sleep(0.8)


def is_highlight_color(img: np.ndarray) -> np.ndarray:
    """紫色高亮掩码 (H,W bool), 判色容差见模块头."""
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    return (
        (r >= _HL_R[0])
        & (r <= _HL_R[1])
        & (g >= _HL_G[0])
        & (g <= _HL_G[1])
        & (b >= _HL_B[0])
        & (b <= _HL_B[1])
    )


def detect_selected_rows(img: np.ndarray, min_lines: int = 10):
    """截图网格竖切图里的紫色高亮波段 [(y0,y1)], 按行线高亮占比>=0.5 判定."""
    frac = is_highlight_color(img).mean(axis=1)
    hit = frac >= 0.5
    bands: list[tuple[int, int]] = []
    start = None
    for y, h in enumerate(hit):
        if h and start is None:
            start = y
        elif not h and start is not None:
            if y - start >= min_lines:
                bands.append((start, y))
            start = None
    if start is not None and len(hit) - start >= min_lines:
        bands.append((start, len(hit)))
    return bands


# ---- 截图读码定位 (09-03 定案: 键入定位判死后的唯一可靠路线) ----

# 代码列窗口相对坐标 (物理 px; 09-03 列剖面实测窗 left=33 时首位数字起绝对 x=485,
# 6 位 ×~10px 步距止 ~543, 裁剪取 483..545 留边)
_CODE_COL_REL = (450, 512)
# 行扫描顶 (表头下方首个数据行) 与底部预留 (页签/状态栏) 的窗口相对偏移
_ROWS_SCAN_TOP_REL = 145
_BOTTOM_KEEP_REL = 106

_THS_TEMPLATES_NPZ = Path(__file__).with_name("_ths_digit_templates.npz")
# 单字匹配距离上限 (留一自洽 94/95, 唯一误配距离 0.083 → 0.15 足够分离)
_DIGIT_MATCH_MAX_D = 0.15

_digit_tpl_cache = None


def _digit_templates():
    global _digit_tpl_cache
    if _digit_tpl_cache is None:
        z = np.load(_THS_TEMPLATES_NPZ)
        _digit_tpl_cache = (z["feats"], z["labels"])
    return _digit_tpl_cache


def _norm_glyph(g: np.ndarray) -> np.ndarray:
    """二值字形 (H,W bool) → 定尺寸 (16,12) float 特征."""
    from PIL import Image

    im = Image.fromarray(g.astype(np.uint8) * 255).resize((12, 16), Image.LANCZOS)
    return (np.asarray(im) > 100).astype(np.float32)


def _match_digit(feat: np.ndarray) -> tuple[int | None, float]:
    feats, labels = _digit_templates()
    d = np.abs(feats - feat[None]).mean(axis=(1, 2))
    i = int(d.argmin())
    return (int(labels[i]) if d[i] <= _DIGIT_MATCH_MAX_D else None), float(d[i])


def _split_digit_cells(seg: np.ndarray) -> list[tuple[int, int]]:
    """行字形二值图按空列切分数字格; <3px 碎片组丢弃 (真数字宽>=4), 触连宽组按中位宽均分."""
    colhit = seg.any(axis=0)
    groups: list[tuple[int, int]] = []
    s = None
    for x, on in enumerate(colhit):
        if on and s is None:
            s = x
        elif not on and s is not None:
            groups.append((s, x))
            s = None
    if s is not None:
        groups.append((s, len(colhit)))
    groups = [(a, b) for a, b in groups if b - a >= 3]
    if not groups:
        return []
    widths = [b - a for a, b in groups]
    med = float(np.median(widths))
    cells = []
    for a, b in groups:
        w = b - a
        k = max(1, round(w / max(med, 1e-6)))
        if k >= 2 and w >= 1.6 * med:
            step = w / k
            for i in range(k):
                cells.append((a + int(i * step), a + int((i + 1) * step)))
        else:
            cells.append((a, b))
    return cells


def _read_rows_from_gray(
    gray: np.ndarray, ytop: int
) -> list[tuple[int, int, str | None, float]]:
    """读码核心 (纯数组): 代码列灰度图 → [(y0, y1, code, conf)], y 已加 ytop 偏移."""
    on = (gray > 140).mean(axis=1) > 0.02
    bands = []
    s = None
    for y, h in enumerate(on):
        if h and s is None:
            s = y
        elif not h and s is not None:
            if y - s >= 8:
                bands.append((s, y))
            s = None
    if s is not None and len(on) - s >= 8:
        bands.append((s, len(on)))
    rows = []
    for ya, yb in bands:
        seg = gray[ya:yb] > 140
        rf = seg.mean(axis=1)
        ys = np.where(rf > 0.05)[0]
        if len(ys) == 0:
            rows.append((ya + ytop, yb + ytop, None, 1.0))
            continue
        seg2 = seg[ys.min() : ys.max() + 1]
        cells = _split_digit_cells(seg2)
        if len(cells) != 6:
            rows.append((ya + ytop, yb + ytop, None, 1.0))
            continue
        digits, conf = [], 0.0
        for ca, cb in cells:
            g = seg2[:, ca:cb]
            ys2 = np.where(g.any(axis=1))[0]
            xs2 = np.where(g.any(axis=0))[0]
            feat = _norm_glyph(g[ys2.min() : ys2.max() + 1, xs2.min() : xs2.max() + 1])
            lab, dist = _match_digit(feat)
            digits.append(lab)
            conf = max(conf, dist)
        code = (
            "".join(str(d) for d in digits)
            if all(d is not None for d in digits)
            else None
        )
        rows.append((ya + ytop, yb + ytop, code, conf))
    return rows


def read_visible_rows(win, log=print) -> list[tuple[int, int, str | None, float]]:
    """读当前可见行代码列 → [(y0_rel, y1_rel, code, conf)] (窗口相对物理 px).

    code=None = 该行切分/匹配失败 (宁缺勿错, 调用方不得猜测); conf = 6 格最大
    匹配距离. 只截图不点击, 用户在场也安全.
    """
    from PIL import ImageGrab

    r = win.BoundingRectangle
    x0 = r.left + _CODE_COL_REL[0]
    x1 = r.left + _CODE_COL_REL[1]
    ytop = r.top + _ROWS_SCAN_TOP_REL
    ybot = r.top + r.height() - _BOTTOM_KEEP_REL
    gray = np.asarray(ImageGrab.grab(bbox=(x0, ytop, x1, ybot)).convert("L"))
    return _read_rows_from_gray(gray, ytop - r.top)


def find_row_by_code(win, code: str, rows=None) -> float | None:
    """在可见行里找代码 → 行带中心 (窗口相对 y) / None."""
    if rows is None:
        rows = read_visible_rows(win)
    for ya, yb, c, _conf in rows:
        if c is not None and c == code:
            return (ya + yb) / 2.0
    return None


def row_is_selected(win, rel_y: float) -> bool:
    """目标行 (窗口相对 y) 是否处于紫色选中态."""
    from PIL import ImageGrab

    r = win.BoundingRectangle
    x0 = r.left + _CODE_COL_REL[0]
    x1 = r.left + _CODE_COL_REL[1]
    y = r.top + int(rel_y)
    strip = np.asarray(ImageGrab.grab(bbox=(x0, y - 9, x1, y + 9)))[:, :, :3]
    return float(is_highlight_color(strip).mean()) >= 0.25


def select_row(win, rel_y: float, log=print) -> bool:
    """点击目标行并验证紫色选中态 (选中验证失败绝不允许后续 Del)."""
    import uiautomation as auto

    r = win.BoundingRectangle
    assert_foreground_hexin(f"点击代码行 y_rel={rel_y:.0f}")
    auto.Click(r.left + (_CODE_COL_REL[0] + _CODE_COL_REL[1]) // 2, r.top + int(rel_y))
    time.sleep(0.8)
    if not row_is_selected(win, rel_y):
        log(f"[ths-ui] 点击后行未选中 (y_rel={rel_y:.0f})")
        return False
    return True


def _code_col_center_click(win, rel_y: float) -> None:
    import uiautomation as auto

    r = win.BoundingRectangle
    auto.Click(r.left + (_CODE_COL_REL[0] + _CODE_COL_REL[1]) // 2, r.top + int(rel_y))


def delete_code_flow(win, code: str, log=print) -> str:
    """读码定位+删除单码, 全程无数字键入. 返回:
      deleted        Del 后复读, 码已消失
      not_in_list    顶/尾两视图都没有该码
      locate_failed  找到但点击后选中验证失败 (绝不发 Del)
      delete_failed  Del 后复读码仍在
    尾视图: 焦点点击首行后 {End} 跳尾行 (键盘可达网格, 09-03 实证), 用完 {Home} 回顶.
    """
    import uiautomation as auto

    rows = read_visible_rows(win, log)
    rel_y = find_row_by_code(win, code, rows)
    at_end = False
    if rel_y is None:
        if not rows:
            return "not_in_list"
        assert_foreground_hexin("焦点点击首行 (尾视图补扫)")
        _code_col_center_click(win, (rows[0][0] + rows[0][1]) / 2)
        time.sleep(0.6)
        assert_foreground_hexin("{End} 跳尾视图")
        auto.SendKeys("{End}")
        time.sleep(1.2)
        rel_y = find_row_by_code(win, code)
        at_end = True
        if rel_y is None:
            assert_foreground_hexin("{Home} 回顶")
            auto.SendKeys("{Home}")
            time.sleep(1.0)
            return "not_in_list"

    if not select_row(win, rel_y, log):
        if at_end:
            assert_foreground_hexin("{Home} 回顶")
            auto.SendKeys("{Home}")
            time.sleep(1.0)
        return "locate_failed"

    assert_foreground_hexin(f"Del 删除 {code}")
    auto.SendKeys("{Delete}")
    time.sleep(1.2)
    gone = find_row_by_code(win, code) is None
    if at_end:
        assert_foreground_hexin("{Home} 回顶")
        auto.SendKeys("{Home}")
        time.sleep(1.0)
    return "deleted" if gone else "delete_failed"


def read_all_codes(win, log=print) -> list[str]:
    """全清单读码: 顶视图 + 尾视图 ({End}) 合并去重, 有序. 只读+焦点点击, 无数字键."""
    import uiautomation as auto

    rows = read_visible_rows(win, log)
    codes_top = [c for _a, _b, c, _f in rows if c]
    if not rows:
        return []
    assert_foreground_hexin("焦点点击首行 (全清单读码)")
    _code_col_center_click(win, (rows[0][0] + rows[0][1]) / 2)
    time.sleep(0.6)
    assert_foreground_hexin("{End} 跳尾视图")
    auto.SendKeys("{End}")
    time.sleep(1.2)
    codes_end = [c for _a, _b, c, _f in read_visible_rows(win, log) if c]
    assert_foreground_hexin("{Home} 回顶")
    auto.SendKeys("{Home}")
    time.sleep(1.0)
    return list(dict.fromkeys(codes_top + codes_end))
