"""同花顺自选股推送: 当日 TOP10 → 同花顺可导入 txt → UI 自动化导入 PC 客户端.

TOP10 口径 (2026-09-02 用户: LEGACY AND PARALLEL 都要推): 双源各取前 10 并集,
parallel 序在前, 逐码去重; 单侧缺失时退化为另一侧前 10 (parallel 交付失败回退).
数据源 (STOCK LIST 目录, 当日):
  parallel_shortlist_{date}__{module}.csv    并行短名单 → rank 列升序前 10
  legacy_stocklist_{date}__{module}.csv      legacy 清单 → 清单序前 10
生成: ths_watchlist_{date}__{module}.txt  每行一个 6 位代码 (同花顺自动识别市场).
双侧 module 标签相同则共用一个文件名, 当日重推会覆盖旧 txt (推送历史以链日志为准).

导入: uiautomation 驱动 hexin.exe (同花顺 PC 客户端). 客户端需已登录同账号 —
导入进自选股后云同步到手机 App. --gen-only 只生成文件不动客户端.

用法: python scripts/_ths_watchlist_push.py [YYYYMMDD] [--gen-only] [--dry-run]
"""

import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob as _glob

from config.settings import STOCK_LIST_DIR
from scripts._ths_ui import THS_HEXIN_PATH  # 单一来源; 本模块再导出兼容旧引用

CODE_RE = re.compile(r"^\d{6}$")


def _newest(pattern: str, list_dir) -> str | None:
    hits = _glob.glob(str(list_dir / pattern))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _module_of(path: str) -> str:
    return os.path.basename(path).split("__", 1)[1][: -len(".csv")]


def _valid_codes(symbols) -> list[str]:
    out: list[str] = []
    for sym in symbols.dropna().astype(str):
        sym = sym.strip()
        if CODE_RE.match(sym) and sym not in out:
            out.append(sym)
    return out


def collect_codes(
    date: str, list_dir=STOCK_LIST_DIR, top_n: int = 10
) -> tuple[str, list[str]]:
    """当日双源 TOP10 并集 → (module, 代码). parallel 前 legacy 后, 逐码去重保序."""
    import pandas as pd

    codes: list[str] = []
    tags: list[str] = []

    parallel = _newest(f"parallel_shortlist_{date}__*.csv", list_dir)
    if parallel is not None:
        df = pd.read_csv(parallel, dtype={"symbol": str})
        if "rank" in df.columns:
            df = df.dropna(subset=["rank"]).sort_values("rank")
        tags.append(_module_of(parallel))
        codes += _valid_codes(df["symbol"])[:top_n]

    legacy = _newest(f"legacy_stocklist_{date}__*.csv", list_dir)
    if legacy is not None:
        df = pd.read_csv(legacy, dtype={"symbol": str})
        tags.append(_module_of(legacy))
        codes += _valid_codes(df["symbol"])[:top_n]

    if not tags:
        raise SystemExit(f"无清单: parallel/legacy_stocklist_{date}__*.csv")

    seen: set[str] = set()
    deduped = [c for c in codes if not (c in seen or seen.add(c))]
    module = "__".join(dict.fromkeys(tags))
    return module, deduped


def ths_txt_path(date: str, module: str, list_dir=STOCK_LIST_DIR):
    return list_dir / f"ths_watchlist_{date}__{module}.txt"


def write_ths_txt(codes: list[str], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(codes) + "\n")


# 冲刷串: 无字母无数字 — 识别器无法解析成任何代码行 (旧 "ths-push" 被识别成
# THS/PUSH 两只美股混进列表, 09-03 截图实证)
FLUSH_STR = "-----"
# 对话框列表区可视 ~16 行 (32px 行距), 12/批留裕量防滚动截断行数核验
CHUNK_SIZE = 12


def _dlg_rows_from_img(img: np.ndarray) -> list[tuple[int, bool]]:
    """对话框区域截图 (x 从对话框左缘起, y 从表头下缘起) → [(行中心y, 是否已勾)].

    行 = 文字区深色带 (高≥10px, x≥33 避开勾选框列); 勾选框内部取 x 17..28
    (校准: 框体 rel 15..29, 边框 725/739), 内部 min 灰度 <80 = 有勾墨迹,
    ≥80 = 空框 (09-03 实测勾 53/54 vs 空 98). 纯数组函数, 可单测.
    """
    text = img[:, 33:]
    rowfrac = (text.mean(axis=2) < 120).mean(axis=1)
    bands = []
    s = None
    for y, on in enumerate(rowfrac):
        if on and s is None:
            s = y
        elif not on and s is not None:
            if y - s >= 10:
                bands.append((s, y))
            s = None
    if s is not None and len(rowfrac) - s >= 10:
        bands.append((s, len(rowfrac)))
    rows = []
    for ya, yb in bands:
        yc = (ya + yb) // 2
        if yc < 7 or yc + 7 > img.shape[0]:
            continue
        cell = img[yc - 7 : yc + 7, 17:28].mean(axis=2)
        rows.append((yc, bool(cell.min() < 80)))
    return rows


def _ensure_dialog_rows_checked(
    dlg, expected: int, log=print, max_rounds: int = 4
) -> bool:
    """核验对话框列表恰好 expected 行且全部勾选, 未勾行逐个点选. True=可点加入.

    行数不符 (识别未齐/杂行混入) 只重查不强点 — 加入是外向动作, 宁可整批失败.
    """
    import time

    import uiautomation as auto
    from PIL import ImageGrab

    from scripts import _ths_ui as ui

    r = dlg.BoundingRectangle
    y0 = r.top + 62  # 表头以下
    y1 = r.bottom - 58  # 按钮行以上
    for rnd in range(max_rounds):
        if rnd:
            time.sleep(1.0)
        img = np.asarray(ImageGrab.grab(bbox=(r.left, y0, r.right - 10, y1)))
        rows = _dlg_rows_from_img(img)
        if len(rows) != expected:
            log(f"[ths] 对话框行数 {len(rows)} != 期望 {expected} (round {rnd + 1})")
            continue
        unchecked = [y0 + yc for yc, ck in rows if not ck]
        if not unchecked:
            return True
        for ay in unchecked:
            ui.assert_foreground_hexin(f"勾选对话框行 y={ay}")
            auto.Click(r.left + 22, ay)
            time.sleep(0.5)
    return False


def push_via_ths(txt_path, dry_run: bool = False) -> bool:
    """UI 自动化: 激活同花顺客户端, 经 工具→复制识别 对话框把代码批量加入自选股.

    实测流程 (2026-09-01, hexin 9.60.20; 09-03 勾选语义修正):
      1. 客户端未运行则拉起 (记住密码自动登录, 直达自选股页)
      2. 关闭残留的复制识别对话框 — 必须空状态开工
      3. 工具→复制识别 经窗口相对校准坐标点击 (UIA 菜单矩形错位不可用,
         09-03 实证; 下拉展开做亮像素确认, 未展开不盲点)
      4. 对话框 SetWindowPos 到 (710,0) 保证按钮可见
      5. 剪贴板先写无字母数字冲刷串 (旧 "ths-push" 会被识别成 THS/PUSH 两只
         美股, 09-03 实证), 再分批写代码串; 监听有 ~2.5s 延迟+同码去重
      6. **点「加入自选股」前必须逐行验证勾选**: 09-03 实证按钮只加勾选行
         (旧行为"不看勾选状态"已失效) — 识别出的行可能未勾, 直接点击=静默
         漏加. 分批 ≤12 只/批, 防对话框列表区滚动截断行数核验
      7. 关闭对话框. 无成功/失败回执 (成功 toast 对部分加也报成功), 无人值守
         判据 = 行数+全勾核验通过

    安全闸 (2026-09-03 误删事故后加, 见 scripts/_ths_ui.py 模块头):
      入口空闲闸 (用户在场直接 return False) + 每次点击前前台 hexin 断言.
    """
    if dry_run:
        print(f"[dry] 将导入同花顺: {txt_path}")
        return True

    import ctypes
    import time

    import uiautomation as auto

    from scripts import _ths_ui as ui

    codes = [c for c in Path(txt_path).read_text(encoding="utf-8").split() if c]
    if not codes:
        print("[ths] txt 为空, 跳过")
        return True

    if not ui.ensure_idle(what="自选股推送"):
        return False

    try:
        # 共享窗口原语: 现存自选股窗 → 主窗 F6 呼出 (客户端在跑但停在别的页,
        # 如上次对话框流程留下的分时图页, 09-03 实证) → 冷启动兜底
        win = ui.ensure_watchlist_window()
    except ui.ForegroundLostError as exc:
        print(f"[ths] {exc}")
        return False
    except Exception as exc:
        print(f"[ths] 自选股窗口不可用: {exc}")
        return False

    def fresh_dialog():
        dlg = ui.find_window("复制识别")
        if dlg is not None:
            ui.close_x(dlg)
            time.sleep(0.5)
        if not ui.open_copy_recognition_dialog(win):
            return None
        dlg = ui.find_window("复制识别")
        if dlg is None:
            return None
        ctypes.windll.user32.SetWindowPos(
            int(dlg.NativeWindowHandle), 0, 710, 0, 0, 0, 0x0001 | 0x0004
        )
        time.sleep(0.5)
        return dlg

    ui.close_stray_windows()
    # 冲刷要在开对话框前: 对话框打开即吸当前剪贴板, 残留代码会在开窗瞬间进列表
    auto.SetClipboardText(FLUSH_STR)
    time.sleep(0.8)

    dlg = fresh_dialog()
    if dlg is None:
        print("[ths] 复制识别对话框未出现 (校准路径失败)")
        return False

    ok = True
    done = 0
    for ci in range(0, len(codes), CHUNK_SIZE):
        chunk = codes[ci : ci + CHUNK_SIZE]
        if ci:
            dlg = fresh_dialog()
            if dlg is None:
                print("[ths] 后续批对话框未出现")
                ok = False
                break
        auto.SetClipboardText("\n".join(chunk))
        time.sleep(2.5 + 0.5 * len(chunk))
        if not _ensure_dialog_rows_checked(dlg, len(chunk)):
            print(f"[ths] 批次 {ci // CHUNK_SIZE + 1}: 行数/勾选核验未过, 不点加入")
            ok = False
            break
        ui.assert_foreground_hexin("点击加入自选股")
        dr = dlg.BoundingRectangle
        auto.Click(dr.right - 205, dr.bottom - 33)  # 加入自选股
        done += len(chunk)
        time.sleep(1.5)

    ui.assert_foreground_hexin("关闭复制识别对话框")
    dlg = ui.find_window("复制识别")
    if dlg is not None:
        ui.close_x(dlg)
    time.sleep(0.5)
    if ok:
        print(f"[ths] 已核验加入自选股 ({done} 只), 云同步稍后到手机")
    return ok


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_only = "--gen-only" in sys.argv
    dry_run = "--dry-run" in sys.argv
    date = args[0] if args else None
    if date is None:
        newest = _newest("legacy_stocklist_*__*.csv", STOCK_LIST_DIR)
        if newest is None:
            raise SystemExit("STOCK LIST 目录无任何 legacy 清单")
        date = re.search(r"legacy_stocklist_(\d{8})__", os.path.basename(newest)).group(
            1
        )

    module, codes = collect_codes(date)
    if not codes:
        print(f"[warn] {date} 清单为空, 不推送")
        return 0
    out = ths_txt_path(date, module)
    write_ths_txt(codes, out)
    print(f"[ths] {out} ({len(codes)} 只, module {module})")

    if not gen_only:
        if not THS_HEXIN_PATH.exists():
            print(f"[warn] 同花顺客户端不存在: {THS_HEXIN_PATH}")
            return 0
        ok = push_via_ths(out, dry_run)
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
