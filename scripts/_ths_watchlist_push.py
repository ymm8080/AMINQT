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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob as _glob

from config.settings import STOCK_LIST_DIR

# 同花顺 PC 客户端路径 (环境变量 THS_HEXIN_PATH 可覆盖)
THS_HEXIN_PATH = Path(os.getenv("THS_HEXIN_PATH", r"C:\同花顺软件\同花顺\hexin.exe"))

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


def collect_codes(date: str, list_dir=STOCK_LIST_DIR, top_n: int = 10) -> tuple[str, list[str]]:
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


def push_via_ths(txt_path, dry_run: bool = False) -> bool:
    """UI 自动化: 激活同花顺客户端, 经 工具→复制识别 对话框把代码批量加入自选股.

    实测流程 (2026-09-01, hexin 9.60.20):
      1. 客户端未运行则拉起 (记住密码自动登录, 直达自选股页)
      2. 关闭残留的复制识别对话框 — 必须空状态开工: 「加入自选股」按钮不按
         勾选状态过滤 (实测未勾选行也会被加入), 只能靠整批干净保证不掺私货
      3. 工具菜单(362,20) → 复制识别(394,262), 逻辑坐标按窗口矩形 1440x741 映射
      4. 对话框 SetWindowPos 到 (710,0) 保证按钮可见
      5. 先写无害串冲刷剪贴板监听, 再写代码串; 监听有 ~2.5s 延迟+同码去重
      6. 点「加入自选股」(锚定对话框右下: 右缘-205, 底缘-33, 与行数无关)
      7. 关闭对话框. 无成功/失败回执, 客户端窗口标题也不带自选股数量 —
         无人值守场景只能以流程走完为判据, 人工抽查兜底
    """
    if dry_run:
        print(f"[dry] 将导入同花顺: {txt_path}")
        return True

    import ctypes
    import time

    import psutil
    import uiautomation as auto

    codes = [c for c in Path(txt_path).read_text(encoding="utf-8").split() if c]
    if not codes:
        print("[ths] txt 为空, 跳过")
        return True

    def hexin_pids() -> set[int]:
        # 必须全量 hexin* 进程: 首个匹配常是 hexinhelper 辅助进程而非主程序
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

    win = find_window("自选股")
    if win is None:
        if not THS_HEXIN_PATH.exists():
            print(f"[ths] 客户端未运行且不存在: {THS_HEXIN_PATH}")
            return False
        print(f"[ths] 拉起客户端: {THS_HEXIN_PATH}")
        os.startfile(THS_HEXIN_PATH)
        for _ in range(60):
            time.sleep(2)
            win = find_window("自选股")
            if win is not None:
                break
        if win is None:
            print("[ths] 120s 内未出现自选股窗口 (可能停在登录页)")
            return False

    ctypes.windll.user32.ShowWindow(int(win.NativeWindowHandle), 9)  # SW_RESTORE
    time.sleep(0.5)
    win.SetActive()
    time.sleep(0.5)

    dlg = find_window("复制识别")
    if dlg is not None:
        dr = dlg.BoundingRectangle
        auto.Click(dr.right - 19, dr.top + 21)  # ✕
        time.sleep(0.8)

    r = win.BoundingRectangle

    def lp(x: int, y: int) -> tuple[int, int]:
        return (r.left + int(x * r.width() / 1440), r.top + int(y * r.height() / 741))

    auto.Click(*lp(362, 20))  # 工具菜单
    time.sleep(1.0)
    auto.Click(*lp(394, 262))  # 复制识别菜单项
    time.sleep(1.5)

    dlg = find_window("复制识别")
    if dlg is None:
        print("[ths] 复制识别对话框未出现")
        return False
    ctypes.windll.user32.SetWindowPos(
        int(dlg.NativeWindowHandle), 0, 710, 0, 0, 0, 0x0001 | 0x0004
    )
    time.sleep(0.5)

    auto.SetClipboardText("ths-push")  # 冲刷: 防开窗前残留剪贴板里的代码被吸入
    time.sleep(1.0)
    auto.SetClipboardText("\n".join(codes))
    time.sleep(2.5 + 0.3 * len(codes))

    dr = dlg.BoundingRectangle
    auto.Click(dr.right - 205, dr.bottom - 33)  # 加入自选股
    time.sleep(2.0)

    dr = dlg.BoundingRectangle
    auto.Click(dr.right - 19, dr.top + 21)  # ✕ 关闭
    time.sleep(0.5)
    print(f"[ths] 已点击加入自选股 ({len(codes)} 只), 云同步稍后到手机")
    return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gen_only = "--gen-only" in sys.argv
    dry_run = "--dry-run" in sys.argv
    date = args[0] if args else None
    if date is None:
        newest = _newest("legacy_stocklist_*__*.csv", STOCK_LIST_DIR)
        if newest is None:
            raise SystemExit("STOCK LIST 目录无任何 legacy 清单")
        date = re.search(r"legacy_stocklist_(\d{8})__", os.path.basename(newest)).group(1)

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
