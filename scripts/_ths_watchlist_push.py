"""同花顺自选股推送: 当日 TOP10 → 同花顺可导入 txt → UI 自动化导入 PC 客户端.

TOP10 口径 (2026-09-05 用户拍板 "MAKE SEPARATE LIST NOT COMBINED ONE"): 双源
各自独立成单各推各的, 不再并集 — parallel 短名单 rank 前 10 一单, legacy 清单
序前 10 一单 (先 parallel 后 legacy 导入, 自选股内成两块); 两单各自单内逐码
去重, **跨源重复股保留在两单** (09-05 用户: "如果有重复, 不要去掉, 保留他们");
撞指数码的 000xxx **保留在清单**, 推送端只入股票不入指数 (09-05 用户澄清
"不是删除股票号"); 单侧缺失只推另一侧.
数据源 (STOCK LIST 目录, 当日):
  parallel_shortlist_{date}__{module}.csv    并行短名单 → rank 列升序前 10
  legacy_stocklist_{date}__{module}.csv      legacy 清单 → 清单序前 10
生成: ths_watchlist_{date}__{HH}__parallel__{模型批次串}.txt /
      ths_watchlist_{date}__{HH}__legacy__{模型批次串}.txt
每行一个 6 位代码 (同花顺自动识别市场); {HH} = 推送小时 (09-05 用户: 日期后加
小时 — 同日重推各留各的 txt 不覆盖, _ths_flush_guard 正则要求日期后紧跟 __ 故
小时放其后); 标签 = 源名__模型批次串 (09-05 用户: LEGACY 文件就叫 LEGACY +
TXT 需要模型名和日期).

导入: uiautomation 驱动 hexin.exe (同花顺 PC 客户端). 客户端需已登录同账号 —
导入进自选股后云同步到手机 App. --gen-only 只生成文件不动客户端.

用法: python scripts/_ths_watchlist_push.py [YYYYMMDD] [--gen-only] [--dry-run]
"""

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import glob as _glob

from config.settings import STOCK_LIST_DIR
from scripts import _deadzone_guard
from scripts._ths_ui import THS_HEXIN_PATH  # 单一来源; 本模块再导出兼容旧引用

CODE_RE = re.compile(r"^\d{6}$")
# 深A 000xxx 股票与上证/中证指数撞码 (000985 大庆华科=中证全指, 09-03 实证):
# 复制识别必出股票+指数两行, 前后缀消歧无效。09-05 用户澄清: 代码不剔 (留在
# 清单里), 推送端把撞码股单独成批垫第二位 → 指数行落 4 行组第 3 位死位不自动
# 勾, 点加入只入股票不入指数; 核验不过 fail-closed 整批跳过 (见 _build_chunks)。
INDEX_COLLIDE_RE = re.compile(r"^000\d{3}$")
# 缺码补推循环上限 (2026-09-05 用户: "LOOP PUSHING TILL COMPLETE" + "最多10轮"):
# 终态仍有缺码 → 只贴缺的整单重扫, 粘贴自动勾选是抽签重贴重抽; 每轮开头重查
# 空闲闸, 用户回座立即中止 (键盘路线绝不与用户抢机器)
PUSH_SWEEPS = 10
# 推送中间产物 (ths_watchlist txt / ths_push_result csv) 链末归档子目录名
# (2026-09-08 用户: STOCK LIST 只留清单/终表, 推送结果单不堆主目录);
# 移动非删除 — read_push_results 兼读此目录, 看板推送状态卡不断粮
THS_PUSH_ARCHIVE = "ths_push_archive"
# 补推轮间静默 (2026-09-08 自毒化修复): 上一轮自己的合成点击/剪贴板让空闲闸
# (IDLE_MIN_S=90) 把第 2 轮误判成"用户在场" → 补推循环从未真正跑过第 2 轮
# (09-08 实弹)。轮头先静默 95s 让合成输入衰减过阈值再查闸 — 用户真回座照常拦。
SWEEP_REST_S = 95
# 一进程双单 (main) 同理: 单1 (parallel) 的合成输入毒化单2 (legacy) 的入口空闲闸
# → legacy 恒 blocked (09-08 实弹)。单间休 150s + 有界真空闲守候 (用户恰在间隙
# 回座 → 最多等 10min, 到点放弃走 blocked, 不死等)。
INTER_ORDER_REST_S = 150
INTER_ORDER_IDLE_WAIT_S = 10 * 60


def _newest(pattern: str, list_dir) -> str | None:
    hits = _glob.glob(str(list_dir / pattern))
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def _module_of(path: str) -> str:
    return os.path.basename(path).split("__", 1)[1][: -len(".csv")]


def _valid_codes(symbols) -> list[str]:
    """过滤非法代码 + 单内逐码去重, 保序 (跨源重复由 collect_lists 保留)."""
    out: list[str] = []
    for sym in symbols.dropna().astype(str):
        sym = sym.strip()
        if CODE_RE.match(sym) and sym not in out:
            out.append(sym)
    return out


def collect_lists(
    date: str, list_dir=STOCK_LIST_DIR, top_n: int = 10
) -> list[tuple[str, list[str]]]:
    """当日双源各自 TOP10 → [("parallel__{模型名}", 代码), ...], 并行在前.

    标签 = 源名__模型批次串 (取自源 CSV 文件名). 每单内部过滤非法代码 + 逐码
    去重; 跨源重复股保留在两单 (2026-09-05 用户: "LEGACY AND PARALLEL 如果有
    重复, 不要去掉, 保留他们"); 撞指数码 000xxx 不剔 (推送端隔离, 见
    _build_chunks). 单侧缺失只返回另一侧.
    """
    import pandas as pd

    lists: list[tuple[str, list[str]]] = []

    parallel = _newest(f"parallel_shortlist_{date}__*.csv", list_dir)
    if parallel is not None:
        df = pd.read_csv(parallel, dtype={"symbol": str})
        if "rank" in df.columns:
            df = df.dropna(subset=["rank"]).sort_values("rank")
        lists.append(
            (f"parallel__{_module_of(parallel)}", _valid_codes(df["symbol"])[:top_n])
        )

    legacy = _newest(f"legacy_stocklist_{date}__*.csv", list_dir)
    if legacy is not None:
        df = pd.read_csv(legacy, dtype={"symbol": str})
        lists.append(
            (f"legacy__{_module_of(legacy)}", _valid_codes(df["symbol"])[:top_n])
        )

    if not lists:
        raise SystemExit(f"无清单: parallel/legacy_stocklist_{date}__*.csv")
    return lists


def ths_txt_path(date: str, module: str, list_dir=STOCK_LIST_DIR):
    return list_dir / f"ths_watchlist_{date}__{module}.txt"


def write_ths_txt(codes: list[str], path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(codes) + "\n")


# 冲刷串: 无字母无数字 — 识别器无法解析成任何代码行 (旧 "ths-push" 被识别成
# THS/PUSH 两只美股混进列表, 09-03 截图实证)
FLUSH_STR = "-----"
# 每批 2 只: 粘贴自动勾选是位置伪影 — 每 4 行组第 3 位 (3/7/11…) 不自动勾
# (09-03 v8 两轮截图复现, 19/19 全过即用 CHUNK=2; 09-05 午前 5 行大批实测
# 第 3/4 位仍不勾且 Space 点不动的假勾判读) → 每批只占组内安全位 1/2.
# 2 ≤ 列表可视 ~16 行, 滚动截断风险自然消除
CHUNK_SIZE = 2
# 每批粘贴核验重试次数 (09-05: 粘贴自动勾选是抽签, 关窗重贴=重抽; 合成点击
# 勾选框在同花顺自绘控件上实测不生效 → 键盘路线补勾, 部分勾选先落袋)
RETRY_PASTE = 3
# 市场列 tag ("深A/沪A/中证") x 带 (09-05 失败截图标定: 灰字 292..318; 上涨股
# 红价从 x≈379 起 — 红字判据若看整行会把上涨股票行误判成指数行, 09-05 实弹
# 002098 +2.44% 红价触发 fail-closed 整批拒加)
INDEX_TAG_X0, INDEX_TAG_X1 = 285, 331
# 核验失败现场存档目录 (测试须重定向到 tmp_path — 09-05 单测曾把 10x10 零图
# dump 进真实 tmp_t, 覆盖丢失 probe5 实弹 forensic 存档 idxchk1.png)
FAIL_DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp_t")


def _row_is_index(band: np.ndarray) -> bool:
    """[诊断信号, 非判据] 行带市场列 x∈[INDEX_TAG_X0, INDEX_TAG_X1) 含红字.

    09-05 实弹证伪: "中证" 标签实渲染为深灰 (25,25,25), 红字判据从未在实弹
    截图命中过指数行 (历史 "指数行已勾" 全是上涨股红价在旧整行判据下的误报).
    现仅作污染保险: 红 tag 行落在股票槽位且已勾 → fail-closed (见核验函数);
    指数行的主识别 = 垫批 None 槽位 (位序, 09-03 v8 + 09-05 retry9 两次复现).
    """
    band = band[:, INDEX_TAG_X0:INDEX_TAG_X1]
    r = band[..., 0].astype(int)
    g = band[..., 1].astype(int)
    return int(((r > 150) & (g < 100)).sum()) >= 2


def _dlg_rows_from_img(img: np.ndarray) -> list[tuple[int, bool, bool]]:
    """对话框区域截图 (x 从对话框左缘起, y 从表头下缘起) → [(行中心y, 是否已勾, 是否指数行)].

    行 = 文字区深色带 (高≥10px, x≥33 避开勾选框列); 勾选框内部取 x 17..28
    (校准: 框体 rel 15..29, 边框 725/739), 内部 min 灰度 <80 = 有勾墨迹,
    ≥80 = 空框 (09-03 实测勾 53/54 vs 空 98); 指数行 = 行带红字 (_row_is_index).
    纯数组函数, 可单测.
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
        rows.append((yc, bool(cell.min() < 80), _row_is_index(img[ya:yb])))
    return rows


def _ensure_dialog_rows_checked(
    dlg, row_codes: list, log=print, max_rounds: int = 4
) -> dict | None:
    """核验行数+位序+勾选 → {code: 是否已勾}; 不可点加入 → None.

    row_codes: 期望行序对应的代码; None = 指数行槽位 (垫批 [X,000,Y] →
    [X, 000, None, Y], 行序假设=指数行紧随股票行 — 09-03 v8 + 09-05 retry9
    两次截图复现; "中证" tag 实为深灰非红字, 位序即识别). 行数不符 (识别未齐/
    杂行混入) 只重查不强点; None 槽位行或红 tag 诊断行已勾 → None (无法取消,
    点加入会连指数入自选). 股票行未勾走键盘路线: 点行文字选中 (合成点击勾选
    框在自绘控件上不生效, 09-05 实证 16 连击无效), Space 切勾; 重查耗尽仍缺 →
    返回现状映射, 调用方点加入只入勾选行先落袋, 缺的重贴重试 (09-05: 粘贴
    自动勾选是抽签, 全勾等不到时部分勾也是净进展).
    """
    import time

    import uiautomation as auto
    from PIL import ImageGrab

    from scripts import _ths_ui as ui

    r = dlg.BoundingRectangle
    y0 = r.top + 62  # 表头以下
    y1 = r.bottom - 58  # 按钮行以上

    def _dump_fail(img, tag: str) -> None:
        # 核验失败现场落盘 (09-05 首晚实弹: 普通股行被红字判据误判指数行,
        # 无截图无法定位 — 存档供事后判读, 不影响流程)
        try:
            from PIL import Image

            fp = os.path.join(FAIL_DUMP_DIR, f"ths_dlg_fail_{tag}.png")
            Image.fromarray(img).save(fp)
            log(f"[ths] 失败现场已存 {os.path.abspath(fp)}")
        except Exception as exc:
            log(f"[ths] 失败现场存档失败: {exc}")

    def _grab():
        img = np.asarray(ImageGrab.grab(bbox=(r.left, y0, r.right - 10, y1)))
        return img, _dlg_rows_from_img(img)

    rows = []
    for rnd in range(max_rounds):
        if rnd:
            time.sleep(1.0)
        img, rows = _grab()
        log(
            f"[ths] round {rnd + 1}: "
            + " ".join(
                f"y{yc}/{'勾' if ck else '空'}/{'指' if ix else '股'}"
                for yc, ck, ix in rows
            )
            + f" (期望 {len(row_codes)} 行)"
        )
        if len(rows) != len(row_codes):
            log(
                f"[ths] 对话框行数 {len(rows)} != 期望 {len(row_codes)} "
                f"(round {rnd + 1})"
            )
            _dump_fail(img, f"rowcount_{len(row_codes)}_got{len(rows)}")
            continue
        for (yc, _ck, ix), rc in zip(rows, row_codes):
            if ix and rc is not None:
                log(f"[ths] y{yc} 红 tag 落在股票槽位 ({rc}) — 诊断")
        # 指数行按位序识别 (None 槽位): 09-05 实弹 "中证" 标签深灰非红字,
        # 红字判据从未实检命中过指数行; 垫批死位 09-03 v8 + 09-05 retry9 两次
        # 复现. None 槽位行/红 tag 行已勾 → 无法取消, 点加入连指数入自选 → 拒
        index_checked = [
            y0 + yc
            for (yc, ck, ix), rc in zip(rows, row_codes)
            if (rc is None or ix) and ck
        ]
        if index_checked:
            log(f"[ths] 指数行已勾 {len(index_checked)} 行 (无法取消), 不点加入")
            _dump_fail(img, f"idxchk{len(index_checked)}")
            return None
        unchecked = [
            y0 + yc
            for (yc, ck, ix), rc in zip(rows, row_codes)
            if rc is not None and not ix and not ck
        ]
        if not unchecked:
            return {rc: True for rc in row_codes if rc is not None}
        for ay in unchecked:
            ui.assert_foreground_hexin(f"勾选对话框行 y={ay}")
            auto.Click(r.left + 150, ay)  # 点行文字选中 (非勾选框)
            time.sleep(0.4)
            img2, rows2 = _grab()
            still = [
                y for y, ck, ix in rows2 if not ix and abs(y0 + y - ay) <= 3 and not ck
            ]
            if not still:
                log(f"[ths] 点击行文字 ({r.left + 150},{ay}) → 已勾")
                continue
            auto.SendKeys("{Space}")  # 空格切换选中行勾选
            time.sleep(0.5)
            img3, rows3 = _grab()
            still = [
                y for y, ck, ix in rows3 if not ix and abs(y0 + y - ay) <= 3 and not ck
            ]
            log(f"[ths] 点击行文字+Space y={ay} → " + ("仍未勾" if still else "已勾"))
    # 重查耗尽: 终检一次, 行数/位序可映射就交现状 (部分勾选也是净进展)
    img, rows = _grab()
    if len(rows) != len(row_codes):
        log(f"[ths] 终检行数 {len(rows)} != 期望 {len(row_codes)}")
        _dump_fail(img, f"final_rowcount_{len(row_codes)}_got{len(rows)}")
        return None
    if any((rc is None or ix) and ck for (yc, ck, ix), rc in zip(rows, row_codes)):
        log("[ths] 终检指数行已勾, 不点加入")
        _dump_fail(img, "final_idxchk")
        return None
    out = {rc: bool(ck) for (yc, ck, ix), rc in zip(rows, row_codes) if rc is not None}
    if not all(out.values()):
        log(
            f"[ths] 重查耗尽仍未勾 {sum(1 for g in out.values() if not g)} 只 "
            "(部分勾选先落袋, 缺的重贴重试)"
        )
    return out


def _build_chunks(codes: list[str]) -> list[list[str]]:
    """分批: 普通码 CHUNK_SIZE/批; 撞指数码 000xxx 单独垫批 [X, 000, Y].

    撞码股复制识别必出股票+指数两行; 每 4 行组第 3 位不自动勾 (09-03 v8 两轮
    截图位置复现). 撞码股垫批内第 2 位 → 行序 X(1)/股票(2)/指数(3)/Y(4),
    指数行落死位不勾, 点加入只入股票. 垫码 X/Y 复用普通码 (加自选幂等无害);
    普通码不足 2 只退化单码批 (指数行会勾上 → 核验 fail-closed 整批跳过,
    宁可这只不入也不入指数). 行序假设=指数行紧随股票行, 核验兜底.
    """
    normal = [c for c in codes if not INDEX_COLLIDE_RE.match(c)]
    collide = [c for c in codes if INDEX_COLLIDE_RE.match(c)]
    chunks = [normal[i : i + CHUNK_SIZE] for i in range(0, len(normal), CHUNK_SIZE)]
    if not collide:
        return chunks
    if len(normal) >= 2:
        chunks.extend([normal[0], c, normal[1]] for c in collide)
    else:
        chunks.extend([c] for c in collide)
    return chunks


def _row_codes(chunk: list[str]) -> list[str | None]:
    """批内代码 → 期望行序 (核验映射用): 撞码股其后紧跟指数行槽位 None.

    纯股票批 → 代码本身 (行 i 即代码 i); 垫批 [X, 000, Y] →
    [X, 000, None, Y]. 位序假设 = 指数行识别 (红 tag 仅诊断保险).
    """
    out: list[str | None] = []
    for c in chunk:
        out.append(c)
        if INDEX_COLLIDE_RE.match(c):
            out.append(None)
    return out


def _result_path(txt_path) -> Path:
    """推送 txt → 结果单路径 (前缀互换, 一一对应):
    ths_watchlist_20260903__09__parallel__M1.txt →
    ths_push_result_20260903__09__parallel__M1.csv"""
    p = Path(txt_path)
    return p.with_name(
        p.name.replace("ths_watchlist_", "ths_push_result_", 1).replace(".txt", ".csv")
    )


def write_push_result(
    txt_path,
    codes: list[str],
    landed: list[str],
    blocked: bool = False,
    note: str | None = None,
) -> Path:
    """推送结果单 (2026-09-05 用户: 缺码要能一眼看到): 每码一行 symbol,status.

    status: landed=核验勾选+点加入落袋 / manual=试过但终态缺 (需手动加) /
    blocked=推送根本没开跑 (空闲闸/窗口/对话框失败) / note 覆写全部行
    (死区停推夜用 "deadzone", 与故障 blocked 区分 — 2026-09-05 用户).
    命名与 txt 一一对应, WORM 口径同 txt (同日重推按小时戳各留各的).
    """
    import csv

    fp = _result_path(txt_path)
    with open(fp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "status"])
        for c in codes:
            if note is not None:
                st = note
            elif blocked:
                st = "blocked"
            elif c in landed:
                st = "landed"
            else:
                st = "manual"
            w.writerow([c, st])
    return fp


def read_push_results(date: str | None = None, list_dir=STOCK_LIST_DIR):
    """读推送结果单 → 每源最新一份的逐码状态 (看板"同花顺推送状态"卡数据源).

    返回 DataFrame[source, symbol, status]; date 缺省取有结果单的最新日期;
    同源同日多份 (小时戳 WORM) 只取 mtime 最新一份.
    """
    import pandas as pd

    cols = ["source", "symbol", "status"]
    # 归档子目录兼读: 推送中间产物 (txt/结果单) 链末移入 ths_push_archive (09-08
    # 用户: STOCK LIST 只留清单/终表), 看板卡/终表重建仍要能读到
    dirs = [Path(list_dir), Path(list_dir) / THS_PUSH_ARCHIVE]
    if date is None:
        hits = [
            h for d in dirs for h in _glob.glob(str(d / "ths_push_result_*__*.csv"))
        ]
        if not hits:
            return pd.DataFrame(columns=cols)
        newest = max(os.path.basename(h) for h in hits)
        date = newest[len("ths_push_result_") :].split("__", 1)[0]
    pats = sorted(
        [
            h
            for d in dirs
            for h in _glob.glob(str(d / f"ths_push_result_{date}__*.csv"))
        ],
        key=os.path.getmtime,
    )
    # 结果单名 = ths_push_result_{date}__{tag}; tag 两代:
    # {HH|HHMM}__{源}__{批次串} (09-05 起带小时; 09-08 起残单/合并单用 HHMM 级戳
    # 防同小时覆盖) / {源}__{批次串}; prob10dens 无批次串 → 单段
    newest_by_src: dict[str, str] = {}
    for fp in pats:  # mtime 升序 → 后写覆盖, 每源留最新
        tag = os.path.basename(fp)[len("ths_push_result_") : -len(".csv")]
        rest = tag.split("__")[1:]  # 首段=日期
        src = (
            rest[1] if len(rest) > 1 and re.fullmatch(r"\d{2,4}", rest[0]) else rest[0]
        )
        newest_by_src[src] = fp
    frames = []
    for src, fp in newest_by_src.items():
        df = pd.read_csv(fp, dtype={"symbol": str})
        df["source"] = src
        frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


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
         (旧行为"不看勾选状态"已失效) — 直接点击=静默漏加. 部分勾选也点加入
         (只入勾选行先落袋, 09-05: 全勾等不到时部分勾也是净进展), 缺码关窗
         重贴重试. 分批 ≤12 只/批, 防对话框列表区滚动截断行数核验
      7. **缺码补推循环 (2026-09-05 用户 "LOOP PUSHING TILL COMPLETE 最多10轮")**:
         终态仍有缺码 → 只贴缺的整单重扫 (粘贴自动勾选是抽签, 重贴重抽),
         最多 PUSH_SWEEPS=10 轮; 每轮开头先静默 SWEEP_REST_S 让合成输入衰减
         (2026-09-08 自毒化修复), 再重查空闲闸, 用户回座立即中止
      8. 关闭对话框. 无成功/失败回执 (成功 toast 对部分加也报成功), 对话框
         像素判词不可信 (09-05 实证) → **终态判词 = read_all_codes 读 PC
         网格真值** (candidates=今晚码单, 约束匹配 — 09-05 接产线, 自由 OCR
         8/0 形歧义曾把真落袋冤判成需手动): 网格在位=落袋, 未见=需手动
         (对话框曾核验但网格未见=疑似掉登录提示). 终态写结果单
         (write_push_result, 看板卡数据源), 推送没开跑也写 (全 blocked)
      9. **掉登录自愈 (2026-09-08 "以后推送自动化")**: 疑似掉登录签名
         (对话框核验过但网格全空) → 杀 hexin 整进程集 + 重启拉起 + settle,
         单次重推; 重启后仍撞登录墙 (晚间登录墙 09-07/09-08 实弹) → 结果单
         blocked + note=login_wall 明示需人工登录, 不静默 no-op.

    安全闸 (2026-09-03 误删事故后加, 见 scripts/_ths_ui.py 模块头):
      入口空闲闸 (用户在场直接 return False) + 每次点击前前台 hexin 断言.
    """
    if dry_run:
        print(f"[dry] 将导入同花顺: {txt_path}")
        return True

    codes = [c for c in Path(txt_path).read_text(encoding="utf-8").split() if c]
    if not codes:
        print("[ths] txt 为空, 跳过")
        return True

    ok, verdict = _push_once(txt_path, codes)
    if verdict != "dead_session":
        return ok

    import subprocess
    import time

    from scripts import _ths_ui as ui

    print("[ths] 疑似掉登录 (加入无效果): 重启客户端单次重推")
    killed = 0
    for pid in ui.hexin_pids():
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=30
            )
            killed += 1
        except Exception:  # noqa: BLE001 — 杀不掉只能放弃重启尝试
            pass
    if not killed:
        print("[ths] 无 hexin 进程可杀, 直接冷启动")
    time.sleep(5)
    try:
        ui.ensure_watchlist_window()  # 冷启动拉起 (记住密码自动登录)
    except Exception as exc:
        write_push_result(txt_path, codes, [], blocked=True, note="login_wall")
        print(f"[ths] 重启后撞登录墙 ({exc}) — 需人工登录, 结果单已标 login_wall")
        return False
    time.sleep(40 + SWEEP_REST_S)  # settle + 重启点击的合成输入同样要衰减过闸
    ok2, verdict2 = _push_once(txt_path, codes)
    if verdict2 == "dead_session":
        print("[ths] 重启后仍加入无效果 — 留待人工, 不再重试")
    return ok2


def _push_once(txt_path, codes: list[str]) -> tuple[bool, str]:
    """单会话推送主体 (push_via_ths 拆出): 返回 (ok, 判词), 判词 dead_session =
    掉登录签名 — 由调用方决定是否重启重推."""
    import ctypes
    import time

    import uiautomation as auto

    from scripts import _ths_ui as ui

    if not ui.ensure_idle(what="自选股推送"):
        write_push_result(txt_path, codes, [], blocked=True)
        return False, ""

    try:
        # 共享窗口原语: 现存自选股窗 → 主窗 F6 呼出 (客户端在跑但停在别的页,
        # 如上次对话框流程留下的分时图页, 09-03 实证) → 冷启动兜底
        win = ui.ensure_watchlist_window()
    except ui.ForegroundLostError as exc:
        print(f"[ths] {exc}")
        write_push_result(txt_path, codes, [], blocked=True)
        return False, ""
    except Exception as exc:
        print(f"[ths] 自选股窗口不可用: {exc}")
        write_push_result(txt_path, codes, [], blocked=True)
        return False, ""

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
        write_push_result(txt_path, codes, [], blocked=True)
        return False, ""

    landed: list[str] = []
    for sweep in range(PUSH_SWEEPS):
        pending = [c for c in codes if c not in landed]
        if not pending:
            break
        if sweep:
            # 轮间静默 (2026-09-08 自毒化修复): 上一轮自己的合成点击/剪贴板会让
            # 空闲闸把本轮误判成用户在场 → 旧码轮询一次即自灭. 先静默让合成输入
            # 衰减过 IDLE_MIN_S 再查闸 — 用户真回座照常立即中止
            time.sleep(SWEEP_REST_S)
            if not ui.ensure_idle(what="缺码补推循环"):
                print(
                    f"[ths] 用户回座, 补推中止 (已落袋 {len(set(landed))}/{len(codes)})"
                )
                break
            print(
                f"[ths] 补推第 {sweep + 1}/{PUSH_SWEEPS} 轮, 缺 {len(pending)}: "
                + " ".join(pending)
            )
            auto.SetClipboardText(FLUSH_STR)
            time.sleep(0.8)
            dlg = fresh_dialog()
            if dlg is None:
                print("[ths] 补推对话框未出现, 中止循环")
                break
        for ci, chunk in enumerate(_build_chunks(pending)):
            n_collide = sum(1 for c in chunk if INDEX_COLLIDE_RE.match(c))
            if ci:
                auto.SetClipboardText(FLUSH_STR)
                time.sleep(0.8)
                dlg = fresh_dialog()
                if dlg is None:
                    print("[ths] 后续批对话框未出现")
                    break
            joined_all = False
            row_codes = _row_codes(chunk)
            for attempt in range(RETRY_PASTE):
                if attempt:
                    # 核验缺码 → 关窗重贴重抽签: 粘贴自动勾选是抽签 (09-05 实测同
                    # 一批有时全勾有时缺 2, 合成点击勾选框在自绘控件上不生效),
                    # 重开对话框重贴 = 重新抽签. 重开前冲刷剪贴板 — 对话框打开即
                    # 吸当前剪贴板, 残留上一批码会污染行数核验.
                    auto.SetClipboardText(FLUSH_STR)
                    time.sleep(0.8)
                    dlg = fresh_dialog()
                    if dlg is None:
                        print("[ths] 重贴对话框未出现")
                        break
                auto.SetClipboardText("\n".join(c for c in row_codes if c))
                time.sleep(2.5 + 0.5 * len(row_codes))
                res = _ensure_dialog_rows_checked(dlg, row_codes)
                if res is None:
                    print(
                        f"[ths] 批次 {ci + 1} 第 {attempt + 1}/{RETRY_PASTE} 次"
                        "核验不可加, 不点加入"
                    )
                    continue
                good = [c for c, g in res.items() if g]
                bad = [c for c, g in res.items() if not g]
                if good:
                    # 加入只入勾选行 (09-03 实证) → 部分勾选先落袋, 缺的重贴重试
                    ui.assert_foreground_hexin("点击加入自选股")
                    dr = dlg.BoundingRectangle
                    auto.Click(dr.right - 205, dr.bottom - 33)  # 加入自选股
                    landed.extend(good)
                    time.sleep(1.5)
                if not bad:
                    joined_all = True
                    break
                print(
                    f"[ths] 批次 {ci + 1} 第 {attempt + 1}/{RETRY_PASTE} 次 "
                    f"入 {len(good)} 缺 {len(bad)}: {' '.join(bad)}"
                )
                if not n_collide:
                    row_codes = bad  # 纯批: 下轮只贴缺的
            if not joined_all:
                missing = [c for c in chunk if c not in landed]
                if missing:
                    print(
                        f"[ths] 批次 {ci + 1} 未入自选 {len(missing)} 只 "
                        f"(需手动加): {' '.join(missing)}"
                    )
                # 撞码批失败只弃该批 (垫码在普通批已入); 纯批缺码留给补推轮重扫
                continue

    ui.assert_foreground_hexin("关闭复制识别对话框")
    dlg = ui.find_window("复制识别")
    if dlg is not None:
        ui.close_x(dlg)
    time.sleep(0.5)
    # 判词 = PC 网格真值 (09-05 午前二次破案: 对话框像素+槽位映射的 "已核验
    # 9/6" 被用户云端实查推翻; 活会话=加入即网格可见, 死会话=加入 no-op 网格
    # 不变 → 网格读把假成功根治成真落袋或当场报警; 云端真值仍只有网页).
    # 判词只需在今晚推的码里挑 → candidates 约束匹配 (09-05 接产线: 自由 OCR
    # 8/0 形歧义冤枉过 3 只真落袋股, 688433 读成 600433 等)
    landed_grid: list[str] = []
    try:
        grid_codes = ui.read_all_codes(win, candidates=codes)
        landed_grid = [c for c in codes if c in set(grid_codes)]
    except Exception as exc:
        print(f"[ths] 网格真值核验未完成: {exc}")
    dlg_landed = [c for c in dict.fromkeys(landed) if c not in landed_grid]
    missing = [c for c in codes if c not in landed_grid]
    ok = not missing
    if ok:
        print(
            f"[ths] 已核验加入自选股 ({len(landed_grid)}/{len(codes)} 只), "
            "云同步稍后到手机"
        )
    else:
        print(
            f"[ths] 网格核验在位 {len(landed_grid)}/{len(codes)}"
            + (f": {' '.join(landed_grid)}" if landed_grid else "")
        )
        hint = (
            "; 对话框曾核验但网格未见 → 疑似掉登录 (加入无效果), 登录后重推"
            if dlg_landed and not landed_grid
            else "; 同轮已有他码落袋, 会话在活 — 属勾选判读假阳性, 已多轮重贴"
            if dlg_landed
            else ""
        )
        print(
            f"[ths] 未入自选 {len(missing)} 只 (需手动加): " + " ".join(missing) + hint
        )
    fp = write_push_result(txt_path, codes, landed_grid)
    print(
        f"[ths] 结果单 {os.path.basename(fp)} "
        f"(落袋 {len(set(landed_grid))} / 缺 {len(missing)})"
    )
    return ok, ("dead_session" if dlg_landed and not landed_grid else "")


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

    lists = collect_lists(date)
    if all(not codes for _, codes in lists):
        print(f"[warn] {date} 双源清单均为空, 不推送")
        return 0
    # 死区停推闸 (2026-09-05 拍板 "那就一起停吧"; 晚间细化: legacy/parallel 各用
    # 各的纯样本赢率互不混合): legacy 单看 top10 线, parallel 单看 parallel 线,
    # 只停报警的单另一单照推; 清单 CSV 照出照存; gen-only/dry-run 人工演练不拦
    dz_line = {"legacy": "top10", "parallel": "parallel"}
    alarms = {
        module: _deadzone_guard.is_alarm(dz_line[module.split("__", 1)[0]], date)
        for module, codes in lists
        if codes
    }
    stopped: set[str] = set()
    for module, codes in lists:
        if not codes:
            continue
        alarm, why = alarms[module]
        if alarm:
            print(f"[deadzone] 死区报警 ({module}): {why}")
            if not gen_only and not dry_run:
                _deadzone_guard.annotate_stop(
                    dz_line[module.split("__", 1)[0]], date, why
                )
                stopped.add(module)
                print(
                    f"[deadzone] {module} 今晚停推: 不写 txt 不加自选"
                    " (清单照出, 只加不删不受影响)"
                )
    if stopped:
        all_codes = sorted({c for m, cs in lists if m in stopped for c in cs})
        write_push_result(
            ths_txt_path(date, "deadzone"), all_codes, [], note="deadzone"
        )
        print(
            "[deadzone] 已标注: STOPPED_DEADZONE 标记 + 清单 md 横幅(top10线)"
            " + 推送结果单 status=deadzone (与没推成功区分)"
        )
    if not gen_only and alarms and all(m in stopped for m in alarms):
        return 0  # 双单全停, 无可推
    if not gen_only and not THS_HEXIN_PATH.exists():
        print(f"[warn] 同花顺客户端不存在: {THS_HEXIN_PATH}")
        return 0

    rc = 0
    hh = datetime.now().strftime("%H")  # 小时戳: 同日重推各留各的 txt (WORM)
    attempted = False  # 已实推过至少一单 (dry-run 不算 — 无合成输入)
    for module, codes in lists:
        if not codes:
            print(f"[warn] {date} {module} 清单为空, 跳过")
            continue
        if module in stopped:
            continue
        out = ths_txt_path(date, f"{hh}__{module}")
        write_ths_txt(codes, out)
        print(f"[ths] {out} ({len(codes)} 只, module {module})")
        if not gen_only:
            if attempted:
                _rest_between_orders(module)
            if not push_via_ths(out, dry_run):
                rc = 1
            attempted = True
    return rc


def _rest_between_orders(next_module: str) -> None:
    """双单连推的合成输入衰减 (2026-09-08): 单1 的剪贴板/点击刚发生, 入口空闲闸
    (IDLE_MIN_S=90) 会把单2 误判成用户在场 → legacy 恒 blocked (09-08 实弹).
    先静默 INTER_ORDER_REST_S 让合成输入衰减, 再有界守候真空闲 — 用户恰在此
    间隙回座 → 到点放弃, 下一单交 push_via_ths 入口闸自然写 blocked, 不死等."""
    import time

    from scripts import _ths_ui as ui

    print(
        f"[ths] 单间休 {INTER_ORDER_REST_S}s (上一单合成输入衰减, 下一单 {next_module})"
    )
    time.sleep(INTER_ORDER_REST_S)
    deadline = time.monotonic() + INTER_ORDER_IDLE_WAIT_S
    while time.monotonic() < deadline:
        if ui.user_idle_seconds() >= ui.IDLE_MIN_S:
            return
        time.sleep(5)
    print("[ths] 单间等不到真空闲 (用户在场), 下一单交入口闸判")


if __name__ == "__main__":
    sys.exit(main())
