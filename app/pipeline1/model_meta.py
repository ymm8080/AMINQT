"""模型包版本元数据 — 预测/清单文件打模块版本戳, 供回归测试评估各模块表现.

current_meta.json 结构 (models/pipeline1/):
  {
    "main": {"tag": "20260805_q234", "file": "main_20260805_q234.pkl",
             "updated": "2026-08-05 12:00"},
    "dual": {"tag": "20260805_q234", "file": "dual_20260805_q234.pkl",
             "updated": "2026-08-05 12:00"}
  }

写入时机: 每次 bundle 指针 (xxx_current.pkl) 更新后 (retrain / extras splice).
读取时机: 预测/清单交付时, 把 module 打进文件名 + 每行 model_version 列.
护栏时机: 历史回放/AB 的评估窗切分时, 用 assert_clean_window 拒绝污染窗 (见文末).
"""

import datetime as dt
import json

META_PATH = "models/pipeline1/current_meta.json"


def load_modules(meta_path: str = META_PATH) -> dict:
    """{board: {tag, file, updated}}; 缺失/损坏返回 {} (调用方回退 'na')."""
    try:
        # utf-8-sig: 容忍 Windows 工具 (PowerShell 等) 写入时带的 BOM
        with open(meta_path, encoding="utf-8-sig") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_modules(modules: dict, meta_path: str = META_PATH) -> None:
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(modules, fh, ensure_ascii=False, indent=2)


BOARD_TO_TRACK = {"main": "main", "GEM": "dual", "STAR": "dual"}


def board_tag(modules: dict, board) -> str:
    """清单 board 值 (main/GEM/STAR) → 训练轨道 (main/dual) → 模块 tag."""
    track = BOARD_TO_TRACK.get(str(board), "na")
    return (modules.get(track) or {}).get("tag", "na")


def module_id(modules: dict) -> str:
    """文件名用模块标识: 双板同 tag → 单一 tag; 否则 M{main}__D{dual}.

    作为回归分组键: 同一 module_id 的预测在评估时归并, 比较各模块 OOS 表现.
    """
    main = (modules.get("main") or {}).get("tag", "na")
    dual = (modules.get("dual") or {}).get("tag", "na")
    if main == dual:
        return main or "na"
    return f"M{main}__D{dual}"


# ── 污染窗口护栏 ──────────────────────────────────────────────────────────
# bundle 在面板 date.max() 上训练, tag 默认训练当天 (见 scripts/_retrain_legacy_full.py),
# 故评估窗内任一日期 <= tag 即含未来信息 —— 模型训练时见过该日, 回放数字是样本内。
# 这条此前只写在文档与各脚本的硬编码常量里, 没有任何代码强制; 下面把它变成可调用的闸。


class ContaminatedWindowError(ValueError):
    """评估窗与 bundle 训练窗重叠 → 数字含未来信息, 结论无效."""


def parse_tag_date(tag) -> dt.date | None:
    """tag ('20260903' / '20260805_q234') → date(2026, 9, 3); 无法解析返回 None."""
    return as_date(tag)


def board_cutoff(
    board, modules: dict | None = None, meta_path: str = META_PATH
) -> dt.date | None:
    """清单 board (main/GEM/STAR) → 该轨 bundle 训练截止日; 无 tag/不可解析返回 None."""
    if modules is None:
        modules = load_modules(meta_path)
    return parse_tag_date(board_tag(modules, board))


def as_date(value) -> dt.date | None:
    """日期归一: str 'YYYYMMDD'/'YYYY-MM-DD' / date / datetime / Timestamp / datetime64."""
    if isinstance(value, dt.datetime):  # datetime 是 date 子类, 必须先判
        return value.date()
    if isinstance(value, dt.date):
        return value
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if len(digits) < 8:
        return None
    try:
        return dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8]))
    except ValueError:
        return None


def assert_clean_window(dates, cutoff, label: str = "bundle") -> None:
    """评估窗内任一日期 <= 训练截止 → ContaminatedWindowError (大声失败, 不静默放行).

    dates: 可迭代日期/日期串, 如 ('20260101', '20260921') 或日期数组.
    cutoff: 训练截止日 (date / 'YYYYMMDD'); None = 无法证明洁净 → 同样拒绝出数.
    """
    cut = as_date(cutoff)
    if cut is None:
        raise ContaminatedWindowError(
            f"{label} 无可用训练截止 ({cutoff!r}) → 无法证明评估窗洁净, 拒绝出数"
        )
    bad = []
    for d in dates:
        dd = as_date(d)
        if dd is None:
            raise ContaminatedWindowError(f"{label} 评估窗含不可解析日期 {d!r}")
        if dd <= cut:
            bad.append(dd)
    if bad:
        raise ContaminatedWindowError(
            f"污染窗口: {label} 训练截止 {cut:%Y%m%d}, 评估窗含 {len(bad)} 个 <= 截止日"
            f" (最早 {min(bad):%Y%m%d}, 最晚 {max(bad):%Y%m%d}) → 含未来信息, 结论无效"
        )
