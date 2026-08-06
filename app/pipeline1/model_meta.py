# -*- coding: utf-8 -*-
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
"""

import json

META_PATH = "models/pipeline1/current_meta.json"


def load_modules(meta_path: str = META_PATH) -> dict:
    """{board: {tag, file, updated}}; 缺失/损坏返回 {} (调用方回退 'na')."""
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
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
