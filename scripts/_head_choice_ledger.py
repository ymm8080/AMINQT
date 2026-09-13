"""cls/reg 头选择台账 (0912 夜用户令) — 双管线四格统一追加式 CSV.

每次重训 (legacy _retrain_legacy_full / parallel 概率头重训) 追加一行:
双头逐视界 (3d/5d/10d, 用户令) + 双头族加权聚合 OOS 指标 + 该批自选的
serving 头。用户令: "每天各格 cls 与 reg 结果 + 当日选了哪个头都要有表可查"。
parallel 侧头名映射: cls≈prob 概率头, reg≈mag 幅度头 (列名共用)。
幂等键 = (pipeline, board, tag); WORM: 只追加, 同键重跑跳过。
"""

from __future__ import annotations

import csv
import time

from config.settings import data_others_path

LEDGER_PATH = data_others_path("head_choice_ledger.csv")
_HORIZONS = (3, 5, 10)
LEDGER_COLUMNS = (
    "ts",
    "pipeline",
    "board",
    "tag",
    # 逐视界双头 OOS (0912 夜用户令: 3d/5d/10d 都要, 非 10d 单列)
    *[f"ic_reg_{k}d" for k in _HORIZONS],
    *[f"ic_cls_{k}d" for k in _HORIZONS],
    # 双头族 LABEL_WEIGHTS 加权聚合 (LEGACY 闸口径; parallel 同列记录)
    "weighted_ic_reg",
    "weighted_ic_cls",
    "chosen",
    "gate_pass",
    "switched",
    "n_features",
)


def append_head_choice_row(
    pipeline: str,
    board: str,
    tag: str,
    *,
    weighted_ic_reg,
    weighted_ic_cls,
    chosen: str,
    gate_pass: bool,
    switched: bool,
    ics: dict | None = None,
    n_features=None,
    ledger_path=None,
) -> bool:
    """追加一行; (pipeline, board, tag) 已存在 → 跳过返回 False (WORM).

    ics: 逐头逐视界 {"{k}d_{reg|cls}": ic} (LEGACY validate_oos["ics"] 同形;
    parallel 侧 reg=mag 键, cls=prob 键)。缺项列留空。
    """
    path = ledger_path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if (
                    r.get("pipeline") == pipeline
                    and r.get("board") == board
                    and r.get("tag") == tag
                ):
                    return False
    ics = ics or {}
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": pipeline,
        "board": board,
        "tag": tag,
        **{
            f"ic_{s}_{k}d": ics.get(f"{k}d_{s}")
            for s in ("reg", "cls")
            for k in _HORIZONS
        },
        "weighted_ic_reg": weighted_ic_reg,
        "weighted_ic_cls": weighted_ic_cls,
        "chosen": chosen,
        "gate_pass": bool(gate_pass),
        "switched": bool(switched),
        "n_features": n_features,
    }
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return True
