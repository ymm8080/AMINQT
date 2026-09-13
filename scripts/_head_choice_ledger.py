"""cls/reg 头选择台账 (0912 夜用户令) — 双管线四格统一追加式 CSV.

每次重训 (legacy _retrain_legacy_full / parallel 概率头重训) 追加一行:
双头 OOS 指标 + 该批自选的 serving 头。用户令: "每天各格 cls 与 reg 结果
+ 当日选了哪个头都要有表可查"。
幂等键 = (pipeline, board, tag); WORM: 只追加, 同键重跑跳过。
"""

from __future__ import annotations

import csv
import time

from config.settings import data_others_path

LEDGER_PATH = data_others_path("head_choice_ledger.csv")
LEDGER_COLUMNS = (
    "ts",
    "pipeline",
    "board",
    "tag",
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
    n_features=None,
    ledger_path=None,
) -> bool:
    """追加一行; (pipeline, board, tag) 已存在 → 跳过返回 False (WORM)."""
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
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline": pipeline,
        "board": board,
        "tag": tag,
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
