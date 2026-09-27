# -*- coding: utf-8 -*-
"""GENIOUS 冠军四段空表 (0 票) 不得裸奔 (0928 缺陷2)。

判词 (见 memory genious-champion-empty-verdict): 空表 **不是** 缺陷 —— 268 交易日
基线 champ==0 占 26.1%, 含 20251106–20251119 连续 9 日。所以终态**不许**标 ERROR/失败
(会天天误报)。但空表也不该与"有票的 ok"完全不可分: 交付物只留表头时读者分不清
"今日无票"与"链路坏了"。

契约 (三处同时暴露, 不动 status/rc):
  1. state 文件带 `empty: true` (机器可判)
  2. A1 banner 追加 EMPTY_NOTE (人在交付物里可读)
  3. 日志升 WARNING (日志侧可扫) —— 这一处在 main() 里, 需面板, 不在本文件守
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline1 import kongduo_triggers as kt  # noqa: E402
from scripts import _genious_excel as ge  # noqa: E402


def _empty_s1() -> pd.DataFrame:
    """与 build_delivery 同构的空表 (只表头)。"""
    return pd.DataFrame(columns=["排名", *kt.DISPLAY_COLUMNS])


def _one_row_s1() -> pd.DataFrame:
    row = {
        "排名": 1,
        "symbol": "600000",
        "涨闸": "过闸",
        "层": kt.CH3_T3_DEEP_QUIET,
        "触发器": "T3",
    }
    return pd.DataFrame([row])


def test_write_state_records_empty_flag(tmp_path, monkeypatch):
    """空表 → state 文件显式 empty=true (status 仍 ok, 不改终态)。"""
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", False)
    ge._write_state("20260924", "ok", s1=0, empty=True, layers={})
    payload = json.loads((tmp_path / "genious_20260924.state.json").read_text("utf-8"))
    assert payload["status"] == "ok"  # 终态不动: 26% 日为空, 标失败=天天误报
    assert payload["empty"] is True
    assert payload["s1"] == 0


def test_write_state_nonempty_flag_false(tmp_path, monkeypatch):
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", False)
    ge._write_state("20260924", "ok", s1=7, empty=False, layers={"CH3": 7})
    payload = json.loads((tmp_path / "genious_20260924.state.json").read_text("utf-8"))
    assert payload["empty"] is False


def test_empty_sheet1_banner_carries_note(tmp_path):
    """空表落在 xlsx 上时, A1 banner 必须含"空属正常"的说明 (人读交付物)。"""
    from openpyxl import load_workbook

    fp = ge.write_xlsx(_empty_s1(), "20260924", list_dir=str(tmp_path))
    ws = load_workbook(fp)["冠军四段"]
    a1 = str(ws.cell(row=1, column=1).value or "")
    assert ge.BANNER1 in a1, "空表仍要有常规 banner"
    assert ge.EMPTY_NOTE in a1, "空表必须带 EMPTY_NOTE"
    assert "26%" in a1 and "放宽层定义" in a1
    assert ws.cell(row=3, column=1).value == "排名", "表头仍在 (不是把表写坏)"


def test_nonempty_sheet1_banner_has_no_empty_note(tmp_path):
    """有票的日子 banner 一个字都不许多 —— 不许把告警撒到正常交付上。"""
    from openpyxl import load_workbook

    fp = ge.write_xlsx(_one_row_s1(), "20260924", list_dir=str(tmp_path))
    ws = load_workbook(fp)["冠军四段"]
    a1 = str(ws.cell(row=1, column=1).value or "")
    assert a1 == ge.BANNER1
    assert ge.EMPTY_NOTE not in a1
