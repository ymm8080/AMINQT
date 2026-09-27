"""污染窗口护栏 (app/pipeline1/model_meta.py) 单测.

背景: bundle 在面板 date.max() 上训练, tag 默认训练当天 —— 评估窗内任一日期 <= tag
即含未来信息 (模型训练时见过该日), 回放数字是样本内。该规则此前只写在文档与各脚本的
硬编码常量里, 无任何代码强制。本测试锁住护栏行为。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.model_meta import (
    ContaminatedWindowError,
    as_date,
    assert_clean_window,
    board_cutoff,
    board_tag,
    load_modules,
    module_id,
    parse_tag_date,
)

META = {
    "main": {"tag": "20260903", "file": "main_20260903.pkl", "updated": "x"},
    "dual": {"tag": "20260913", "file": "dual_20260913.pkl", "updated": "x"},
}

_REAL_META = (
    Path(__file__).resolve().parents[1] / "models" / "pipeline1" / "current_meta.json"
)


@pytest.fixture()
def meta_file(tmp_path):
    fp = tmp_path / "current_meta.json"
    fp.write_text(json.dumps(META), encoding="utf-8")
    return str(fp)


# ── parse_tag_date ────────────────────────────────────────────────────────
def test_parse_tag_date_plain():
    assert parse_tag_date("20260903") == dt.date(2026, 9, 3)


def test_parse_tag_date_with_suffix():
    assert parse_tag_date("20260805_q234") == dt.date(2026, 8, 5)


@pytest.mark.parametrize("bad", ["", None, "na", "2026", "abc"])
def test_parse_tag_date_unparseable(bad):
    assert parse_tag_date(bad) is None


def test_parse_tag_date_rejects_impossible_date():
    assert parse_tag_date("20261399") is None


# ── board_cutoff ──────────────────────────────────────────────────────────
def test_board_cutoff_maps_board_to_track(meta_file):
    assert board_cutoff("main", meta_path=meta_file) == dt.date(2026, 9, 3)
    assert board_cutoff("GEM", meta_path=meta_file) == dt.date(2026, 9, 13)
    assert board_cutoff("STAR", meta_path=meta_file) == dt.date(2026, 9, 13)


def test_board_cutoff_unknown_board_is_none(meta_file):
    assert board_cutoff("NOPE", meta_path=meta_file) is None


# ── as_date ───────────────────────────────────────────────────────────────
def test_as_date_coercions():
    d = dt.date(2026, 9, 13)
    assert as_date("20260913") == d
    assert as_date("2026-09-13") == d
    assert as_date(d) == d
    assert as_date(dt.datetime(2026, 9, 13, 15, 30)) == d
    assert as_date(pd.Timestamp("2026-09-13 15:30")) == d
    assert as_date(np.datetime64("2026-09-13")) == d


def test_as_date_garbage_is_none():
    assert as_date("na") is None


# ── assert_clean_window ───────────────────────────────────────────────────
def test_accepts_window_after_cutoff():
    assert_clean_window(("20260914", "20260922"), "20260913")


def test_accepts_first_clean_day():
    assert_clean_window(["20260914"], dt.date(2026, 9, 13))


def test_rejects_cutoff_day_itself():
    """截止日当天仍属训练窗 (模型在面板 date.max() 上训练) → 必须拒绝."""
    with pytest.raises(ContaminatedWindowError):
        assert_clean_window(["20260913"], "20260913")


def test_rejects_the_real_ab_window():
    """原 _0924_legacy_blend_ab 的 TE 段 2026-01-01→09-21 与 dual 训练截止重叠."""
    with pytest.raises(ContaminatedWindowError) as ei:
        assert_clean_window(("20260101", "20260921"), "20260913", label="dual bundle")
    assert "污染窗口" in str(ei.value)
    assert "20260913" in str(ei.value)


def test_rejects_unprovable_cutoff():
    """无法证明洁净 = 拒绝出数 (不静默放行)."""
    with pytest.raises(ContaminatedWindowError):
        assert_clean_window(["20260914"], None)


def test_rejects_unparseable_date():
    with pytest.raises(ContaminatedWindowError):
        assert_clean_window(["not-a-date"], "20260913")


def test_accepts_datetime64_array():
    assert_clean_window(np.array(["2026-09-14", "2026-09-22"]), dt.date(2026, 9, 13))


# ── 既有行为回归 (改动未破坏 board_tag / module_id / load_modules) ─────────
def test_module_id_and_board_tag_unchanged(meta_file):
    mods = load_modules(meta_file)
    assert board_tag(mods, "main") == "20260903"
    assert board_tag(mods, "GEM") == "20260913"
    assert module_id(mods) == "M20260903__D20260913"


# ── 真实 current_meta.json (集成) ─────────────────────────────────────────
def test_real_meta_cutoff_is_usable():
    """生产 meta 必须能解析出截止日 —— 否则护栏会把所有评估都拒掉."""
    assert board_cutoff("main", meta_path=str(_REAL_META)) is not None
    assert board_cutoff("GEM", meta_path=str(_REAL_META)) is not None
