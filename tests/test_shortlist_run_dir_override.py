"""Tests for scripts/_shortlist_t5_t10 交付端 --run-dir 补跑 override (2026-09-05).

背景: 09-05 隔日补跑 09-04 链 — run_dir 按实际运行日命名 (20260905_*), 交付端按
tag 前缀找 20260904_* 永远找不到 → 守卫拒绝 (当日 parallel 清单缺失). 修复 =
--run-dir 显式指定 run_dir, 声明 "目录内数据即该 tag 交易日数据".
"""

from pathlib import Path

import scripts._shortlist_t5_t10 as sl
from scripts._shortlist_t5_t10 import _resolve_run_dir_for_delivery


def _make_fullrun(base: Path, name: str, with_shortlist: bool = True) -> Path:
    d = base / "BACKTESTING RESULT" / name
    d.mkdir(parents=True)
    if with_shortlist:
        (d / "shortlist_main.csv").write_text("symbol\n600001\n", encoding="utf-8")
    return d


def test_override_takes_explicit_run_dir_even_if_date_mismatches(tmp_path, monkeypatch):
    """补跑核心场景: 目录日期前缀 ≠ tag, 显式指定即采纳."""
    monkeypatch.setattr(sl, "DATA_OTHERS_DIR", tmp_path)
    mismatched = _make_fullrun(tmp_path, "20260905_141710")
    assert _resolve_run_dir_for_delivery("20260904", str(mismatched)) == mismatched


def test_override_missing_dir_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(sl, "DATA_OTHERS_DIR", tmp_path)
    assert _resolve_run_dir_for_delivery("20260904", str(tmp_path / "nope")) is None


def test_no_override_finds_latest_prefix_run_dir(tmp_path, monkeypatch):
    """常态路径不变: 按 tag 前缀挑最新, 不受其它日期 run_dir 干扰."""
    monkeypatch.setattr(sl, "DATA_OTHERS_DIR", tmp_path)
    d04_latest = _make_fullrun(tmp_path, "20260904_101010")
    _make_fullrun(tmp_path, "20260904_090000")
    _make_fullrun(tmp_path, "20260905_141710")
    assert _resolve_run_dir_for_delivery("20260904", None) == d04_latest


def test_no_override_no_match_returns_none(tmp_path, monkeypatch):
    """守卫语义保留: 无当日 run_dir → None (调用方大声失败, 拒绝交付旧数据)."""
    monkeypatch.setattr(sl, "DATA_OTHERS_DIR", tmp_path)
    _make_fullrun(tmp_path, "20260905_141710")
    assert _resolve_run_dir_for_delivery("20260904", None) is None
