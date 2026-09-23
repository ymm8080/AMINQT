# -*- coding: utf-8 -*-
"""GENIOUS 交付 --dry-run 状态文件契约 (0923 缺陷1)。

契约: --dry-run = 只打印不落任何文件 ⇒ 生产状态文件
logs/genious_{tag}.state.json 不得被创建/改写, 含 running/failed/skipped
等失败路径。看门狗/新鲜度判据把该文件当真 —— 一次演练会吃掉当天真实状态
(盘上 genious_20260923.state.json 曾被 dry_run 覆盖且当天无 xlsx)。

真入口级验证见 tmp_t/_0923_genious_dryrun_state_verify.py (subprocess 跑真实
CLI); 本文件守单元级不变量: _write_state 的抑制 + main() 的置位接线。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import _genious_excel as ge  # noqa: E402


def test_dry_run_write_state_is_suppressed(tmp_path, monkeypatch):
    """_DRY_RUN=True → 任何 status 都不得落盘 (含失败终态)。"""
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", True)
    ge._write_state("20260923", "dry_run", s1=5)
    ge._write_state("20260923", "running")
    ge._write_state("20260923", "failed", reason="freshness")
    assert not (tmp_path / "genious_20260923.state.json").exists()
    assert not list(tmp_path.glob("genious_*.state.json"))


def test_non_dry_run_write_state_persists(tmp_path, monkeypatch):
    """_DRY_RUN=False → 照旧落生产状态路径 (正常跑不许被改坏)。"""
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", False)
    ge._write_state("20260923", "ok", s1=3)
    p = tmp_path / "genious_20260923.state.json"
    assert p.exists()
    assert '"status": "ok"' in p.read_text(encoding="utf-8")


def test_main_sets_dry_run_flag_from_argv(tmp_path, monkeypatch):
    """main() 按 --dry-run 置位 _DRY_RUN, 且该模式下早退路径不落状态。

    GENIOUS.enable=False 是最便宜的真实 main() 早退路径 (写 disabled 后 return)。"""
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", False)
    monkeypatch.setitem(ge.GENIOUS, "enable", False)
    monkeypatch.setattr(sys, "argv", ["_genious_excel.py", "20260923", "--dry-run"])
    assert ge.main() == 0
    assert ge._DRY_RUN is True
    assert not (tmp_path / "genious_20260923.state.json").exists()


def test_main_early_exit_writes_state_when_not_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ge, "LOG_DIR", tmp_path)
    monkeypatch.setattr(ge, "_DRY_RUN", True)
    monkeypatch.setitem(ge.GENIOUS, "enable", False)
    monkeypatch.setattr(sys, "argv", ["_genious_excel.py", "20260923"])
    assert ge.main() == 0
    assert ge._DRY_RUN is False
    p = tmp_path / "genious_20260923.state.json"
    assert p.exists() and '"status": "disabled"' in p.read_text(encoding="utf-8")
