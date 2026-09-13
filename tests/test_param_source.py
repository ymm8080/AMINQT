# -*- coding: utf-8 -*-
"""LEGACY 超参 json 旋钮源 (app.pipeline1.param_source, 0913 #8) — 单测.

rank_source 式三件套: WORM 只增 / trained_through 新鲜度闸 / 异常 fail-open
回代码表。集成面: dual_track_trainer.model_params 叠 resolve_param_override
(无 json = 零行为变化, 代码表 NUM_LEAVES_OVERRIDE/PARAMS_OVERRIDE 不动).
"""

from __future__ import annotations

import json

from app.pipeline1 import param_source
from app.pipeline1.dual_track_trainer import model_params


def _payload(overrides=None, trained_through="2026-09-12"):
    return {
        "board": "main",
        "trained_through": trained_through,
        "overrides": overrides
        if overrides is not None
        else {"5d_cls": {"num_leaves": 63}},
        "evidence": {"sweep": "test", "judge": "TOP10 实净"},
    }


def test_save_worm_same_ts_suffix(tmp_path):
    p1 = param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    p2 = param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    assert p1 != p2
    assert (
        json.loads(p1.read_text(encoding="utf-8"))["overrides"]["5d_cls"]["num_leaves"]
        == 63
    )


def test_load_latest_and_board_isolation(tmp_path):
    param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    param_source.save_param_source(
        "main",
        _payload({"10d_cls": {"num_leaves": 7}}),
        directory=tmp_path,
        ts="20260913_0300",
    )
    rec = param_source.load_latest_param_source("main", directory=tmp_path)
    assert rec["overrides"]["10d_cls"]["num_leaves"] == 7
    assert param_source.load_latest_param_source("dual", directory=tmp_path) is None


def test_load_latest_bad_json_returns_none(tmp_path):
    (tmp_path / "main_param_source_20260913_0200.json").write_text(
        "not-json", encoding="utf-8"
    )
    assert param_source.load_latest_param_source("main", directory=tmp_path) is None


def test_resolve_no_json_returns_empty(tmp_path):
    assert (
        param_source.resolve_param_override("main", "5d_cls", directory=tmp_path) == {}
    )


def test_resolve_fresh_json_returns_override(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"}
    )
    param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    got = param_source.resolve_param_override(
        "main", "5d_cls", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
    )
    assert got == {"num_leaves": 63}
    # json 内无该 kind 条目 → {} (不动该头)
    assert (
        param_source.resolve_param_override(
            "main", "3d_reg", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        == {}
    )


def test_resolve_stale_json_falls_back_code_table(tmp_path):
    param_source.save_param_source(
        "main",
        _payload(trained_through="2026-06-01"),
        directory=tmp_path,
        ts="20260601_0200",
    )
    assert (
        param_source.resolve_param_override(
            "main", "5d_cls", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        == {}
    )


def test_resolve_knob_code_forces_code_table(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"}
    )
    param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    assert (
        param_source.resolve_param_override(
            "main", "5d_cls", as_of="2026-09-13", knob="code", directory=tmp_path
        )
        == {}
    )


def test_resolve_missing_trained_through_falls_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"}
    )
    param_source.save_param_source(
        "main",
        {"board": "main", "overrides": {"5d_cls": {"num_leaves": 63}}},
        directory=tmp_path,
        ts="20260913_0200",
    )
    assert (
        param_source.resolve_param_override(
            "main", "5d_cls", as_of="2026-09-13", directory=tmp_path
        )
        == {}
    )
    assert "代码表" in capsys.readouterr().out


def test_model_params_code_table_without_json(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.pipeline1.param_source.param_source_dir", lambda directory=None: tmp_path
    )
    p = model_params("main", "5d_cls")
    assert p["num_leaves"] == 15  # 代码表 NUM_LEAVES_OVERRIDE (0808 定案)
    assert p["objective"] == "binary"


def test_model_params_json_overlays_code_table(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.pipeline1.param_source.param_source_dir", lambda directory=None: tmp_path
    )
    monkeypatch.setattr(
        "config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"}
    )
    param_source.save_param_source(
        "main",
        _payload({"5d_cls": {"num_leaves": 63, "min_child_samples": 40}}),
        directory=tmp_path,
        ts="20260913_0200",
    )
    p = model_params("main", "5d_cls")
    assert p["num_leaves"] == 63  # json 压过代码表 15
    assert p["min_child_samples"] == 40
    assert p["objective"] == "binary"  # 家族基底不动
    # 未覆盖的 kind 保持代码表
    assert model_params("main", "10d_cls")["num_leaves"] == 15


def test_model_params_knob_code_restores_code_table(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.pipeline1.param_source.param_source_dir", lambda directory=None: tmp_path
    )
    monkeypatch.setattr(
        "config.settings.LEGACY_PARAM_SOURCE", {"main": "code", "dual": "auto"}
    )
    param_source.save_param_source(
        "main", _payload(), directory=tmp_path, ts="20260913_0200"
    )
    assert model_params("main", "5d_cls")["num_leaves"] == 15


# ── [0913 #8] cls 半衰期旋钮 (resolve_cls_half_life; 独立顶层字段非 LGBM) ──
def test_cls_half_life_fresh_json_returns_value(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"})
    param_source.save_param_source(
        "main", {**_payload(), "cls_half_life_days": 60}, directory=tmp_path, ts="20260913_0300"
    )
    assert (
        param_source.resolve_cls_half_life(
            "main", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        == 60
    )


def test_cls_half_life_absent_or_no_json_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"})
    param_source.save_param_source("main", _payload(), directory=tmp_path, ts="20260913_0300")
    assert (
        param_source.resolve_cls_half_life(
            "main", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        is None
    )
    assert (
        param_source.resolve_cls_half_life(
            "dual", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        is None
    )


def test_cls_half_life_stale_or_illegal_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"})
    param_source.save_param_source(
        "main",
        {**_payload(trained_through="2026-06-01"), "cls_half_life_days": 60},
        directory=tmp_path,
        ts="20260601_0300",
    )
    assert (
        param_source.resolve_cls_half_life(
            "main", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        is None
    )
    param_source.save_param_source(
        "main", {**_payload(), "cls_half_life_days": "60"}, directory=tmp_path, ts="20260913_0400"
    )
    assert (
        param_source.resolve_cls_half_life(
            "main", as_of="2026-09-13", directory=tmp_path, max_stale_days=45
        )
        is None
    )


def test_cls_half_life_knob_code_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr("config.settings.LEGACY_PARAM_SOURCE", {"main": "auto", "dual": "auto"})
    param_source.save_param_source(
        "main", {**_payload(), "cls_half_life_days": 60}, directory=tmp_path, ts="20260913_0300"
    )
    assert param_source.resolve_cls_half_life("main", knob="code", directory=tmp_path) is None
