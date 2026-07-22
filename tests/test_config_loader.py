# -*- coding: utf-8 -*-
"""Tests for app/core/config_loader — YAML 加载 + 点号取值."""

import app.core.config_loader as cl
from app.core.config_loader import get, load_config


class TestLoadConfig:
    def test_load_existing_config(self):
        cfg = load_config("selection_config")
        assert isinstance(cfg, dict)
        assert cfg.get("universe") == "all"
        assert "model" in cfg

    def test_missing_file_returns_empty(self):
        assert load_config("no_such_config_xyz") == {}

    def test_load_from_custom_dir(self, tmp_path, monkeypatch):
        (tmp_path / "custom.yaml").write_text(
            "a: 1\nnested:\n  b: hello\n", encoding="utf-8"
        )
        monkeypatch.setattr(cl, "CONFIG_DIR", tmp_path)
        cfg = cl.load_config("custom")
        assert cfg == {"a": 1, "nested": {"b": "hello"}}

    def test_empty_yaml_returns_empty(self, tmp_path, monkeypatch):
        (tmp_path / "empty.yaml").write_text("", encoding="utf-8")
        monkeypatch.setattr(cl, "CONFIG_DIR", tmp_path)
        assert cl.load_config("empty") == {}

    def test_non_dict_top_level_returns_empty(self, tmp_path, monkeypatch):
        (tmp_path / "list.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
        monkeypatch.setattr(cl, "CONFIG_DIR", tmp_path)
        assert cl.load_config("list") == {}


class TestGet:
    CFG = {
        "scoring": {"model_weight": 0.6, "rules": {"top_n": 20}},
        "universe": "all",
        "zero_val": 0,
        "none_val": None,
    }

    def test_nested_path(self):
        assert get(self.CFG, "scoring.model_weight") == 0.6
        assert get(self.CFG, "scoring.rules.top_n") == 20

    def test_top_level(self):
        assert get(self.CFG, "universe") == "all"

    def test_missing_key_returns_default(self):
        assert get(self.CFG, "scoring.missing", 0.5) == 0.5
        assert get(self.CFG, "a.b.c", "dflt") == "dflt"

    def test_non_dict_intermediate_returns_default(self):
        # "universe" 是字符串, 不能再下钻
        assert get(self.CFG, "universe.sub", 42) == 42

    def test_falsy_values_not_replaced_by_default(self):
        assert get(self.CFG, "zero_val", 99) == 0
        assert get(self.CFG, "none_val", 99) is None

    def test_empty_config(self):
        assert get({}, "any.key", "x") == "x"

    def test_non_dict_config(self):
        assert get(None, "a.b", "x") == "x"
