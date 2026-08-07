"""predict_runner 模型包定位 — current_meta 解析 + find_bundles 回退."""

from __future__ import annotations

from pathlib import Path

from app.pipeline1.predict_runner import resolve_current_bundles


def _touch(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"dummy")
    return p


def test_resolve_from_current_meta(tmp_path, monkeypatch):
    _touch(tmp_path, "main_20260805_a.pkl")
    _touch(tmp_path, "dual_20260805_a.pkl")
    monkeypatch.setattr(
        "app.pipeline1.model_meta.load_modules",
        lambda: {
            "main": {"file": "main_20260805_a.pkl", "tag": "20260805_a"},
            "dual": {"file": "dual_20260805_a.pkl", "tag": "20260805_a"},
        },
    )

    bundles = resolve_current_bundles(model_dir=str(tmp_path))

    assert bundles == {
        "main": str(tmp_path / "main_20260805_a.pkl"),
        "dual": str(tmp_path / "dual_20260805_a.pkl"),
    }


def test_meta_points_to_missing_file_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.pipeline1.model_meta.load_modules",
        lambda: {"main": {"file": "main_ghost.pkl", "tag": "x"}},
    )
    monkeypatch.setattr(
        "app.pipeline1.predict_runner.find_bundles",
        lambda model_dir: {"main": "models/pipeline1/main_fallback.pkl"},
    )

    assert resolve_current_bundles(model_dir=str(tmp_path)) == {
        "main": "models/pipeline1/main_fallback.pkl"
    }


def test_empty_meta_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline1.model_meta.load_modules", lambda: {})
    monkeypatch.setattr(
        "app.pipeline1.predict_runner.find_bundles",
        lambda model_dir: {},
    )

    assert resolve_current_bundles(model_dir=str(tmp_path)) == {}


def test_corrupt_meta_falls_back(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("corrupt json")

    monkeypatch.setattr("app.pipeline1.model_meta.load_modules", boom)
    monkeypatch.setattr(
        "app.pipeline1.predict_runner.find_bundles",
        lambda model_dir: {"main": "models/pipeline1/main_fallback.pkl"},
    )

    assert resolve_current_bundles(model_dir=str(tmp_path)) == {
        "main": "models/pipeline1/main_fallback.pkl"
    }
