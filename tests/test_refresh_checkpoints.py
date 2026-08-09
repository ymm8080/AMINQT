"""_refresh_parallel_checkpoints.py 编排契约测试 (mock 重组件).

内存分期回归: refresh 必须逐板块 build_board_slice (main→释放→dual),
且每板块用对应检查点路径. 该脚本的参考内存是在 build_board_slice 之前
用 dict.pop 释放已处理板块的清洗切片, 避免 main/dual 两个切片同时常驻.
"""

from __future__ import annotations

import importlib.util
import os

import pandas as pd

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SCRIPT = os.path.join(_REPO, "scripts", "_refresh_parallel_checkpoints.py")


def _load_refresh_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("refresh_ckpt_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_board(board: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": ["600000"], "date": [pd.Timestamp("2026-08-07")], "board": [board]}
    )


class TestRefreshCheckpoints:
    def test_builds_main_then_dual_in_order(self, monkeypatch, capsys):
        mod = _load_refresh_module(monkeypatch)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod.pd, "read_parquet", lambda path: _tiny_board("main"))
        main_df = _tiny_board("main")
        dual_df = _tiny_board("dual")
        monkeypatch.setattr(
            mod.CleaningPipeline,
            "run_train",
            lambda self, df, board=None: (main_df, dual_df),
        )
        monkeypatch.setattr(mod, "FeatureEngineV35", lambda: object())

        calls = []
        monkeypatch.setattr(
            mod,
            "build_board_slice",
            lambda cleaner, fe, bdf, board, ckpt: (
                calls.append((board, ckpt, bdf)) or bdf
            ),
        )

        rc = mod.main()

        assert rc == 0
        assert [c[0] for c in calls] == ["main", "dual"]  # 逐板块, 有先有后
        # main 板块用 MAIN_CHECKPOINT, dual 用 DUAL_CHECKPOINT, 不得串
        assert calls[0][1] == mod.MAIN_CHECKPOINT and calls[0][2].equals(main_df)
        assert calls[1][1] == mod.DUAL_CHECKPOINT and calls[1][2].equals(dual_df)
        out = capsys.readouterr().out
        assert "main rows=1" in out  # run_train 行数打印保留

    def test_empty_board_skipped(self, monkeypatch):
        mod = _load_refresh_module(monkeypatch)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod.pd, "read_parquet", lambda path: _tiny_board("main"))
        monkeypatch.setattr(
            mod.CleaningPipeline,
            "run_train",
            lambda self, df, board=None: (_tiny_board("main"), pd.DataFrame()),
        )
        monkeypatch.setattr(mod, "FeatureEngineV35", lambda: object())

        calls = []
        monkeypatch.setattr(
            mod,
            "build_board_slice",
            lambda cleaner, fe, bdf, board, ckpt: calls.append(board) or bdf,
        )

        rc = mod.main()
        assert rc == 0
        assert calls == ["main"]  # dual 空 → 跳过, 不调 build_board_slice


class TestRefreshFingerprintSkip:
    """指纹 + 无新增交易日 → 跳过重建 (参数/代码变了指纹必变 → 必重建)."""

    def _mod(self, monkeypatch, tmp_path):
        mod = _load_refresh_module(monkeypatch)
        main_ck = tmp_path / "main.parquet"
        dual_ck = tmp_path / "dual.parquet"
        main_ck.write_bytes(b"x")
        dual_ck.write_bytes(b"x")
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(main_ck))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(dual_ck))
        monkeypatch.setattr(mod, "compute_fingerprint", lambda: "fp1")
        return mod

    def test_skip_when_no_new_data_and_fp_match(self, monkeypatch, tmp_path, capsys):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "fp1", "latest_date": "2026-08-07"},
        )
        monkeypatch.setattr(
            mod.pd,
            "read_parquet",
            lambda path, **kw: pd.DataFrame({"date": [pd.Timestamp("2026-08-07")]}),
        )
        assert mod._skip_if_unchanged(False) is True
        assert "[skip]" in capsys.readouterr().out

    def test_force_always_rebuilds(self, monkeypatch, tmp_path):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_read_fingerprint_meta", lambda: None)
        assert mod._skip_if_unchanged(True) is False

    def test_fp_mismatch_rebuilds(self, monkeypatch, tmp_path):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "OLD", "latest_date": "2026-08-07"},
        )
        assert mod._skip_if_unchanged(False) is False

    def test_new_data_rebuilds(self, monkeypatch, tmp_path):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "fp1", "latest_date": "2026-08-07"},
        )
        monkeypatch.setattr(
            mod.pd,
            "read_parquet",
            lambda path, **kw: pd.DataFrame({"date": [pd.Timestamp("2026-08-08")]}),
        )
        assert mod._skip_if_unchanged(False) is False

    def test_missing_checkpoint_rebuilds(self, monkeypatch, tmp_path):
        mod = _load_refresh_module(monkeypatch)
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(tmp_path / "no_main.parquet"))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(tmp_path / "no_dual.parquet"))
        monkeypatch.setattr(mod, "compute_fingerprint", lambda: "fp1")
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "fp1", "latest_date": "2026-08-07"},
        )
        assert mod._skip_if_unchanged(False) is False

    def test_main_skips_without_stale_or_rebuild(self, monkeypatch, tmp_path):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "fp1", "latest_date": "2026-08-07"},
        )
        monkeypatch.setattr(
            mod.pd,
            "read_parquet",
            lambda path, **kw: pd.DataFrame({"date": [pd.Timestamp("2026-08-07")]}),
        )
        called = []
        monkeypatch.setattr(
            mod.os, "rename", lambda a, b: called.append("rename") or None
        )
        monkeypatch.setattr(
            mod,
            "build_board_slice",
            lambda *a, **k: called.append("build") or _tiny_board("main"),
        )
        assert mod.main() == 0
        assert called == []  # 未改名、未重建

    def test_fingerprint_covers_feature_config(self, monkeypatch, tmp_path):
        # fe.build 从 config/settings.py 读 LHB_V2_SPEC → settings 变化必须触发重建,
        # 否则 LHB 特征改了却静默跳过检查点重建 (质量风险). 该文件必须留在指纹列表.
        mod = _load_refresh_module(monkeypatch)
        assert "config/settings.py" in mod._FINGERPRINT_FILES

    def test_rebuild_writes_fingerprint_meta(self, monkeypatch, tmp_path, capsys):
        mod = self._mod(monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(
            mod.pd, "read_parquet", lambda path, **kw: _tiny_board("main")
        )
        monkeypatch.setattr(
            mod.CleaningPipeline,
            "run_train",
            lambda self, df, board=None: (_tiny_board("main"), _tiny_board("dual")),
        )
        monkeypatch.setattr(mod, "FeatureEngineV35", lambda: object())
        monkeypatch.setattr(
            mod,
            "build_board_slice",
            lambda cleaner, fe, bdf, board, ckpt: _tiny_board(board),
        )
        written = {}
        monkeypatch.setattr(
            mod,
            "_write_fingerprint_meta",
            lambda latest: written.update({"latest": latest}),
        )
        assert mod.main() == 0
        assert written.get("latest") == "2026-08-07"  # 重建后写指纹 (日期=最新交易日)
