"""_refresh_parallel_checkpoints.py 编排契约测试 (mock 重组件).

内存分期回归: refresh 必须逐板块 build_board_slice (main→释放→dual),
且每板块用对应检查点路径. 该脚本的参考内存是在 build_board_slice 之前
用 dict.pop 释放已处理板块的清洗切片, 避免 main/dual 两个切片同时常驻.
"""

from __future__ import annotations

import importlib.util
import os

import pandas as pd
import pytest

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_SCRIPT = os.path.join(_REPO, "scripts", "_refresh_parallel_checkpoints.py")


@pytest.fixture
def refresh_mod(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("refresh_ckpt_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 锁/指纹 meta 一律重定向到 tmp: 测试绝不触碰真实 data/ 下文件. 未 mock
    # _write_fingerprint_meta 的旧测试曾把假 latest_date (08-07) 写进真实 meta,
    # 导致下次 refresh 误判数据落后而白白全量重建 (2026-08-25 事故).
    monkeypatch.setattr(mod, "_LOCK_FILE", str(tmp_path / "_refresh_parallel.lock"))
    monkeypatch.setattr(
        mod, "_FINGERPRINT_META", str(tmp_path / "_diag_stage_3y.fingerprint.json")
    )
    return mod


def _tiny_board(board: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"symbol": ["600000"], "date": [pd.Timestamp("2026-08-07")], "board": [board]}
    )


def _redirect_fresh_artifacts(mod, monkeypatch, tmp_path, date: str = "2026-08-07"):
    """重建成功路径现在带检查点新鲜度后置断言 (2026-09-02): 两检查点落盘且
    max(date)==面板 max. 走重建路径的 mock 测试把两检查点与面板都指到 tmp 迷你
    parquet (同日), 保持原测试目的 (编排/锁/指纹) 不变, 也绝不读真实多 GB 文件."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame({"date": [pd.Timestamp(date)]})
    for name in ("main", "dual", "panel"):
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            str(tmp_path / f"{name}.parquet"),
        )
    monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(tmp_path / "main.parquet"))
    monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(tmp_path / "dual.parquet"))
    monkeypatch.setattr(mod, "PANEL_V3_PATH", str(tmp_path / "panel.parquet"))


class TestRefreshCheckpoints:
    def test_builds_dual_then_main_in_order(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        mod = refresh_mod
        _redirect_fresh_artifacts(mod, monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod, "load_panel_v3", lambda **kw: _tiny_board("main"))
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
        # 内存优化定案 2026-08-11: dual 先建 → main 后建 (main 1.4M 行是内存大头,
        # 后建时另一板块清洗帧已弹出释放, 防 block consolidate OOM)
        assert [c[0] for c in calls] == ["dual", "main"]  # 逐板块, 有先有后
        # 板块检查点路径不得串: dual→DUAL_CHECKPOINT, main→MAIN_CHECKPOINT
        assert calls[0][1] == mod.DUAL_CHECKPOINT and calls[0][2].equals(dual_df)
        assert calls[1][1] == mod.MAIN_CHECKPOINT and calls[1][2].equals(main_df)
        out = capsys.readouterr().out
        assert "run_train[main]: rows=1" in out  # run_train 行数打印保留

    def test_empty_board_skipped(self, monkeypatch, refresh_mod, tmp_path):
        mod = refresh_mod
        _redirect_fresh_artifacts(mod, monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod, "load_panel_v3", lambda **kw: _tiny_board("main"))
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

    def _mod(self, mod, monkeypatch, tmp_path):
        main_ck = tmp_path / "main.parquet"
        dual_ck = tmp_path / "dual.parquet"
        main_ck.write_bytes(b"x")
        dual_ck.write_bytes(b"x")
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(main_ck))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(dual_ck))
        monkeypatch.setattr(mod, "compute_fingerprint", lambda: "fp1")
        return mod

    def test_skip_when_no_new_data_and_fp_match(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
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

    def test_force_always_rebuilds(self, monkeypatch, refresh_mod, tmp_path):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_read_fingerprint_meta", lambda: None)
        assert mod._skip_if_unchanged(True) is False

    def test_fp_mismatch_rebuilds(self, monkeypatch, refresh_mod, tmp_path):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "OLD", "latest_date": "2026-08-07"},
        )
        assert mod._skip_if_unchanged(False) is False

    def test_new_data_rebuilds(self, monkeypatch, refresh_mod, tmp_path):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
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

    def test_missing_checkpoint_rebuilds(self, monkeypatch, refresh_mod, tmp_path):
        mod = refresh_mod
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(tmp_path / "no_main.parquet"))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(tmp_path / "no_dual.parquet"))
        monkeypatch.setattr(mod, "compute_fingerprint", lambda: "fp1")
        monkeypatch.setattr(
            mod,
            "_read_fingerprint_meta",
            lambda: {"fingerprint": "fp1", "latest_date": "2026-08-07"},
        )
        assert mod._skip_if_unchanged(False) is False

    def test_main_skips_without_stale_or_rebuild(
        self, monkeypatch, refresh_mod, tmp_path
    ):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
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

    def test_fingerprint_covers_lhb_spec_not_whole_settings(
        self, monkeypatch, refresh_mod
    ):
        # fe.build 从 config/settings.py 只读 LHB_V2_SPEC → 该键必须参与指纹
        # (LHB 特征参数改了却静默跳过重建 = 质量风险). settings.py 整文件不再入指纹
        # (2026-08-25): serving 闸 (LEGACY_ENTRY_GATE 等) 不进检查点, 只改闸不应触发
        # ~1-1.5h 全量重建.
        mod = refresh_mod
        assert "config/settings.py" not in mod._FINGERPRINT_FILES
        base = mod.compute_fingerprint()
        monkeypatch.setattr(mod, "LHB_V2_SPEC", {"h_inst": 999})
        assert mod.compute_fingerprint() != base

    def test_fingerprint_covers_panel_path(self, monkeypatch, refresh_mod):
        # load_panel_v3/_reclassify 从 settings 只读 PANEL_V3_PATH → 面板源路径变化
        # (如 env 换 PANEL_PATH) 必须触发重建.
        mod = refresh_mod
        base = mod.compute_fingerprint()
        monkeypatch.setattr(mod, "PANEL_V3_PATH", "D:/nowhere/other_panel.parquet")
        assert mod.compute_fingerprint() != base

    def test_rebuild_writes_fingerprint_meta(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        mod = self._mod(refresh_mod, monkeypatch, tmp_path)
        _redirect_fresh_artifacts(mod, monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod, "load_panel_v3", lambda **kw: _tiny_board("main"))
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


class TestRefreshConcurrencyLock:
    """并发双建是 08-24 OOM 事故根因 (自动化+手动 refresh 撞车 → 16GB 双份构建).
    锁存在且持锁进程存活 → 第二实例必须零副作用退出; 跑完/跳过必须释放锁."""

    def _mock_rebuild(self, monkeypatch, mod, tmp_path=None):
        if tmp_path is not None:
            _redirect_fresh_artifacts(mod, monkeypatch, tmp_path)
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod, "load_panel_v3", lambda **kw: _tiny_board("main"))
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
        monkeypatch.setattr(mod, "_write_fingerprint_meta", lambda latest: None)

    def test_second_instance_blocked_zero_side_effects(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        mod = refresh_mod
        lock = tmp_path / "_refresh_parallel.lock"
        lock.write_text(str(os.getpid()), encoding="utf-8")  # 本测试进程 = 存活 PID
        touched = []
        monkeypatch.setattr(
            mod, "_skip_if_unchanged", lambda force: touched.append("skip") or False
        )
        assert mod.main() == 0
        assert touched == []  # 未进入任何重建路径
        assert lock.exists()  # 别人的锁不得动
        assert "already running" in capsys.readouterr().out

    def test_stale_lock_reclaimed_and_released(
        self, monkeypatch, refresh_mod, tmp_path
    ):
        mod = refresh_mod
        lock = tmp_path / "_refresh_parallel.lock"
        lock.write_text("999999999", encoding="utf-8")  # 死 PID
        monkeypatch.setattr(mod.psutil, "pid_exists", lambda pid: False)
        self._mock_rebuild(monkeypatch, mod, tmp_path)
        assert mod.main() == 0
        assert not lock.exists()  # 陈旧锁被回收, 跑完正常释放

    def test_lock_released_after_normal_run(self, monkeypatch, refresh_mod, tmp_path):
        mod = refresh_mod
        self._mock_rebuild(monkeypatch, mod, tmp_path)
        assert mod.main() == 0
        assert not (tmp_path / "_refresh_parallel.lock").exists()  # try/finally 释放

    def test_lock_released_on_skip_path(self, monkeypatch, refresh_mod, tmp_path):
        mod = refresh_mod
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: True)
        assert mod.main() == 0
        assert not (tmp_path / "_refresh_parallel.lock").exists()  # 跳过路径也释放


class TestCheckpointFreshnessAssert:
    """检查点新鲜度后置断言 (2026-09-02): refresh 是 parallel/deliver_parallel 的
    前置, 检查点落后 = 下游短名单/回测缺最新交易日 (全脏) → 重建"成功"也必须
    大声 exit 1, 且不静默放行."""

    def _mock_rebuild_only(self, monkeypatch, mod):
        """只 mock 重组件, 不 redirect 断言工件 — 断言读的就是测试预置的检查点."""
        monkeypatch.setattr(mod, "_skip_if_unchanged", lambda force: False)
        monkeypatch.setattr(mod.os, "rename", lambda a, b: None)
        monkeypatch.setattr(mod, "load_panel_v3", lambda **kw: _tiny_board("main"))
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
        monkeypatch.setattr(mod, "_write_fingerprint_meta", lambda latest: None)

    def _panel(self, monkeypatch, mod, tmp_path, date: str) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq

        p = str(tmp_path / "panel.parquet")
        df = pd.DataFrame({"date": [pd.Timestamp(date)]})
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
        monkeypatch.setattr(mod, "PANEL_V3_PATH", p)
        return p

    def _ckpt(self, tmp_path, date: str) -> str:
        import pyarrow as pa
        import pyarrow.parquet as pq

        p = str(tmp_path / "ck.parquet")
        df = pd.DataFrame({"date": [pd.Timestamp(date)]})
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p)
        return p

    def test_stale_checkpoint_fails_loud(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        """重建出的检查点 (08-07) 落后面板 (08-08) → rc=1, 大声报错."""
        mod = refresh_mod
        self._mock_rebuild_only(monkeypatch, mod)
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", self._ckpt(tmp_path, "2026-08-07"))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", self._ckpt(tmp_path, "2026-08-07"))
        self._panel(monkeypatch, mod, tmp_path, "2026-08-08")
        assert mod.main() == 1
        assert "ASSERT-FAIL" in capsys.readouterr().out

    def test_missing_checkpoint_fails_loud(self, monkeypatch, refresh_mod, tmp_path):
        """检查点文件缺失 (板块重建为空却静默交差) → rc=1, 不静默放行."""
        mod = refresh_mod
        self._mock_rebuild_only(monkeypatch, mod)
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", str(tmp_path / "no_main.parquet"))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", str(tmp_path / "no_dual.parquet"))
        self._panel(monkeypatch, mod, tmp_path, "2026-08-07")
        assert mod.main() == 1

    def test_fresh_checkpoints_pass_assert(
        self, monkeypatch, refresh_mod, tmp_path, capsys
    ):
        """两检查点与面板同日 → rc=0, 打印 [assert] ok (happy path 锁定)."""
        mod = refresh_mod
        self._mock_rebuild_only(monkeypatch, mod)
        monkeypatch.setattr(mod, "MAIN_CHECKPOINT", self._ckpt(tmp_path, "2026-08-07"))
        monkeypatch.setattr(mod, "DUAL_CHECKPOINT", self._ckpt(tmp_path, "2026-08-07"))
        self._panel(monkeypatch, mod, tmp_path, "2026-08-07")
        assert mod.main() == 0
        assert "[assert]" in capsys.readouterr().out
