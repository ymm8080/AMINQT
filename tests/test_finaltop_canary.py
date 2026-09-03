"""晋升后 canary (2026-09-02 防坏签#7b): 判定纯函数 + 回退端到端 (假工具)."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

import config.settings as st

MAIN_REAL = {
    "days": 48,
    "d3_full": 0.005533993402052515,
    "d3_h1": 0.007849548508338236,
    "d3_h2": 0.0032184382957667957,
    "win_days": 30,
    "lose_days": 18,
}
DECISIVE_BAD = dict(MAIN_REAL, d3_full=-0.006, d3_h1=-0.007, d3_h2=-0.0052,
                    win_days=20, lose_days=28)
NON_DECISIVE_FAIL = dict(MAIN_REAL, d3_full=-0.004, d3_h1=-0.006, d3_h2=-0.003,
                         win_days=23, lose_days=25)


class TestPureDecisions:
    def test_decisive_fail(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload
        from scripts._finaltop_canary import decisive_fail

        v = verdict_from_payload({"main": {"delta_B_vs_A": DECISIVE_BAD}}, "main")
        assert decisive_fail(v) is True  # -0.006 < 0 - 0.005

    def test_fail_within_band_not_decisive(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload
        from scripts._finaltop_canary import decisive_fail

        v = verdict_from_payload({"main": {"delta_B_vs_A": NON_DECISIVE_FAIL}}, "main")
        assert v["pass"] is False
        assert decisive_fail(v) is False  # -0.004 在非劣带附近

    def test_pass_or_no_verdict_not_decisive(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload
        from scripts._finaltop_canary import decisive_fail

        v = verdict_from_payload({"main": {"delta_B_vs_A": MAIN_REAL}}, "main")
        assert v["pass"] is True
        assert decisive_fail(v) is False
        assert decisive_fail({"ok": False, "reason": "timeout"}) is False

    def test_should_run(self):
        from scripts._finaltop_canary import should_run

        entry = {"status": "active", "days": 10, "ran": {}}
        assert should_run(entry, "2026-09-03") is True
        entry["ran"]["2026-09-03"] = {}
        assert should_run(entry, "2026-09-03") is False  # 今日已跑
        assert should_run(entry, "2026-09-04") is True
        entry["status"] = "done"
        assert should_run(entry, "2026-09-04") is False
        full = {"status": "active", "days": 2, "ran": {"a": 1, "b": 2}}
        assert should_run(full, "2026-09-04") is False  # 窗口满


class TestEndToEnd:
    @pytest.fixture
    def m(self, tmp_path, monkeypatch):
        import scripts._finaltop_canary as m

        monkeypatch.setattr(st, "DATA_OTHERS_DIR", tmp_path / "DATA OTHERS")
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(m, "MODEL_DIR", str(models))
        (models / "main_current.pkl").write_bytes(b"NEW")
        (models / "main_20260903.pkl").write_bytes(b"OLD")
        state_path = tmp_path / "DATA OTHERS" / "diag" / "finaltop_canary_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text(
            json.dumps(
                {
                    "main": {
                        "tag": "20260902",
                        "prev_tag": "20260903",
                        "backup": str(models / "main_20260903.pkl"),
                        "promoted": "2026-09-02",
                        "days": 10,
                        "ran": {},
                        "status": "active",
                    }
                }
            ),
            encoding="utf-8",
        )
        return m

    def _fake_tool(self, monkeypatch, m, delta):
        counter = iter(range(1000))

        def fake_run(cmd, **kw):
            assert "--guard-exclude-pid" in cmd
            diag = Path(st.DATA_OTHERS_DIR, "diag")
            diag.mkdir(parents=True, exist_ok=True)
            name = f"_dual_pkg_finaltop_compare_2099{next(counter):08d}_000000.json"
            (diag / name).write_text(
                json.dumps({"main": {"delta_B_vs_A": delta}}), encoding="utf-8"
            )
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(
            m,
            "subprocess",
            types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
        )

    def test_pass_maintains_and_records(self, m, monkeypatch):
        self._fake_tool(monkeypatch, m, MAIN_REAL)
        monkeypatch.setattr("sys.argv", ["x"])
        assert m.main() == 0
        state = m.load_state()
        assert state["main"]["status"] == "active"
        assert len(state["main"]["ran"]) == 1
        assert list(state["main"]["ran"].values())[0]["pass"] is True
        assert (Path(m.MODEL_DIR) / "main_current.pkl").read_bytes() == b"NEW"

    def test_decisive_fail_revert_only_with_flag(self, m, monkeypatch):
        self._fake_tool(monkeypatch, m, DECISIVE_BAD)
        monkeypatch.setattr("sys.argv", ["x"])
        assert m.main() == 0  # 缺省只留证
        state = m.load_state()
        assert state["main"]["status"] == "active"
        assert (Path(m.MODEL_DIR) / "main_current.pkl").read_bytes() == b"NEW"

        # 重置今日 ran (模拟次日) 再带 --revert 跑
        state["main"]["ran"] = {}
        state_path = Path(st.DATA_OTHERS_DIR, "diag", "finaltop_canary_state.json")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        monkeypatch.setattr("sys.argv", ["x", "--revert"])
        assert m.main() == 0
        state = m.load_state()
        assert state["main"]["status"] == "reverted"
        assert (Path(m.MODEL_DIR) / "main_current.pkl").read_bytes() == b"OLD"

    def test_non_decisive_fail_keeps_observing(self, m, monkeypatch):
        self._fake_tool(monkeypatch, m, NON_DECISIVE_FAIL)
        monkeypatch.setattr("sys.argv", ["x"])
        assert m.main() == 0
        state = m.load_state()
        assert state["main"]["status"] == "active"
        assert (Path(m.MODEL_DIR) / "main_current.pkl").read_bytes() == b"NEW"

    def test_window_full_marks_done(self, m, monkeypatch):
        self._fake_tool(monkeypatch, m, MAIN_REAL)
        monkeypatch.setattr("sys.argv", ["x"])
        state = m.load_state()
        state["main"]["days"] = 1
        Path(st.DATA_OTHERS_DIR, "diag", "finaltop_canary_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        assert m.main() == 0
        assert m.load_state()["main"]["status"] == "done"

    def test_no_state_exits_clean(self, tmp_path, monkeypatch):
        import scripts._finaltop_canary as m

        monkeypatch.setattr(st, "DATA_OTHERS_DIR", tmp_path / "DATA OTHERS")
        monkeypatch.setattr("sys.argv", ["x"])
        assert m.main() == 0

    def test_revert_updates_meta(self, m, tmp_path, monkeypatch):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps({"main": {"tag": "20260902", "file": "main_20260902.pkl"}}),
            encoding="utf-8",
        )
        monkeypatch.setitem(sys.modules, "app.pipeline1.model_meta", _fake_meta(meta_path))
        entry = {
            "backup": str(Path(m.MODEL_DIR) / "main_20260903.pkl"),
            "prev_tag": "20260903",
        }
        m.revert("main", entry)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["main"]["tag"] == "20260903"
        assert meta["main"]["file"] == "main_20260903.pkl"


def _fake_meta(meta_path):
    import app.pipeline1.model_meta as real

    ns = types.SimpleNamespace(
        load_modules=lambda: real.load_modules(str(meta_path)),
        save_modules=lambda mods: real.save_modules(mods, str(meta_path)),
    )
    return ns
