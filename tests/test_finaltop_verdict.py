"""finaltop 第二票判词 (2026-09-02 终榜口径) — 纯判词 + retrain 工具闸接线.

判据: 全窗 ≥ 0 且双半 ≥ tol_half 且配对胜率 ≥ win_rate_min; 无判词 fail-safe
保留旧包. 锚定数字 = 2026-09-02 真实回放 WORM
(_dual_pkg_finaltop_compare_20260902_201413.json): main +0.55pp/日 PASS /
dual -0.44pp/日 FAIL.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
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
DUAL_REAL = {
    "days": 48,
    "d3_full": -0.00442,
    "d3_h1": -0.00337,
    "d3_h2": -0.00548,
    "win_days": 22,
    "lose_days": 26,
}


def _payload(board: str, delta: dict) -> dict:
    return {board: {"delta_B_vs_A": delta}}


class TestVerdictFromPayload:
    def test_real_main_20260902_passes(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        v = verdict_from_payload(_payload("main", MAIN_REAL), "main")
        assert v["ok"] is True
        assert v["pass"] is True
        assert all(v["checks"].values())
        assert v["win_rate"] == pytest.approx(30 / 48)

    def test_real_dual_20260902_fails(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        v = verdict_from_payload(_payload("dual", DUAL_REAL), "dual")
        assert v["ok"] is True
        assert v["pass"] is False
        assert v["checks"]["full"] is False

    def test_half_below_tolerance_fails(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        d = dict(MAIN_REAL, d3_h2=-0.006)
        v = verdict_from_payload(_payload("main", d), "main")
        assert v["pass"] is False
        assert v["checks"]["half2"] is False

    def test_win_rate_below_min_fails_even_if_full_positive(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        d = dict(MAIN_REAL, win_days=21, lose_days=27)
        v = verdict_from_payload(_payload("main", d), "main")
        assert v["pass"] is False
        assert v["checks"]["win_rate"] is False

    def test_boundary_ties_pass(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        d = dict(
            MAIN_REAL,
            d3_full=0.0,
            d3_h1=-0.005,
            d3_h2=-0.005,
            win_days=24,
            lose_days=24,
        )
        v = verdict_from_payload(_payload("main", d), "main")
        assert v["pass"] is True

    def test_insufficient_days_no_verdict(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        d = dict(MAIN_REAL, days=9, win_days=9, lose_days=0)
        v = verdict_from_payload(_payload("main", d), "main")
        assert v["ok"] is False
        assert v["reason"] == "insufficient_days"

    def test_missing_delta_no_verdict(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        assert verdict_from_payload({}, "main")["ok"] is False
        assert verdict_from_payload({"main": {}}, "main")["ok"] is False

    def test_arm_param_reads_other_arm_delta(self):
        from app.pipeline1.finaltop_verdict import verdict_from_payload

        payload = {"main": {"delta_B_vs_A": MAIN_REAL, "delta_C_vs_A": DUAL_REAL}}
        assert verdict_from_payload(payload, "main")["pass"] is True
        assert verdict_from_payload(payload, "main", arm="C")["pass"] is False


def _gate_cfg() -> dict:
    return {
        "enable": True,
        "caliber": "final_list_tool",
        "tol_half": -0.005,
        "win_rate_min": 0.5,
        "eval_days": 48,
        "min_days": 10,
    }


class TestRecentDrawBundles:
    """稳定性抽签臂发现: 文件名日期窗内近 N 个同板构建包, 新→旧."""

    def _mk(self, models, name, mtime_age_days, now):
        p = models / name
        p.write_bytes(b"x")
        t = now - mtime_age_days * 86400
        os.utime(p, (t, t))

    def test_window_and_limit(self, tmp_path, monkeypatch):
        import scripts._retrain_legacy_full as m

        monkeypatch.setattr(
            m, "time", types.SimpleNamespace(strftime=lambda *_a: "20260902")
        )
        models = tmp_path / "models"
        monkeypatch.setattr(m, "MODEL_DIR", str(models))
        models.mkdir()
        now = time.time()
        self._mk(models, "main_20260901.pkl", 1, now)  # 窗内 新
        self._mk(models, "main_20260831.pkl", 2, now)  # 窗内
        self._mk(models, "main_20260830excessfix.pkl", 3, now)  # 窗内 (限 2 截掉)
        self._mk(models, "main_20260825.pkl", 8, now)  # 窗外
        self._mk(models, "main_current.pkl", 0, now)  # 无日期段
        self._mk(models, "main_current_retrain_backup.pkl", 0, now)  # 无日期段
        self._mk(models, "dual_20260901.pkl", 1, now)  # 别的板
        cfg = {"stability_max_extra": 2, "stability_window_days": 3}
        hits = m._recent_draw_bundles("main", exclude=set(), cfg=cfg)
        assert [os.path.basename(p) for p in hits] == [
            "main_20260901.pkl",
            "main_20260831.pkl",
        ]

    def test_exclude_and_short_tag(self, tmp_path, monkeypatch):
        import scripts._retrain_legacy_full as m

        monkeypatch.setattr(
            m, "time", types.SimpleNamespace(strftime=lambda *_a: "20260902")
        )
        models = tmp_path / "models"
        monkeypatch.setattr(m, "MODEL_DIR", str(models))
        models.mkdir()
        now = time.time()
        self._mk(models, "main_20260901.pkl", 1, now)
        self._mk(models, "main_20260831.pkl", 2, now)
        cfg = {"stability_max_extra": 2, "stability_window_days": 3}
        hits = m._recent_draw_bundles(
            "main", exclude={str(models / "main_20260901.pkl")}, cfg=cfg
        )
        assert [os.path.basename(p) for p in hits] == ["main_20260831.pkl"]


class TestRetrainFinaltopGate:
    """_finaltop_gate 接线: 工具产出→判词裁决; 失败/超时/无产出→保留旧包."""

    @pytest.fixture
    def m(self, tmp_path, monkeypatch):
        import scripts._retrain_legacy_full as m

        monkeypatch.setattr(st, "DATA_OTHERS_DIR", tmp_path / "DATA OTHERS")
        monkeypatch.setattr(m, "MODEL_DIR", str(tmp_path / "models"))
        monkeypatch.setattr(
            m, "time", types.SimpleNamespace(strftime=lambda *_a: "20260902")
        )
        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "main_current.pkl").write_bytes(b"x")
        (tmp_path / "new.pkl").write_bytes(b"x")
        return m

    def _fake_run(
        self, monkeypatch, m, rc=0, delta=None, board="main", raise_exc=None,
        extra_arms=None,
    ):
        calls: dict = {}

        def fake_run(cmd, **kw):
            calls["cmd"] = cmd
            if raise_exc is not None:
                raise raise_exc
            if rc == 0 and delta is not None:
                diag = Path(st.DATA_OTHERS_DIR, "diag")
                diag.mkdir(parents=True, exist_ok=True)
                body = {"delta_B_vs_A": delta}
                for arm, d in (extra_arms or {}).items():
                    body[f"delta_{arm}_vs_A"] = d
                (
                    diag / "_dual_pkg_finaltop_compare_20990101_000000.json"
                ).write_text(json.dumps({board: body}), encoding="utf-8")
            return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

        monkeypatch.setattr(
            m,
            "subprocess",
            types.SimpleNamespace(run=fake_run, TimeoutExpired=subprocess.TimeoutExpired),
        )
        return calls

    def test_pass_verdict_promotes(self, m, tmp_path, monkeypatch):
        calls = self._fake_run(monkeypatch, m, delta=MAIN_REAL)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is True
        assert "--guard-exclude-pid" in calls["cmd"]
        assert calls["cmd"][calls["cmd"].index("--eval-days") + 1] == "48"

    def test_fail_verdict_keeps_old(self, m, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, m, delta=DUAL_REAL)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is False

    def test_tool_rc3_keeps_old(self, m, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, m, rc=3)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is False

    def test_tool_no_output_keeps_old(self, m, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, m, rc=0, delta=None)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is False

    def test_tool_timeout_keeps_old(self, m, tmp_path, monkeypatch):
        self._fake_run(
            monkeypatch,
            m,
            raise_exc=subprocess.TimeoutExpired(cmd="x", timeout=5400),
        )
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is False

    def test_no_current_ic_alone(self, m, tmp_path, monkeypatch):
        (tmp_path / "models" / "main_current.pkl").unlink()
        calls = self._fake_run(monkeypatch, m, delta=MAIN_REAL)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), _gate_cfg()) is True
        assert calls == {}  # 未起工具

    # ── 防坏签稳定性 (stability_enable) ────────────────────────────

    def _stability_cfg(self):
        return dict(
            _gate_cfg(), stability_enable=True, stability_max_extra=2,
            stability_window_days=3,
        )

    def test_stability_blocked_lucky_single_draw(self, m, tmp_path, monkeypatch):
        # 新包自身 PASS, 但近窗另一抽 FAIL → 严格多数 (1/2) 不过 → 拦
        (tmp_path / "models" / "main_20260901.pkl").write_bytes(b"x")
        calls = self._fake_run(
            monkeypatch, m, delta=MAIN_REAL, extra_arms={"C": DUAL_REAL}
        )
        cfg = self._stability_cfg()
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), cfg) is False
        joined = " ".join(calls["cmd"])
        assert "a,b,c" in joined  # 三臂对拍
        assert str(tmp_path / "models" / "main_20260901.pkl") in joined

    def test_stability_pass_when_extra_also_passes(self, m, tmp_path, monkeypatch):
        (tmp_path / "models" / "main_20260901.pkl").write_bytes(b"x")
        self._fake_run(
            monkeypatch, m, delta=MAIN_REAL, extra_arms={"C": MAIN_REAL}
        )
        assert (
            m._finaltop_gate("main", str(tmp_path / "new.pkl"), self._stability_cfg())
            is True
        )

    def test_stability_no_recent_draws_today_alone_decides(self, m, tmp_path, monkeypatch):
        self._fake_run(monkeypatch, m, delta=MAIN_REAL)
        assert (
            m._finaltop_gate("main", str(tmp_path / "new.pkl"), self._stability_cfg())
            is True
        )

    def test_stability_unusable_extra_not_counted(self, m, tmp_path, monkeypatch):
        # 额外臂无判词 (可比日不足) → 不计入多数, 新包自身判词决定
        (tmp_path / "models" / "main_20260901.pkl").write_bytes(b"x")
        bad_days = dict(MAIN_REAL, days=5)
        self._fake_run(
            monkeypatch, m, delta=MAIN_REAL, extra_arms={"C": bad_days}
        )
        assert (
            m._finaltop_gate("main", str(tmp_path / "new.pkl"), self._stability_cfg())
            is True
        )

    def test_stability_off_ignores_extras(self, m, tmp_path, monkeypatch):
        # stability_enable=False → 单抽判词 (旧语义), 邻近坏抽不拦
        (tmp_path / "models" / "main_20260901.pkl").write_bytes(b"x")
        calls = self._fake_run(
            monkeypatch, m, delta=MAIN_REAL, extra_arms={"C": DUAL_REAL}
        )
        cfg = dict(_gate_cfg(), stability_enable=False)
        assert m._finaltop_gate("main", str(tmp_path / "new.pkl"), cfg) is True
        assert "a,b" in " ".join(calls["cmd"]) and "a,b,c" not in " ".join(calls["cmd"])
