"""Frontier 前端数据 API 测试 (FastAPI TestClient)."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
except ImportError:
    app = None  # type: ignore[assignment]
    client = None  # type: ignore[assignment]

pytestmark = pytest.mark.skipif(client is None, reason="httpx not installed")


class TestFrontierAPI:
    def test_latest_list(self):
        r = client.get("/api/frontier/list/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == "1.4"
        assert len(body["items"]) > 0
        assert {"symbol", "score", "prob_up", "pred_ret_1d"} <= set(body["items"][0])

    def test_list_dates_and_404(self):
        r = client.get("/api/frontier/list/20990101")
        assert r.status_code == 404

    def test_ohlc_and_intraday(self, monkeypatch):
        from app.streamlit import data_service as ds

        monkeypatch.setattr(ds, "fetch_real_intraday", lambda s: None)
        r = client.get("/api/frontier/ohlc/600519?days=60")
        assert r.status_code == 200 and len(r.json()["items"]) == 60
        r = client.get("/api/frontier/intraday/600519")
        assert r.status_code == 200 and len(r.json()["items"]) == 120

    def test_watchlist_toggle(self):
        r = client.post(
            "/api/frontier/watchlist/toggle",
            json={"symbol": "600519", "name": "贵州茅台"},
        )
        assert r.status_code == 200
        first = r.json()["watched"]
        r = client.post("/api/frontier/watchlist/toggle", json={"symbol": "600519"})
        assert r.json()["watched"] is not first

    def test_backtest_run(self):
        r = client.post(
            "/api/frontier/backtest/run", json={"max_hold_days": 3, "prob_exit": 0.5}
        )
        assert r.status_code == 200
        body = r.json()
        for key in ("total_return", "net_excess_annual", "max_drawdown", "sharpe"):
            assert key in body["metrics"]
        assert len(body["nav_curve"]) > 0

    def test_gate_eval(self, tmp_path, monkeypatch):
        """gate-eval 端点: 无报告 → exists=false; 有报告 → 返回最新内容."""
        import glob
        import json

        r = client.get("/api/frontier/backtest/gate-eval")
        assert r.status_code == 200 and "exists" in r.json()

        report = tmp_path / "gate_eval_20990101.json"
        report.write_text(
            json.dumps({"generated_at": "t", "window": ["a", "b"], "scenarios": []}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            glob, "glob", lambda p: [str(report)] if "gate_eval_" in p else []
        )
        body = client.get("/api/frontier/backtest/gate-eval").json()
        assert body["exists"] is True and body["window"] == ["a", "b"]

    def test_tune_and_validation(self):
        r = client.post(
            "/api/frontier/backtest/tune",
            json={"params": ["max_hold_days", "prob_exit"], "top_k": 2},
        )
        assert r.status_code == 200
        assert "best_params" in r.json()
        r = client.post("/api/frontier/backtest/tune", json={"params": ["not_a_param"]})
        assert r.status_code == 400

    def test_rule_config(self):
        r = client.get("/api/frontier/config/rules")
        assert r.status_code == 200
        tunable = r.json()["tunable"]
        assert "max_hold_days" in tunable
        assert tunable["max_hold_days"]["bounds"] == [2, 5, 1]

    def test_tuning_report(self):
        r = client.get("/api/frontier/tuning/report")
        assert r.status_code == 200
        assert "exists" in r.json()
