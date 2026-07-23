# -*- coding: utf-8 -*-
"""Frontier 前端数据 API 测试 (FastAPI TestClient)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestFrontierAPI:
    def test_latest_list(self):
        r = client.get("/api/frontier/list/latest")
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == "1.0"
        assert len(body["items"]) > 0
        assert {"symbol", "score", "prob_up", "pred_ret_1d"} <= set(body["items"][0])

    def test_list_dates_and_404(self):
        r = client.get("/api/frontier/list/20990101")
        assert r.status_code == 404

    def test_ohlc_and_intraday(self):
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
