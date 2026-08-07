"""prediction_db 多视界概率持久化 (v1.4: prob_up_2d/3d/5d)."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from app.pipeline1.prediction_db import PredictionDB


def _stocks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002"],
            "board": ["main", "main"],
            "pred_ret_1d": [0.01, 0.02],
            "pred_ret_2d": [0.02, 0.03],
            "pred_ret_3d": [0.03, 0.04],
            "pred_ret_5d": [0.05, 0.06],
            "prob_up": [0.6, 0.7],
            "prob_up_2d": [0.55, 0.65],
            "prob_up_3d": [0.58, 0.68],
            "prob_up_5d": [0.62, 0.72],
            "score": [0.1, 0.2],
            "momentum": ["up", "down"],
            "weight": [0.1, 0.1],
        }
    )


class TestPredictionDB:
    def test_insert_run_persists_pred_ret_2d(self, tmp_path):
        db = PredictionDB(path=str(tmp_path / "p.db"))
        assert db.insert_run("2026-08-02", _stocks()) == 2
        run = db.get_run("2026-08-02")
        by_sym = {s["symbol"]: s for s in run["stocks"]}
        assert by_sym["600001"]["pred_ret_2d"] == pytest.approx(0.02)
        assert by_sym["600002"]["pred_ret_2d"] == pytest.approx(0.03)

    def test_insert_run_persists_multihorizon_probs(self, tmp_path):
        db = PredictionDB(path=str(tmp_path / "p.db"))
        assert db.insert_run("2026-08-03", _stocks()) == 2
        run = db.get_run("2026-08-03")
        by_sym = {s["symbol"]: s for s in run["stocks"]}
        assert by_sym["600001"]["prob_up_2d"] == pytest.approx(0.55)
        assert by_sym["600001"]["prob_up_3d"] == pytest.approx(0.58)
        assert by_sym["600001"]["prob_up_5d"] == pytest.approx(0.62)

    def test_migrates_legacy_db_adding_multihorizon_columns(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE prediction_stocks (
                date TEXT, symbol TEXT, pred_ret_1d REAL, UNIQUE(date, symbol))"""
        )
        conn.execute(
            """CREATE TABLE prediction_outcomes (
                date TEXT, symbol TEXT, actual_ret_1d REAL)"""
        )
        conn.commit()
        conn.close()

        PredictionDB(path=path)  # 触发 SCHEMA + _migrate
        conn = sqlite3.connect(path)
        stock_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(prediction_stocks)")
        }
        out_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(prediction_outcomes)")
        }
        conn.close()
        assert "pred_ret_2d" in stock_cols
        assert "actual_ret_2d" in out_cols
        for col in ("prob_up_2d", "prob_up_3d", "prob_up_5d"):
            assert col in stock_cols

    def test_backfill_outcomes_persists_actual_ret_2d(self, tmp_path):
        db = PredictionDB(path=str(tmp_path / "p.db"))
        db.insert_run("2026-08-02", _stocks())
        outcomes = pd.DataFrame(
            {
                "symbol": ["600001", "600002"],
                "actual_ret_1d": [0.01, 0.02],
                "actual_ret_2d": [0.02, 0.03],
                "actual_ret_3d": [0.03, 0.04],
                "actual_ret_5d": [0.05, 0.06],
            }
        )
        assert db.backfill_outcomes("2026-08-02", outcomes) == 2
        run = db.get_run("2026-08-02")
        by_sym = {s["symbol"]: s for s in run["stocks"]}
        assert by_sym["600002"]["actual_ret_2d"] == pytest.approx(0.03)
