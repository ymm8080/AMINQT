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


# ── legacy 双轨影子 (2026-08-07) ──
def _shadow_candidates() -> pd.DataFrame:
    """三板块候选, pred_ret_3d 不排序 (验证 shadow_pool_frame 按幅度排序+每板块截断)."""
    return pd.DataFrame(
        {
            "symbol": [
                "600001",
                "600002",
                "600003",
                "300001",
                "300002",
                "300003",
                "688001",
            ],
            "board": ["main", "main", "main", "GEM", "GEM", "GEM", "STAR"],
            "pred_ret_3d": [0.01, 0.05, None, 0.02, 0.09, 0.04, 0.07],
            "prob_up_3d": [0.5, 0.6, 0.55, 0.45, 0.7, 0.5, 0.65],
            "prob_up": [0.52, 0.61, 0.56, 0.46, 0.71, 0.51, 0.66],
        }
    )


class TestShadowTrack:
    def test_shadow_pool_frame_per_board_top_n(self):
        from app.pipeline1.prediction_db import shadow_pool_frame

        df = shadow_pool_frame(_shadow_candidates(), n=2)
        got = {b: df[df["board"] == b]["symbol"].tolist() for b in df["board"].unique()}
        assert got == {
            "main": ["600002", "600001"],  # pred_ret_3d 0.05, 0.01
            "GEM": ["300002", "300003"],  # 0.09, 0.04
            "STAR": ["688001"],
        }
        assert df["pred_ret_3d"].notna().all()  # NaN 剔除

    def test_shadow_pool_frame_empty_and_missing_board(self):
        from app.pipeline1.prediction_db import shadow_pool_frame

        assert shadow_pool_frame(pd.DataFrame()).empty
        no_board = _shadow_candidates().drop(columns=["board"])
        df = shadow_pool_frame(no_board, n=2)
        assert (df["board"] == "main").all()  # 缺 board 列回退 main

    def test_insert_shadow_persists_rank_by_ret(self, tmp_path):
        db = PredictionDB(path=str(tmp_path / "p.db"))
        df = _shadow_candidates()
        df = df[df["pred_ret_3d"].notna()].sort_values("pred_ret_3d", ascending=False)
        assert db.insert_shadow("2026-08-07", df) == 6
        shadow = db.get_shadow("2026-08-07")
        by_sym = {s["symbol"]: s for s in shadow}
        assert by_sym["300002"]["rank_by_ret"] == 1  # 幅度最大 → 排名 1
        assert by_sym["600001"]["rank_by_ret"] == 6
        assert by_sym["300002"]["pred_ret_3d"] == pytest.approx(0.09)
        # 幂等: 重复插入覆盖不翻倍
        assert db.insert_shadow("2026-08-07", df.head(3)) == 3
        assert len(db.get_shadow("2026-08-07")) == 6

    def test_backfill_shadow_outcomes_and_quality(self, tmp_path):
        db = PredictionDB(path=str(tmp_path / "p.db"))
        df = _shadow_candidates()
        df = df[df["pred_ret_3d"].notna()].sort_values("pred_ret_3d", ascending=False)
        db.insert_shadow("2026-08-07", df)
        outcomes = pd.DataFrame(
            {"symbol": ["300002", "600002"], "actual_ret_3d": [0.05, -0.02]}
        )
        assert db.backfill_shadow_outcomes("2026-08-07", outcomes) == 2
        q = db.shadow_quality("2026-08-07")
        assert q["n"] == 6 and q["matured_3d"] == 2
        by_sym = {s["symbol"]: s for s in db.get_shadow("2026-08-07")}
        assert by_sym["300002"]["actual_ret_3d"] == pytest.approx(0.05)
