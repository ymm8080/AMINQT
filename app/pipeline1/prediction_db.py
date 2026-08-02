# -*- coding: utf-8 -*-
"""
预测池数据库 (SQLite) — 每日清单入库 + 实际收益回填
====================================================
三张表:
  prediction_runs   — 每日运行记录 (日期/股数/版本)
  prediction_stocks — 每只选股详情 (预测值/排序分/权重)
  prediction_outcomes — T+1/3/5 实际收益 + 方向正确性 (收盘后回填)

用法:
  from app.pipeline1.prediction_db import PredictionDB
  db = PredictionDB()
  db.insert_run(date, stocks_df)
  db.backfill_outcomes(date, actuals_df)
  db.get_run(date) → {meta, stocks, outcomes}
"""

from __future__ import annotations

import logging
import os
import sqlite3

import numpy as np
import pandas as pd

from config.settings import data_others_path

logger = logging.getLogger(__name__)

DB_PATH = str(data_others_path("data/predictions.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    date         TEXT PRIMARY KEY,
    n_stocks     INTEGER NOT NULL DEFAULT 0,
    schema_version TEXT NOT NULL DEFAULT '1.2',
    created_at   TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);

CREATE TABLE IF NOT EXISTS prediction_stocks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date         TEXT NOT NULL REFERENCES prediction_runs(date),
    symbol       TEXT NOT NULL,
    board        TEXT NOT NULL DEFAULT 'main',
    rank         INTEGER NOT NULL DEFAULT 0,
    pred_ret_1d  REAL,
    pred_ret_2d  REAL,
    pred_ret_3d  REAL,
    pred_ret_5d  REAL,
    prob_up      REAL,
    score        REAL,
    momentum     TEXT,
    weight       REAL,
    pain_prob    REAL,
    consensus_score REAL,
    signal_conflict INTEGER DEFAULT 0,
    UNIQUE(date, symbol)
);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    date         TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    actual_ret_1d REAL,
    actual_ret_2d REAL,
    actual_ret_3d REAL,
    actual_ret_5d REAL,
    direction_correct_1d INTEGER,  -- 1=true, 0=false, NULL=未计算
    pred_error_1d REAL,           -- pred - actual
    computed_at  TEXT NOT NULL DEFAULT (datetime('now','localtime')),
    PRIMARY KEY (date, symbol),
    FOREIGN KEY (date, symbol) REFERENCES prediction_stocks(date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_stocks_date ON prediction_stocks(date);
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON prediction_stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_outcomes_date ON prediction_outcomes(date);
"""


class PredictionDB:
    """预测池 SQLite 数据库."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """为历史 DB 补 T+2 列 (CREATE IF NOT EXISTS 不修改既有表)."""
        with sqlite3.connect(self.path) as conn:
            for table, col, ddl in (
                (
                    "prediction_stocks",
                    "pred_ret_2d",
                    "ALTER TABLE prediction_stocks ADD COLUMN pred_ret_2d REAL",
                ),
                (
                    "prediction_outcomes",
                    "actual_ret_2d",
                    "ALTER TABLE prediction_outcomes ADD COLUMN actual_ret_2d REAL",
                ),
            ):
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
                if col not in cols:
                    conn.execute(ddl)
            conn.commit()

    # ── 写入 ──
    def insert_run(
        self, date_str: str, stocks: pd.DataFrame, schema_version: str = "1.2"
    ) -> int:
        """插入当日清单 (幂等: 已存在则跳过)."""
        with sqlite3.connect(self.path) as conn:
            # 幂等检查
            cur = conn.execute(
                "SELECT 1 FROM prediction_runs WHERE date=?", (date_str,)
            )
            if cur.fetchone():
                logger.info("清单 %s 已入库, 跳过", date_str)
                return 0

            conn.execute(
                "INSERT INTO prediction_runs (date, n_stocks, schema_version) VALUES (?,?,?)",
                (date_str, len(stocks), schema_version),
            )
            for rank, (_, row) in enumerate(stocks.iterrows(), 1):
                conn.execute(
                    """INSERT OR IGNORE INTO prediction_stocks
                       (date, symbol, board, rank, pred_ret_1d, pred_ret_2d, pred_ret_3d, pred_ret_5d,
                        prob_up, score, momentum, weight, pain_prob, consensus_score, signal_conflict)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        date_str,
                        str(row.get("symbol", "")),
                        str(row.get("board", "main")),
                        rank,
                        _safe_float(row, "pred_ret_1d"),
                        _safe_float(row, "pred_ret_2d"),
                        _safe_float(row, "pred_ret_3d"),
                        _safe_float(row, "pred_ret_5d"),
                        _safe_float(row, "prob_up"),
                        _safe_float(row, "score"),
                        str(row.get("momentum", "")),
                        _safe_float(row, "weight"),
                        _safe_float(row, "pain_prob"),
                        _safe_float(row, "consensus_score"),
                        int(row.get("signal_conflict", 0)),
                    ),
                )
            conn.commit()
        logger.info("清单 %s: %d 只入库", date_str, len(stocks))
        return len(stocks)

    def backfill_outcomes(self, date_str: str, outcomes: pd.DataFrame) -> int:
        """回填 T+1/3/5 实际收益 (收盘后调用).

        outcomes 需含: symbol, actual_ret_1d, actual_ret_3d, actual_ret_5d
        自动计算 direction_correct_1d 和 pred_error_1d.
        """
        count = 0
        with sqlite3.connect(self.path) as conn:
            preds = conn.execute(
                "SELECT symbol, pred_ret_1d FROM prediction_stocks WHERE date=?",
                (date_str,),
            ).fetchall()
            pred_map = {s: p for s, p in preds}

            for _, row in outcomes.iterrows():
                sym = str(row["symbol"])
                pred_1d = pred_map.get(sym)
                actual_1d = _safe_float(row, "actual_ret_1d")
                dir_correct = None
                pred_err = None
                if pred_1d is not None and actual_1d is not None:
                    dir_correct = 1 if (pred_1d > 0) == (actual_1d > 0) else 0
                    pred_err = round(pred_1d - actual_1d, 6)

                conn.execute(
                    """INSERT OR REPLACE INTO prediction_outcomes
                       (date, symbol, actual_ret_1d, actual_ret_2d, actual_ret_3d, actual_ret_5d,
                        direction_correct_1d, pred_error_1d)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        date_str,
                        sym,
                        _safe_float(row, "actual_ret_1d"),
                        _safe_float(row, "actual_ret_2d"),
                        _safe_float(row, "actual_ret_3d"),
                        _safe_float(row, "actual_ret_5d"),
                        dir_correct,
                        pred_err,
                    ),
                )
                count += 1
            conn.commit()
        logger.info("回填 %s: %d 条实际收益", date_str, count)
        return count

    # ── 查询 ──
    def get_run(self, date_str: str) -> dict | None:
        """获取单日完整记录 (meta + stocks + outcomes)."""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            meta = conn.execute(
                "SELECT * FROM prediction_runs WHERE date=?", (date_str,)
            ).fetchone()
            if not meta:
                return None
            stocks = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM prediction_stocks WHERE date=? ORDER BY rank",
                    (date_str,),
                ).fetchall()
            ]
            outcomes = {
                r["symbol"]: dict(r)
                for r in conn.execute(
                    "SELECT * FROM prediction_outcomes WHERE date=?",
                    (date_str,),
                ).fetchall()
            }
            # Merge outcomes into stocks
            for s in stocks:
                sym = s["symbol"]
                if sym in outcomes:
                    s.update(outcomes[sym])
            return {
                "date": date_str,
                "meta": dict(meta),
                "stocks": stocks,
            }

    def list_runs(self, limit: int = 60) -> list[dict]:
        """列出所有运行日期 (最近 N 条)."""
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT date, n_stocks, schema_version, created_at "
                "FROM prediction_runs ORDER BY date DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def quality_summary(self, date_str: str) -> dict:
        """单日预测质量汇总."""
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """SELECT
                     COUNT(*) as n,
                     AVG(CASE WHEN direction_correct_1d=1 THEN 1.0 ELSE 0.0 END) as dir_acc,
                     AVG(pred_error_1d) as bias_1d,
                     AVG(ABS(pred_error_1d)) as mae_1d
                   FROM prediction_outcomes WHERE date=?""",
                (date_str,),
            ).fetchone()
            if not row or row[0] == 0:
                return {"n": 0}
            return {
                "n": int(row[0]),
                "direction_accuracy": round(float(row[1] or 0), 4),
                "bias_1d": round(float(row[2] or 0), 6),
                "mae_1d": round(float(row[3] or 0), 6),
            }

    def all_quality(self, limit: int = 60) -> list[dict]:
        """所有日期质量汇总 (用于趋势图)."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """SELECT
                     o.date,
                     COUNT(*) as n,
                     ROUND(AVG(CASE WHEN direction_correct_1d=1 THEN 1.0 ELSE 0.0 END), 4) as dir_acc,
                     ROUND(AVG(pred_error_1d), 6) as bias_1d,
                     ROUND(AVG(ABS(pred_error_1d)), 6) as mae_1d
                   FROM prediction_outcomes o
                   GROUP BY o.date
                   ORDER BY o.date DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [
                {
                    "date": r[0],
                    "n": r[1],
                    "direction_accuracy": r[2],
                    "bias_1d": r[3],
                    "mae_1d": r[4],
                }
                for r in rows
            ]


def _safe_float(row, col: str) -> float | None:
    val = row.get(col) if hasattr(row, "get") else row[col]
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return round(float(val), 6)
    except (ValueError, TypeError):
        return None
