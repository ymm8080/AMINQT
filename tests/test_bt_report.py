"""测试 app.streamlit.bt_report — 回测 backtest.json 纯解析.

用内联合成 fixture 覆盖: 新 schema 全字段 / 旧 schema (无 conclusion) 防御 /
磁盘 IO (list_runs, load_run_json). 真实 BACKTEST_RESULT_DIR 存在时追加冒烟.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.streamlit import bt_report as btr


def _horizon(mag, wr, n, base_wr, ok=True):
    return {
        "mag": mag,
        "winrate": wr,
        "n": n,
        "ok": ok,
        "baseline": {"winrate": base_wr, "mag": mag - 0.01, "n": 10000},
    }


def _make_fixture() -> dict:
    """合成一份新 schema backtest.json (结构与真实产出同构)."""
    per_h = {
        h: _horizon(0.05 + 0.01 * i, 0.81 + 0.01 * i, 500, 0.80 + 0.01 * i)
        for i, h in enumerate(["3d", "5d", "10d"])
    }
    board = {
        "label": "主板",
        "latest": "2026-08-05 00:00:00",
        "stale": False,
        "rows": 1000,
        "stocks": 200,
        "criteria": {"min_winrate": 0.55, "min_mag": 0.03},
        "merged": {
            "top5": {"oos": {"6m": {"per_horizon": per_h, "kept": True}}},
            "top10": {"oos": {"6m": {"per_horizon": per_h, "kept": True}}},
        },
        "systems": {
            "sniper": {
                "oos": {
                    "6m": {"primary": {"per_horizon": per_h, "top_n": 5, "kept": True}}
                },
            },
            "fusion": {
                "oos": {
                    "6m": {"primary": {"per_horizon": per_h, "top_n": 10, "kept": True}}
                },
            },
        },
        "last_days": {
            "days": [
                {
                    "date": "2026-07-30",
                    "sniper_top5": {
                        "picks": [
                            {
                                "symbol": "600519",
                                "rk": 1,
                                "score": 0.9,
                                "mfe_3d": 0.02,
                                "mfe_5d": 0.03,
                                "mfe_10d": None,
                            }
                        ]
                    },
                    "fusion_top10": {
                        "picks": [
                            {
                                "symbol": "000001",
                                "rk": 1,
                                "score": 0.85,
                                "mfe_3d": 0.01,
                                "mfe_5d": 0.02,
                                "mfe_10d": 0.04,
                            }
                        ]
                    },
                }
            ]
        },
    }
    return {
        "ts": "20260805_120000",
        "objective": "OOS 双头判定",
        "window": {
            "full": {"start": "2023-08-07", "end": "2026-08-05", "trading_days": 726},
            "oos": {"6m": {"trading_days": 126}},
        },
        "rows": 1000,
        "stocks": 200,
        "boards": {
            "main": board,
            "dual": board,
        },
        "conclusion": {
            "oos_label": "6m",
            "boards": {
                "main": {
                    "label": "主板",
                    "latest": "2026-08-05 00:00:00",
                    "stale": False,
                    "cuts": {
                        "top5": {
                            "kept": True,
                            "best_horizon": "10d",
                            "winrate": 0.91,
                            "mag": 0.09,
                            "delta_wr": -0.01,
                            "baseline_wr": 0.92,
                            "n": 500,
                        },
                        "top10": {
                            "kept": False,
                            "best_horizon": "5d",
                            "winrate": 0.88,
                            "mag": 0.07,
                            "delta_wr": -0.02,
                            "baseline_wr": 0.90,
                            "n": 900,
                        },
                    },
                    "systems": {"sniper": True, "fusion": True, "slow_bull": False},
                    "improvements": ["末尾 T+10d 多数入选不可测", "建议收紧 TOP-5"],
                },
                "dual": {
                    "label": "创业板+科创板",
                    "latest": "2026-08-05 00:00:00",
                    "stale": False,
                    "cuts": {
                        "top5": {
                            "kept": True,
                            "best_horizon": "10d",
                            "winrate": 0.94,
                            "mag": 0.12,
                            "delta_wr": 0.004,
                            "baseline_wr": 0.93,
                            "n": 510,
                        },
                        "top10": {
                            "kept": True,
                            "best_horizon": "10d",
                            "winrate": 0.95,
                            "mag": 0.11,
                            "delta_wr": 0.007,
                            "baseline_wr": 0.94,
                            "n": 1000,
                        },
                    },
                    "systems": {"sniper": True, "fusion": True, "slow_bull": False},
                    "improvements": [],
                },
            },
        },
    }


def _make_old_schema() -> dict:
    """旧 schema: 无 conclusion, 无 merged/systems/last_days."""
    return {
        "ts": "20260803_100000",
        "boards": {"main": {"label": "主板", "rows": 100, "stocks": 20}},
    }


# ───────────────────────── 新 schema ─────────────────────────
class TestParseNewSchema:
    def test_conclusion_summary(self):
        df = btr.parse_conclusion_summary(_make_fixture())
        assert len(df) == 4
        assert set(df["board"]) == {"main", "dual"}
        assert set(df["cut"]) == {"top5", "top10"}
        main_top5 = df[(df["board"] == "main") & (df["cut"] == "top5")].iloc[0]
        assert bool(main_top5["kept"]) is True
        assert main_top5["winrate"] == 0.91
        assert main_top5["best_horizon"] == "10d"
        dual_top10 = df[(df["board"] == "dual") & (df["cut"] == "top10")].iloc[0]
        assert bool(dual_top10["kept"]) is True

    def test_board_overview(self):
        ov = btr.parse_board_overview(_make_fixture(), "main")
        assert ov["criteria"]["min_winrate"] == 0.55
        assert ov["improvements"]
        assert "cuts" in ov
        assert ov["rows"] == 1000

    def test_cut_summary(self):
        c = btr.cut_summary(_make_fixture(), "main", "top10")
        assert c["kept"] is False
        assert c["winrate"] == 0.88

    def test_per_horizon(self):
        df = btr.parse_per_horizon(_make_fixture(), "main", "top5")
        assert list(df["horizon"]) == ["3d", "5d", "10d"]
        assert df["winrate"].isna().sum() == 0
        assert (df["base_winrate"] < df["winrate"]).all()  # 合成值刻意 winrate>base

    def test_systems(self):
        df = btr.parse_systems(_make_fixture(), "main")
        assert set(df["system"]) == {"sniper", "fusion"}
        assert len(df) == 6  # 2 系统 × 3 视界

    def test_picks(self):
        df = btr.parse_picks(_make_fixture(), "main")
        assert len(df) == 2
        assert set(df.columns) == {
            "date",
            "system",
            "rk",
            "symbol",
            "score",
            "mfe_3d",
            "mfe_5d",
            "mfe_10d",
        }
        assert df["date"].iloc[0] == "2026-07-30"
        assert pd.isna(df.loc[df["system"] == "sniper_top5", "mfe_10d"].iloc[0])

    def test_list_boards(self):
        assert btr.list_boards(_make_fixture()) == ["main", "dual"]


# ───────────────────────── 旧 schema 防御 ─────────────────────────
class TestOldSchema:
    def test_conclusion_summary_empty(self):
        assert btr.parse_conclusion_summary(_make_old_schema()).empty

    def test_board_overview_graceful(self):
        ov = btr.parse_board_overview(_make_old_schema(), "main")
        assert ov["board"] == "main"
        assert "cuts" not in ov
        assert ov.get("rows") == 100

    def test_per_horizon_empty(self):
        df = btr.parse_per_horizon(_make_old_schema(), "main", "top5")
        assert df.empty
        assert list(df.columns) == btr._PH_COLS

    def test_systems_empty(self):
        assert btr.parse_systems(_make_old_schema(), "main").empty

    def test_picks_empty(self):
        assert btr.parse_picks(_make_old_schema(), "main").empty

    def test_cut_summary_missing(self):
        assert btr.cut_summary(_make_old_schema(), "main", "top5") == {}

    def test_list_boards_old(self):
        assert btr.list_boards(_make_old_schema()) == ["main"]


# ───────────────────────── 磁盘 IO ─────────────────────────
class TestIO:
    def test_list_runs_and_load(self, tmp_path):
        run = tmp_path / "20260805_120000"
        run.mkdir()
        (run / "backtest.json").write_text(
            json.dumps(_make_fixture()), encoding="utf-8"
        )
        runs = btr.list_runs(str(tmp_path))
        assert runs and runs[0]["ts"] == "20260805_120000"
        assert "mtime" in runs[0]
        d = btr.load_run_json("20260805_120000", str(tmp_path))
        assert d and d["ts"] == "20260805_120000"

    def test_load_missing_returns_none(self, tmp_path):
        assert btr.load_run_json("nope", str(tmp_path)) is None

    def test_load_corrupt_returns_none(self, tmp_path):
        run = tmp_path / "bad"
        run.mkdir()
        (run / "backtest.json").write_text("{not json", encoding="utf-8")
        assert btr.load_run_json("bad", str(tmp_path)) is None

    def test_list_runs_empty_dir(self, tmp_path):
        assert btr.list_runs(str(tmp_path)) == []


# ───────────────────────── oos_days 单窗 run ─────────────────────────
def _make_single_window() -> dict:
    """oos_days=N 单窗 run: 窗口 label = "oos" (非 6m/3m/10d)."""
    per_h = {
        h: _horizon(0.05 + 0.01 * i, 0.81 + 0.01 * i, 500, 0.80 + 0.01 * i)
        for i, h in enumerate(["3d", "5d", "10d"])
    }
    board = {
        "label": "主板",
        "merged": {"top5": {"oos": {"oos": {"per_horizon": per_h, "kept": True}}}},
        "systems": {
            "sniper": {
                "oos": {
                    "oos": {"primary": {"per_horizon": per_h, "top_n": 5, "kept": True}}
                }
            }
        },
        "last_days": {"days": []},
    }
    return {"ts": "20260806_055631", "boards": {"main": board}}


class TestSingleWindowRun:
    def test_per_horizon(self):
        df = btr.parse_per_horizon(_make_single_window(), "main", "top5")
        assert list(df["horizon"]) == ["3d", "5d", "10d"]
        assert df["winrate"].isna().sum() == 0

    def test_systems(self):
        df = btr.parse_systems(_make_single_window(), "main")
        assert set(df["system"]) == {"sniper"}
        assert len(df) == 3  # 1 系统 × 3 视界


# ───────────────────────── 真实数据冒烟 (若存在) ─────────────────────────
def test_real_run_parses():
    from config.settings import BACKTEST_RESULT_DIR

    if not BACKTEST_RESULT_DIR.is_dir():
        pytest.skip("BACKTEST_RESULT_DIR 不存在")
    base = str(BACKTEST_RESULT_DIR)
    d = btr.load_run_json("20260805_224841", base)
    if d is not None:
        assert not btr.parse_conclusion_summary(d).empty
    runs = [r["ts"] for r in btr.list_runs(base)][:5]
    parsed = 0
    for ts in runs:
        dd = btr.load_run_json(ts, base)
        if dd is None:
            continue
        parsed += 1
        assert not btr.parse_conclusion_summary(dd).empty
        for board in btr.list_boards(dd):
            btr.parse_board_overview(dd, board)
            btr.parse_per_horizon(dd, board, "top5")
            btr.parse_systems(dd, board)
            btr.parse_picks(dd, board)
    if parsed == 0:
        pytest.skip("BACKTEST_RESULT_DIR 无可用 run")


# ───────────────────────── 收益与风险 (净值/回撤/夏普, 新 schema) ─────────────────────────
def _make_series_fixture() -> dict:
    """merged 含 stats+series 的新 schema (10d 带曲线, 3d 不带)."""
    per_h = {
        "3d": _horizon(0.02, 0.55, 100, 0.50),
        "10d": {
            **_horizon(0.04, 0.59, 100, 0.50),
            "stats": {
                "n": 100,
                "winrate": 0.59,
                "avg": 0.04,
                "median": 0.035,
                "avg_win": 0.08,
                "avg_loss": -0.03,
                "profit_factor": 1.8,
                "max_win": 0.21,
                "max_loss": -0.12,
                "max_drawdown": -0.05,
                "annualized": 0.30,
                "sharpe": 1.2,
                "volatility": 0.25,
                "beta": 0.85,
                "alpha_annual": 0.12,
                "deepest_date": "2026-03-09",
                "recovery_days": 1,
                "underwater_days": 1,
                "method": "逐票复利",
            },
            "series": {
                "dates": ["2026-03-06", "2026-03-09", "2026-03-10"],
                "cum": [1.01, 1.005, 1.03],
                "dd": [0.0, -0.00495, 0.0],
                "ret": [0.01, -0.005, 0.025],
                "bench_cum": [1.002, 1.008, 1.005],
            },
        },
    }
    board = {
        "merged": {
            "top5": {"oos": {"6m": {"per_horizon": per_h, "kept": True}}},
            "top10": {"oos": {"6m": {"per_horizon": per_h, "kept": True}}},
        }
    }
    return {"ts": "20260905_000000", "boards": {"main": board}}


class TestEquityAndStats:
    def test_parse_equity(self):
        eq = btr.parse_equity(_make_series_fixture(), "main", "top10")
        assert set(eq) == {"10d"}  # 3d 无 series
        df = eq["10d"]
        assert list(df.columns) == ["cum", "dd", "ret", "bench"]  # bench_cum → bench 列
        assert list(df.index) == ["2026-03-06", "2026-03-09", "2026-03-10"]
        assert df["cum"].iloc[2] == pytest.approx(1.03)
        assert df["bench"].iloc[0] == pytest.approx(1.002)

    def test_parse_equity_no_bench_old_series(self):
        # series 无 bench_cum (round-1 产出) → 不加 bench 列
        d = _make_series_fixture()
        del d["boards"]["main"]["merged"]["top10"]["oos"]["6m"]["per_horizon"]["10d"][
            "series"
        ]["bench_cum"]
        df = btr.parse_equity(d, "main", "top10")["10d"]
        assert list(df.columns) == ["cum", "dd", "ret"]

    def test_parse_equity_default_cut_top10(self):
        assert "10d" in btr.parse_equity(_make_series_fixture(), "main")

    def test_parse_pick_stats(self):
        df = btr.parse_pick_stats(_make_series_fixture(), "main", "top10")
        assert list(df["horizon"]) == ["10d"]
        row = df.iloc[0]
        assert row["profit_factor"] == 1.8
        assert row["max_drawdown"] == -0.05
        assert row["sharpe"] == 1.2
        # 第二期新列
        assert row["beta"] == 0.85
        assert row["alpha_annual"] == 0.12
        assert row["deepest_date"] == "2026-03-09"
        assert row["recovery_days"] == 1
        assert row["underwater_days"] == 1

    def test_parse_diversification(self):
        d = _make_series_fixture()
        d["boards"]["main"]["merged"]["top10"]["diversification"] = {
            "avg_pairwise_corr": 0.45,
            "n_dates": 120,
            "lookback": 60,
        }
        div = btr.parse_diversification(d, "main", "top10")
        assert div["avg_pairwise_corr"] == 0.45
        assert div["n_dates"] == 120
        # 无该键 (round-1 产出) → {} 防御
        assert btr.parse_diversification(_make_series_fixture(), "main", "top10") == {}

    def test_old_schema_defensive(self):
        assert btr.parse_equity(_make_old_schema(), "main") == {}
        assert btr.parse_pick_stats(_make_old_schema(), "main").empty
        assert btr.parse_diversification(_make_old_schema(), "main") == {}

    def test_single_window_label_fallback(self):
        # 单窗 run (label="oos") 走 fallback 路径, 无 series → 空不崩
        assert btr.parse_equity(_make_single_window(), "main", "top5") == {}

    def test_semi_new_schema_no_stats(self):
        # horizon 节点存在但被剥掉 stats (半新 schema) → 空防御
        d = _make_series_fixture()
        node = d["boards"]["main"]["merged"]["top10"]["oos"]["6m"]["per_horizon"]["10d"]
        del node["stats"]
        assert btr.parse_pick_stats(d, "main", "top10").empty
