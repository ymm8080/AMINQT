"""Tests for app/pipeline1/drift_monitor 幅度漂移监控核心逻辑 (2026-08-17).

口径与诊断回放逐字一致: buy=决策日后第 buy_lag 交易日收盘, sell=第 sell_lag 交易日
收盘, ps/pb-1-COST, 停牌每股 ffill; 决策日可为非交易日 (周六生产跑批).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.drift_monitor import (
    accumulate_parallel_picks,
    bin_calibration,
    board_of,
    check_calibration,
    check_drift,
    compute_realized,
    daily_bias,
    rolling_bias,
    rolling_calibration,
)

# 8 个交易日, 周末跳过 (2026-08-10 周一 .. 08-19 周三)
CAL = [
    "2026-08-10",
    "2026-08-11",
    "2026-08-12",
    "2026-08-13",
    "2026-08-14",
    "2026-08-17",
    "2026-08-18",
    "2026-08-19",
]


def _panel() -> pd.DataFrame:
    """两股面板: A 全勤, B 08-12 停牌 (close NaN → ffill)."""
    rows = []
    for d in CAL:
        rows.append(
            {
                "symbol": "A",
                "date": pd.Timestamp(d),
                "close_hfq": 10.0 + CAL.index(d) * 0.5,
            }
        )
        if d != "2026-08-12":
            rows.append(
                {
                    "symbol": "B",
                    "date": pd.Timestamp(d),
                    "close_hfq": 20.0 + CAL.index(d) * 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_board_of_maps_legacy_boards():
    assert board_of("main") == "main"
    assert board_of("GEM") == "dual"
    assert board_of("STAR") == "dual"
    assert board_of("unknown") == "unknown"


def test_compute_realized_matches_formula():
    panel = _panel()
    # 决策日 = 首个交易日 (索引 0): buy=cal[1]=08-11, sell=cal[11] 不存在 → 不成熟, 丢弃
    # 决策日 = 索引 2 交易日 (08-12): buy=08-13, sell=cal[13] 不存在 → 不成熟
    realized = compute_realized(panel, pd.Series([pd.Timestamp(CAL[2])]))
    assert realized.empty

    # 决策日 = 索引 0, 但 sell_lag=3 → buy=cal[1], sell=cal[3] → 成熟
    realized = compute_realized(panel, pd.Series([pd.Timestamp(CAL[0])]), sell_lag=3)
    assert len(realized) == 2
    a = realized[realized["symbol"] == "A"].iloc[0]
    assert a["date"] == pd.Timestamp(CAL[0])
    # A: buy=cal[1]=10.5, sell=cal[3]=11.5 → 11.5/10.5-1-0.002
    assert a["realized_net"] == pytest.approx(11.5 / 10.5 - 1.0 - 0.0020)
    # B: buy=cal[1]=21.0, sell=cal[3]=23.0 → 23/21-1-0.002
    b = realized[realized["symbol"] == "B"].iloc[0]
    assert b["realized_net"] == pytest.approx(23.0 / 21.0 - 1.0 - 0.0020)


def test_compute_realized_suspension_ffill():
    panel = _panel()
    # 决策日 = 08-10, sell_lag=2: buy=cal[1]=08-11, sell=cal[2]=08-12
    # B 在 08-12 停牌 → ffill 到 08-11 的 close (21.0) → 平盘
    realized = compute_realized(panel, pd.Series([pd.Timestamp(CAL[0])]), sell_lag=2)
    b = realized[realized["symbol"] == "B"].iloc[0]
    assert b["realized_net"] == pytest.approx(21.0 / 21.0 - 1.0 - 0.0020)


def test_compute_realized_weekend_decision_date():
    """决策日 08-15 (周六) → 前最近交易日 08-14 (索引 4); buy=cal[5]=08-17 (12.5),
    sell=cal[6]=08-18 (13.0, sell_lag 为绝对偏移=2) → 13.0/12.5-1-0.002."""
    panel = _panel()
    realized = compute_realized(
        panel, pd.Series([pd.Timestamp("2026-08-15")]), sell_lag=2
    )
    assert len(realized) == 2
    a = realized[realized["symbol"] == "A"].iloc[0]
    assert a["date"] == pd.Timestamp("2026-08-15")
    assert a["realized_net"] == pytest.approx(13.0 / 12.5 - 1.0 - 0.0020)


def test_compute_realized_insufficient_history_dropped():
    """决策日 08-19 (最后交易日) + sell_lag=11 → i+11 越界 → 无成熟行."""
    panel = _panel()
    realized = compute_realized(panel, pd.Series([pd.Timestamp(CAL[-1])]), sell_lag=11)
    assert realized.empty


def test_daily_bias_grouping_and_values():
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")] * 4 + [pd.Timestamp("2026-08-11")] * 2,
            "symbol": ["A", "B", "C", "D", "A", "B"],
            "board": ["main", "main", "dual", "dual", "main", "main"],
            "pred_ret_10d": [0.05, 0.03, 0.04, 0.02, 0.06, 0.04],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")] * 4 + [pd.Timestamp("2026-08-11")] * 2,
            "symbol": ["A", "B", "C", "D", "A", "B"],
            "realized_net": [0.01, -0.01, 0.00, 0.02, 0.03, 0.01],
        }
    )
    out = daily_bias(preds, realized)
    assert list(out["board"]) == ["dual", "main", "main"]
    # 08-10 dual: pred mean 0.03, real mean 0.01 → bias +0.02
    row = out[
        (out["date"] == pd.Timestamp("2026-08-10")) & (out["board"] == "dual")
    ].iloc[0]
    assert row["n"] == 2
    assert row["pred_mean"] == pytest.approx(0.03)
    assert row["real_mean"] == pytest.approx(0.01)
    assert row["bias"] == pytest.approx(0.02)
    # 08-11 main: pred mean 0.05, real mean 0.02 → bias +0.03
    row = out[
        (out["date"] == pd.Timestamp("2026-08-11")) & (out["board"] == "main")
    ].iloc[0]
    assert row["bias"] == pytest.approx(0.03)


def test_daily_bias_inner_join_drops_unmatched():
    """realized 无行的 symbol → 不参与 (inner join)."""
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")] * 2,
            "symbol": ["A", "Z"],
            "board": ["main", "main"],
            "pred_ret_10d": [0.05, 0.99],
        }
    )
    realized = pd.DataFrame(
        {"date": [pd.Timestamp("2026-08-10")], "symbol": ["A"], "realized_net": [0.01]}
    )
    out = daily_bias(preds, realized)
    assert (
        len(out) == 1
        and out["n"].iloc[0] == 1
        and out["bias"].iloc[0] == pytest.approx(0.04)
    )


def test_rolling_bias_window_and_min_days():
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp(f"2026-08-{d:02d}") for d in range(10, 16)],
            "board": ["main"] * 6,
            "bias": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            "n": [10] * 6,
        }
    )
    # min_matured_days=6, window=10 → 全 6 日均值
    r = rolling_bias(daily, window_days=10, min_matured_days=6)
    assert len(r) == 1
    assert r["bias"].iloc[0] == pytest.approx(
        np.mean([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    )
    assert r["n_days"].iloc[0] == 6
    # min_matured_days=7 > 6 → None (积累期)
    r = rolling_bias(daily, window_days=10, min_matured_days=7)
    assert r["bias"].iloc[0] is None
    # window=3 → 尾 3 日均值
    r = rolling_bias(daily, window_days=3, min_matured_days=3)
    assert r["bias"].iloc[0] == pytest.approx(np.mean([0.04, 0.05, 0.06]))


def test_rolling_bias_per_board():
    daily = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")] * 2,
            "board": ["main", "dual"],
            "bias": [0.01, 0.05],
            "n": [10, 10],
        }
    )
    r = rolling_bias(daily, window_days=5, min_matured_days=1)
    assert set(r["board"]) == {"main", "dual"}


def test_check_drift_thresholds():
    rolling = pd.DataFrame(
        {
            "board": ["main", "dual", "other"],
            "n_days": [30, 30, 30],
            "bias": [0.05, 0.06, 0.99],
            "latest_bias": [0.05, 0.06, 0.99],
            "latest_date": [pd.Timestamp("2026-08-10")] * 3,
        }
    )
    alerts = check_drift(rolling, {"main": 0.04, "dual": 0.07})
    assert [a["board"] for a in alerts] == ["main"]  # dual 0.06 < 0.07, other 无阈值
    assert alerts[0]["bias"] == pytest.approx(0.05)
    assert alerts[0]["threshold"] == pytest.approx(0.04)


PICKS_HEADER = "date,system,rk,symbol,score,mfe_3d,mfe_5d,mfe_10d"


def _write_picks(run_root, run_ts: str, rows: list[tuple]) -> None:
    """rows = (date, symbol, mfe_10d) → run_root/<run_ts>/last_15_days_picks_dual.csv"""
    d = run_root / run_ts
    d.mkdir(parents=True)
    lines = [PICKS_HEADER]
    for i, (date, symbol, mfe10) in enumerate(rows):
        lines.append(
            f"{date},sniper,{i + 1},{symbol},{mfe10:.4f},{mfe10:.4f},{mfe10:.4f},{mfe10:.4f}"
        )
    (d / "last_15_days_picks_dual.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_accumulate_parallel_picks_dedup_keep_first(tmp_path):
    """同一 (date,symbol) 出现在多个 run 里 → 保留 ts 最早那份 (决策日当天生成)."""
    _write_picks(
        tmp_path,
        "20260817T010000",
        [("2026-08-17", "000001", 0.05), ("2026-08-16", "000002", 0.04)],
    )
    _write_picks(
        tmp_path,
        "20260818T010000",
        [("2026-08-17", "000001", 0.09), ("2026-08-18", "000003", 0.03)],
    )
    out = accumulate_parallel_picks(tmp_path)
    assert len(out) == 3  # 2026-08-17/000001 只出现一次
    assert set(out["symbol"]) == {"000001", "000002", "000003"}
    assert out["date"].tolist() == ["2026-08-17", "2026-08-16", "2026-08-18"]
    a = out[out["symbol"] == "000001"].iloc[0]
    assert a["pred_ret_10d"] == pytest.approx(0.05)  # 保留最早 run 的值
    assert a["board"] == "dual"
    assert out["symbol"].dtype == object  # 前导零不被吃掉


def test_accumulate_parallel_picks_empty(tmp_path):
    out = accumulate_parallel_picks(tmp_path)
    assert out.empty
    assert list(out.columns) == ["date", "symbol", "board", "pred_ret_10d"]


def test_accumulate_parallel_picks_ignores_main_file(tmp_path):
    """只收 *_dual.csv, main 短名单不混入."""
    d = tmp_path / "20260817T010000"
    d.mkdir(parents=True)
    (d / "last_15_days_picks.csv").write_text(
        f"{PICKS_HEADER}\n2026-08-17,sniper,1,000001,0.05,0.05,0.05,0.05\n",
        encoding="utf-8",
    )
    assert accumulate_parallel_picks(tmp_path).empty


def test_check_drift_none_bias_no_alert():
    rolling = pd.DataFrame(
        {
            "board": ["main"],
            "n_days": [5],
            "bias": [None],
            "latest_bias": [None],
            "latest_date": [pd.Timestamp("2026-08-10")],
        }
    )
    assert check_drift(rolling, {"main": 0.04}) == []


# ── p_reg 校准 (ECE) ──

WELL_PROBS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
WELL_EVENTS = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]   # 低 prob 少涨, 高 prob 多涨 → 校准好
ANTI_EVENTS = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]   # 反向 → 校准差


def test_bin_calibration_anti_worse_than_well():
    well_tab, well_ece = bin_calibration(
        pd.Series(WELL_PROBS), pd.Series(WELL_EVENTS), n_bins=5
    )
    anti_tab, anti_ece = bin_calibration(
        pd.Series(WELL_PROBS), pd.Series(ANTI_EVENTS), n_bins=5
    )
    assert len(well_tab) == 5 and len(anti_tab) == 5
    assert 0.0 <= well_ece <= 1.0 and 0.0 <= anti_ece <= 1.0
    assert anti_ece > well_ece


def test_bin_calibration_overconfident_high_ece():
    """全高 prob 但几乎不涨 → 每桶 realized≈0, mean_prob≈0.7 → ECE 大."""
    probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78]
    events = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    _, ece = bin_calibration(pd.Series(probs), pd.Series(events), n_bins=5)
    assert ece > 0.4


def test_bin_calibration_insufficient_samples():
    tab, ece = bin_calibration(pd.Series([0.3, 0.4, 0.5]), pd.Series([0, 1, 0]), n_bins=5)
    assert tab.empty and np.isnan(ece)
    # 事件单值 (全不涨) → 无法校准
    tab, ece = bin_calibration(
        pd.Series(WELL_PROBS * 2), pd.Series([0] * 20), n_bins=5
    )
    assert tab.empty and np.isnan(ece)


def test_rolling_calibration_event_threshold():
    """事件 = gross_cc = realized_net + cost > thr; 边界 0.005-0.002=0.003."""
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "board": ["main", "main"],
            "prob_up_10d": [0.3, 0.7],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "realized_net": [0.0031, 0.0029],  # gross 0.0051 → 事件; 0.0049 → 不事件
        }
    )
    cal = rolling_calibration(
        preds, realized, cost=0.002, thr=0.005, n_bins=1,
        window_days=2, min_matured_days=2,
    )
    assert len(cal) == 1
    assert cal["n_days"].iloc[0] == 2 and cal["n_rows"].iloc[0] == 2
    # 两桶合一: mean_prob 0.5, realized 0.5 → ECE 0 (事件映射正确)
    assert cal["ece"].iloc[0] == pytest.approx(0.0)


def test_rolling_calibration_min_matured_accumulation():
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "board": ["main", "main"],
            "prob_up_10d": [0.3, 0.7],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "realized_net": [0.01, -0.01],
        }
    )
    cal = rolling_calibration(
        preds, realized, cost=0.002, thr=0.005, n_bins=1,
        window_days=2, min_matured_days=3,  # > 成熟日 → 积累期
    )
    assert len(cal) == 1
    assert cal["ece"].iloc[0] is None and cal["n_rows"].iloc[0] == 0


def test_check_calibration_thresholds():
    rolling = pd.DataFrame(
        {
            "board": ["main", "dual", "other"],
            "n_days": [30, 30, 30],
            "ece": [0.05, 0.12, None],
            "bins": [[], [], []],
        }
    )
    alerts = check_calibration(rolling, {"main": 0.10, "dual": 0.10})
    assert [a["board"] for a in alerts] == ["dual"]  # main 0.05 < 0.10, other 无阈值
    assert alerts[0]["ece"] == pytest.approx(0.12)
