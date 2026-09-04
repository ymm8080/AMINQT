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
    check_winner_auc,
    compute_realized,
    compute_realized_mfe,
    daily_bias,
    daily_winner_auc,
    monthly_winner_auc,
    rolling_bias,
    rolling_calibration,
    rolling_calibration_mfe,
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
WELL_EVENTS = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]  # 低 prob 少涨, 高 prob 多涨 → 校准好
ANTI_EVENTS = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]  # 反向 → 校准差


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
    tab, ece = bin_calibration(
        pd.Series([0.3, 0.4, 0.5]), pd.Series([0, 1, 0]), n_bins=5
    )
    assert tab.empty and np.isnan(ece)
    # 事件单值 (全不涨) → 无法校准
    tab, ece = bin_calibration(pd.Series(WELL_PROBS * 2), pd.Series([0] * 20), n_bins=5)
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
        preds,
        realized,
        cost=0.002,
        thr=0.005,
        n_bins=1,
        window_days=2,
        min_matured_days=2,
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
        preds,
        realized,
        cost=0.002,
        thr=0.005,
        n_bins=1,
        window_days=2,
        min_matured_days=3,  # > 成熟日 → 积累期
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


# ── parallel MFE 校准 (2026-08-26, compute_realized_mfe + rolling_calibration_mfe) ──


def _panel_mfe() -> pd.DataFrame:
    """两股面板含 high_hfq: A 全勤; B 08-12 停牌 (close NaN → ffill, high 缺口不 ffill)."""
    rows = []
    for d in CAL:
        i = CAL.index(d)
        rows.append(
            {
                "symbol": "A",
                "date": pd.Timestamp(d),
                "high_hfq": 10.5 + i * 0.5,  # 恒为 close+0.5
                "close_hfq": 10.0 + i * 0.5,
            }
        )
        if d != "2026-08-12":
            rows.append(
                {
                    "symbol": "B",
                    "date": pd.Timestamp(d),
                    "high_hfq": 21.0 + i * 1.0,
                    "close_hfq": 20.0 + i * 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_compute_realized_mfe_matches_formula():
    """决策日=cal[0], horizon=3: T+1 收盘买, 峰=max(high[cal[2]..cal[4]])."""
    panel = _panel_mfe()
    # 决策日 = cal[0], horizon=10 → 需 cal[12] 不存在 → 不成熟
    assert compute_realized_mfe(panel, pd.Series([pd.Timestamp(CAL[0])])).empty

    realized = compute_realized_mfe(
        panel, pd.Series([pd.Timestamp(CAL[0])]), horizon=3, cost=0.003
    )
    assert len(realized) == 2
    a = realized[realized["symbol"] == "A"].iloc[0]
    # A: buy=cal[1]=10.5, 峰=max(11.5,12.0,12.5)=12.5 → 12.5/10.5-1-0.003
    assert a["realized_mfe"] == pytest.approx(12.5 / 10.5 - 1.0 - 0.003)
    # B: buy=cal[1]=21.0, 峰=max(23.0,24.0,25.0)=25.0 → 25/21-1-0.003
    b = realized[realized["symbol"] == "B"].iloc[0]
    assert b["realized_mfe"] == pytest.approx(25.0 / 21.0 - 1.0 - 0.003)


def test_compute_realized_mfe_suspension_ffill():
    """决策日=cal[1] (08-11), horizon=1: T+1=cal[2]=08-12 停牌 → B close ffill 到 21.0."""
    panel = _panel_mfe()
    realized = compute_realized_mfe(
        panel, pd.Series([pd.Timestamp(CAL[1])]), horizon=1, cost=0.0
    )
    b = realized[realized["symbol"] == "B"].iloc[0]
    # B: buy=ffill(21.0), 峰=high[cal[3]]=24.0 (停牌日 high 缺失不参与 max)
    assert b["realized_mfe"] == pytest.approx(24.0 / 21.0 - 1.0)


def test_rolling_calibration_mfe_event_and_ece():
    """事件 = net MFE > thr (无 cost 还原); 过高估 prob → ECE = 高估幅度."""
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "board": ["dual", "dual"],
            "pred_prob_10d": [0.8, 0.6],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10"), pd.Timestamp("2026-08-11")],
            "symbol": ["A", "A"],
            "realized_mfe": [0.061, 0.059],  # >0.06 事件; <0.06 不事件
        }
    )
    cal = rolling_calibration_mfe(
        preds, realized, thr=0.06, n_bins=1, window_days=2, min_matured_days=2
    )
    assert len(cal) == 1
    assert cal["n_rows"].iloc[0] == 2
    # 单桶: mean_prob 0.7, realized 0.5 → ECE 0.2 (过信)
    assert cal["ece"].iloc[0] == pytest.approx(0.2)


def test_rolling_calibration_mfe_accumulation():
    preds = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")],
            "symbol": ["A"],
            "board": [
                "dual",
            ],
            "pred_prob_10d": [0.5],
        }
    )
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-08-10")],
            "symbol": ["A"],
            "realized_mfe": [0.07],
        }
    )
    cal = rolling_calibration_mfe(preds, realized, thr=0.06, min_matured_days=3)
    assert len(cal) == 1
    assert cal["ece"].iloc[0] is None and cal["n_rows"].iloc[0] == 0


# ── 排名键赢家判别 AUC (2026-09-03, winner-leak 复盘后新增) ──


def _wauc_base_preds(date, symbols, board, preds_):
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * len(symbols),
            "symbol": symbols,
            "board": [board] * len(symbols),
            "pred_ret_10d": preds_,
        }
    )


def test_daily_winner_auc_mann_whitney_values():
    """赢家 pred 最高 → AUC 1.0; 居中 → 0.5; 最低 → 0.0 (全池 Mann-Whitney)."""
    date = "2026-08-10"
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * 3,
            "symbol": ["W", "M", "L"],
            "realized_net": [0.06, -0.01, -0.02],  # 仅 W ≥ 0.05 → 赢家
        }
    )
    # 赢家 pred 最高: ranks L=1, M=2, W=3 → (3 - 1) / 2 = 1.0
    preds = _wauc_base_preds(date, ["W", "M", "L"], "main", [0.9, 0.5, 0.1])
    out = daily_winner_auc(preds, realized)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["board"] == "main" and r["n"] == 3 and r["winners"] == 1
    assert r["density"] == pytest.approx(1 / 3)
    assert r["auc"] == pytest.approx(1.0)

    # 赢家 pred 居中: ranks L=1, W=2, M=3 → (2 - 1) / 2 = 0.5
    preds = _wauc_base_preds(date, ["W", "M", "L"], "main", [0.5, 0.9, 0.1])
    assert daily_winner_auc(preds, realized)["auc"].iloc[0] == pytest.approx(0.5)

    # 赢家 pred 最低: ranks W=1 → (1 - 1) / 2 = 0.0
    preds = _wauc_base_preds(date, ["W", "M", "L"], "main", [0.1, 0.5, 0.9])
    assert daily_winner_auc(preds, realized)["auc"].iloc[0] == pytest.approx(0.0)


def test_daily_winner_auc_no_winner_day_kept_nan():
    """全输日 → auc NaN 但行保留, density=0; 无赢家/无非赢家都不可判别."""
    date = "2026-08-10"
    preds = _wauc_base_preds(date, ["A", "B", "C"], "main", [0.9, 0.5, 0.1])
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * 3,
            "symbol": ["A", "B", "C"],
            "realized_net": [0.04, -0.01, -0.02],  # 全 < 0.05 → 无赢家
        }
    )
    out = daily_winner_auc(preds, realized, win_t=0.05)
    assert len(out) == 1
    assert np.isnan(out["auc"].iloc[0])
    assert out["winners"].iloc[0] == 0 and out["density"].iloc[0] == 0


def test_daily_winner_auc_inner_join_and_dropna():
    """realized 无行的票不参与 (inner); pred_ret_10d NaN 不参与."""
    date = "2026-08-10"
    preds = _wauc_base_preds(date, ["A", "B", "C", "Z"], "main", [0.9, np.nan, 0.5, 0.5])
    realized = pd.DataFrame(
        {
            "date": [pd.Timestamp(date)] * 2,
            "symbol": ["A", "C"],
            "realized_net": [0.06, -0.01],
        }
    )
    out = daily_winner_auc(preds, realized)
    assert out["n"].iloc[0] == 2  # B pred NaN, Z 无 realized → 只剩 A/C
    assert out["winners"].iloc[0] == 1
    assert out["density"].iloc[0] == pytest.approx(0.5)
    # 幸存 2 票 (A 赢 C 输): ranks C=1, A=2 → (2-1)/(1*1) = 1.0
    assert out["auc"].iloc[0] == pytest.approx(1.0)

    # 全空合并 → 空表, 列齐全
    realized_none = realized[realized["symbol"] == "X"]
    out = daily_winner_auc(preds, realized_none)
    assert out.empty
    assert list(out.columns) == ["date", "board", "n", "winners", "density", "auc"]


def test_monthly_winner_auc_median_and_maturity():
    """月度 = 中位 AUC; 成熟 AUC 日 <min_days → mature=False 不参与判定."""
    dates_jul = [pd.Timestamp(f"2026-07-{d:02d}") for d in range(1, 11)]  # 10 日
    dates_aug = [pd.Timestamp(f"2026-08-{d:02d}") for d in range(3, 13)]  # 10 日
    daily = pd.DataFrame(
        {
            "date": dates_jul + dates_aug,
            "board": ["main"] * 20,
            "n": [10] * 20,
            "winners": [3] * 20,
            "density": [0.3] * 20,
            "auc": [0.6] * 5 + [np.nan] * 2 + [0.7] * 3 + [0.52] * 9 + [np.nan],
        }
    )
    m = monthly_winner_auc(daily, min_days=8)
    assert len(m) == 2
    jul = m[m["month"] == "2026-07"].iloc[0]
    assert jul["n_days"] == 10 and jul["n_auc_days"] == 8
    assert jul["mature"]
    assert jul["auc"] == pytest.approx(float(np.median([0.6] * 5 + [0.7] * 3)))
    aug = m[m["month"] == "2026-08"].iloc[0]
    assert aug["n_auc_days"] == 9 and aug["mature"]
    assert aug["auc"] == pytest.approx(0.52)

    # min_days=9 → 7 月只有 8 个 AUC 日 → 不成熟
    m = monthly_winner_auc(daily, min_days=9)
    assert not m[m["month"] == "2026-07"]["mature"].iloc[0]


def test_monthly_winner_auc_per_board():
    """main/dual 同月各出一行, 互不混算."""
    date = pd.Timestamp("2026-08-10")
    daily = pd.DataFrame(
        {
            "date": [date] * 2,
            "board": ["main", "dual"],
            "n": [10, 12],
            "winners": [3, 4],
            "density": [0.3, 1 / 3],
            "auc": [0.61, 0.49],
        }
    )
    m = monthly_winner_auc(daily, min_days=1)
    assert set(m["board"]) == {"main", "dual"}
    assert m[m["board"] == "main"]["auc"].iloc[0] == pytest.approx(0.61)
    assert m[m["board"] == "dual"]["auc"].iloc[0] == pytest.approx(0.49)


def _wauc_monthly(board, months, aucs, n_auc_days=None, mature=None):
    n = len(months)
    if n_auc_days is None:
        n_auc_days = [18] * n
    if mature is None:
        mature = [True] * n
    return pd.DataFrame(
        {
            "board": [board] * n,
            "month": months,
            "n_days": [20] * n,
            "n_auc_days": n_auc_days,
            "auc": aucs,
            "mature": mature,
        }
    )


def test_check_winner_auc_two_consecutive_months_alert():
    m = _wauc_monthly("main", ["2026-06", "2026-07", "2026-08"], [0.60, 0.53, 0.52])
    alerts = check_winner_auc(m, threshold=0.55, consecutive_months=2)
    assert len(alerts) == 1
    assert alerts[0]["board"] == "main"
    assert alerts[0]["months"] == ["2026-07", "2026-08"]
    assert alerts[0]["aucs"] == pytest.approx([0.53, 0.52])
    assert alerts[0]["threshold"] == pytest.approx(0.55)


def test_check_winner_auc_recovery_no_alert():
    """6-7 月断裂但 8 月恢复 → 尾两月不全低 → 不告警."""
    m = _wauc_monthly("main", ["2026-06", "2026-07", "2026-08"], [0.60, 0.53, 0.57])
    assert check_winner_auc(m, threshold=0.55, consecutive_months=2) == []


def test_check_winner_auc_ignores_immature_months():
    """8 月不成熟 (AUC 日不足) → 只 1 个成熟月 → 不告警; 单月低也不告警."""
    m = _wauc_monthly(
        "main",
        ["2026-07", "2026-08"],
        [0.53, 0.50],
        n_auc_days=[18, 3],
        mature=[True, False],
    )
    assert check_winner_auc(m, threshold=0.55, consecutive_months=2) == []
    # 仅 1 个成熟月也不够
    m1 = _wauc_monthly("main", ["2026-07"], [0.53])
    assert check_winner_auc(m1, threshold=0.55, consecutive_months=2) == []


def test_check_winner_auc_per_board_independent():
    """main 报警不影响 dual 不报警."""
    m = pd.concat(
        [
            _wauc_monthly("main", ["2026-07", "2026-08"], [0.53, 0.52]),
            _wauc_monthly("dual", ["2026-07", "2026-08"], [0.58, 0.61]),
        ]
    )
    alerts = check_winner_auc(m, threshold=0.55, consecutive_months=2)
    assert [a["board"] for a in alerts] == ["main"]
