"""Tests for scripts/_stall_marker 滞涨标记 (2026-08-19 用户方案: legacy+parallel 双交付).

入选 = 今日交付清单股; 滞涨 = 近 10 日涨幅 < STALL_MARKER.ret_10d;
高频 = 近 20 个交付交易日入选 ≥ 3 次; 市场 = 当日 base_rate < base_rate_max
(低基线日, _diag_stall_regime 定案) → stall_flag = "洗盘待爆发".
"""

import pandas as pd

from scripts._stall_marker import _history_counts, stall_marker

TRADE_DATE = "20260205"


def _panel(tmp_path, close_map, high_factor=1.01, board="GEM"):
    """close_map: {symbol: [39 日收盘价]} → tmp parquet (39 工作日, 末日=2026-02-05).

    high_factor: high = close × factor → 高基线日 (factor 大) mfe_3d 达标率高.
    """
    rows = []
    dates = pd.date_range(end=TRADE_DATE, periods=39, freq="B")
    for sym, closes in close_map.items():
        for dt, c in zip(dates, closes):
            rows.append(
                {
                    "symbol": sym,
                    "date": dt,
                    "close_hfq": c,
                    "high_hfq": c * high_factor,
                    "amount": 1e8,
                    "board": board,
                }
            )
    fp = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(fp, index=False)
    return fp


def _hist(tmp_path, files, prefix="legacy_stocklist_"):
    """files: {date8: [symbols]} → 历史交付 CSV."""
    for date8, syms in files.items():
        pd.DataFrame({"symbol": syms}).to_csv(
            tmp_path / f"{prefix}{date8}.csv", index=False
        )


def _picks(symbols):
    return pd.DataFrame({"symbol": symbols})


def test_stall_flagged(tmp_path):
    # 低基线日 (high=close×1.01 → base_rate≈0) + 滞涨 + 高频 → 标记
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})  # 近 10 日 +1%
    _hist(
        tmp_path,
        {"20260202": ["300911"], "20260203": ["300911"], "20260204": ["300911"]},
    )
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == "洗盘待爆发"
    assert out.loc[0, "sel_20d"] == 3
    assert out.loc[0, "market_base_rate"] < 0.732


def test_high_base_rate_not_flagged(tmp_path):
    # 高基线日 (high=close×1.05 → base_rate≈1) → 不标记 (市场条件决定性)
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]}, high_factor=1.05)
    _hist(
        tmp_path,
        {"20260202": ["300911"], "20260203": ["300911"], "20260204": ["300911"]},
    )
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == ""
    assert out.loc[0, "market_base_rate"] > 0.732


def test_risen_not_flagged(tmp_path):
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.5]})  # 近 10 日 +5% (已涨)
    _hist(
        tmp_path,
        {"20260202": ["300911"], "20260203": ["300911"], "20260204": ["300911"]},
    )
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == ""
    assert out.loc[0, "ret_10d"] >= 0.02


def test_low_frequency_not_flagged(tmp_path):
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    _hist(tmp_path, {"20260203": ["300911"], "20260204": ["300911"]})  # 仅 2 次
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == ""
    assert out.loc[0, "sel_20d"] == 2


def test_no_history_not_flagged(tmp_path):
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == ""
    assert out.loc[0, "sel_20d"] == 0


def test_panel_missing_row_not_flagged(tmp_path):
    # 面板只有 300911, 清单含另一只 → ret_10d NaN → 不标
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    _hist(
        tmp_path,
        {"20260202": ["300999"], "20260203": ["300999"], "20260204": ["300999"]},
    )
    out = stall_marker(
        _picks(["300999"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "stall_flag"] == ""
    assert pd.isna(out.loc[0, "ret_10d"])


def test_parallel_prefix_isolated(tmp_path):
    _hist(tmp_path, {"20260204": ["300911"]}, prefix="legacy_stocklist_")
    _hist(tmp_path, {"20260204": ["300911"]}, prefix="parallel_shortlist_")
    counts = _history_counts(str(tmp_path), TRADE_DATE, "legacy_stocklist_", 20)
    assert counts.get("300911") == 1  # 只统计 legacy 前缀
    counts_p = _history_counts(str(tmp_path), TRADE_DATE, "parallel_shortlist_", 20)
    assert counts_p.get("300911") == 1


def test_history_after_trade_date_excluded(tmp_path):
    _hist(tmp_path, {"20260206": ["300911"]})  # 晚于 trade_date → 不算
    counts = _history_counts(str(tmp_path), TRADE_DATE, "legacy_stocklist_", 20)
    assert "300911" not in counts


def test_history_window_limit(tmp_path):
    _hist(tmp_path, {f"202601{i:02d}": ["300911"] for i in range(1, 15)})  # 14 个历史日
    counts = _history_counts(str(tmp_path), TRADE_DATE, "legacy_stocklist_", 3)
    assert counts["300911"] == 3  # 只取最近 3 个交付交易日


def test_advice_high_base_rate(tmp_path):
    # 高基线日 (base_rate>0.732) → 参与度提示: 建议降低参与度
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]}, high_factor=1.05)
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "advice"] != ""
    assert "降低参与度" in out.loc[0, "advice"]


def test_advice_low_base_rate(tmp_path):
    # 低基线日 (base_rate<0.732) → 正常参与
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "advice"] != ""
    assert "正常参与" in out.loc[0, "advice"]


def test_limit_up_flagged(tmp_path):
    # 双创股 (GEM): 昨日 +23.75% ≥ 19.5% → 涨停次日不追
    panel = _panel(tmp_path, {"300911": [10.0] * 37 + [8.0, 9.9]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "limit_flag"] == "涨停次日不追"
    assert out.loc[0, "ret_1d"] > 0.195


def test_limit_up_main_threshold(tmp_path):
    # 主板 (MAIN): +10.5% 涨停, +9% 不涨停 (阈值 9.5%)
    panel = _panel(
        tmp_path,
        {"600001": [10.0] * 37 + [8.0, 8.84], "600002": [10.0] * 37 + [8.0, 8.72]},
        board="MAIN",
    )
    out = stall_marker(
        _picks(["600001", "600002"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[out["symbol"] == "600001", "limit_flag"].iloc[0] == "涨停次日不追"
    assert out.loc[out["symbol"] == "600002", "limit_flag"].iloc[0] == ""


def test_limit_up_absent_flag_empty(tmp_path):
    # 昨日无涨停 → limit_flag 空
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "limit_flag"] == ""


def test_limit_up_lowercase_board_threshold(tmp_path):
    # 生产清单 board 值小写 (main/gem) → 阈值表键大写须转大写匹配 (08-20 修复;
    # 旧实现全 miss 落入 fillna 9.5% → 双创 10.5% 被误标涨停)
    panel = _panel(
        tmp_path, {"300911": [10.0] * 37 + [8.0, 8.84]}, board="gem"
    )  # T-1 +10.5%
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "limit_flag"] == ""  # 10.5% < 双创 19.5% 阈值


def test_limit_up_lowercase_gem_flagged(tmp_path):
    # 小写 gem + T-1 涨停 (+23.75% ≥ 19.5%) → 仍正确打标
    panel = _panel(tmp_path, {"300911": [10.0] * 37 + [8.0, 9.9]}, board="gem")
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "limit_flag"] == "涨停次日不追"
