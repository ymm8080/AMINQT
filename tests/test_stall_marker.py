"""Tests for scripts/_stall_marker 横盘提示 (legacy+parallel+genious 三交付).

0922 中文口径: 横盘 = 近 10 日涨幅 < STALL_MARKER.ret_10d; 冷静市 = 当日
base_rate < base_rate_max (决定性条件) → 横盘提示 = "近10日未涨·冷静市".
频次条件 (近20日入选≥3) 已消融判死砍掉 — 低频/无历史仍标记, 次数列仅参考.
涨停提示 = 昨日 (T-1) 涨幅 ≥ 板块涨停阈值 → "涨停次日不追".
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
    # 低基线日 (high=close×1.01 → base_rate≈0) + 横盘 → 标记
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
    assert out.loc[0, "横盘提示"] == "近10日未涨·冷静市"
    assert out.loc[0, "近20日入选次数"] == 3
    assert out.loc[0, "市场温度"] < 0.732


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
    assert out.loc[0, "横盘提示"] == ""
    assert out.loc[0, "市场温度"] > 0.732


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
    assert out.loc[0, "横盘提示"] == ""
    assert out.loc[0, "近10日涨幅"] >= 0.02


def test_low_frequency_still_flagged(tmp_path):
    # 0922 频次条件已砍: 仅 2 次历史入选, 横盘+冷静市 → 仍标记 (次数列只作参考)
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    _hist(tmp_path, {"20260203": ["300911"], "20260204": ["300911"]})  # 仅 2 次
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "横盘提示"] == "近10日未涨·冷静市"
    assert out.loc[0, "近20日入选次数"] == 2


def test_no_history_still_flagged(tmp_path):
    # 0922 频次条件已砍: 无历史入选 → 仍标记
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "横盘提示"] == "近10日未涨·冷静市"
    assert out.loc[0, "近20日入选次数"] == 0


def test_panel_missing_row_not_flagged(tmp_path):
    # 面板只有 300911, 清单含另一只 → 近10日涨幅 NaN → 不标
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
    assert out.loc[0, "横盘提示"] == ""
    assert pd.isna(out.loc[0, "近10日涨幅"])


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
    # 高基线日 (base_rate>0.732) → 参与建议: 建议轻仓/降参与
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]}, high_factor=1.05)
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "参与建议"] != ""
    assert "轻仓" in out.loc[0, "参与建议"]


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
    assert out.loc[0, "参与建议"] != ""
    assert "正常参与" in out.loc[0, "参与建议"]


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
    assert out.loc[0, "涨停提示"] == "涨停次日不追"
    assert out.loc[0, "昨日涨幅"] > 0.195


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
    assert out.loc[out["symbol"] == "600001", "涨停提示"].iloc[0] == "涨停次日不追"
    assert out.loc[out["symbol"] == "600002", "涨停提示"].iloc[0] == ""


def test_limit_up_absent_flag_empty(tmp_path):
    # 昨日无涨停 → 涨停提示 空
    panel = _panel(tmp_path, {"300911": [10.0] * 38 + [10.1]})
    out = stall_marker(
        _picks(["300911"]),
        TRADE_DATE,
        "legacy_stocklist_",
        hist_dir=str(tmp_path),
        panel_path=panel,
    )
    assert out.loc[0, "涨停提示"] == ""


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
    assert out.loc[0, "涨停提示"] == ""  # 10.5% < 双创 19.5% 阈值


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
    assert out.loc[0, "涨停提示"] == "涨停次日不追"
