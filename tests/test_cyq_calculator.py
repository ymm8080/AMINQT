"""cyq_ext 扩展导出列测试 (V3 候选列, 独立模块).

验证: 150 档分布导出的统计列存在、成本分位线单调、质量占比在 [0,1].
"""

import numpy as np
import pandas as pd

from app.pipeline1.cyq_ext import (
    TARGET_COLS,
    _compute_cyq_for_stock,
    compute_cyq_panel,
    compute_cyq_today,
)

EXTENDED_COLS = [
    "cost_10pct",
    "cost_20pct",
    "cost_30pct",
    "cost_40pct",
    "cost_60pct",
    "cost_70pct",
    "cost_80pct",
    "pct_60_con",
    "pct_80_con",
    "peak_price",
    "peak_mass",
    "chip_entropy",
    "chip_gini",
    "chip_skew_dist",
    "mass_above_close",
    "mass_above_1_1x",
    "mass_above_1_2x",
    "mass_below_0_9x",
    "resistance_dist",
    "support_dist",
    "peak_roc_5d",
    "peak_roc_20d",
]

COST_COLS = [
    "cost_5pct",
    "cost_10pct",
    "cost_15pct",
    "cost_20pct",
    "cost_30pct",
    "cost_40pct",
    "cost_50pct",
    "cost_60pct",
    "cost_70pct",
    "cost_80pct",
    "cost_85pct",
    "cost_95pct",
]


def _synthetic_kdata(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, n)).round(2)
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=n, freq="B"),
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "turnover_rate": rng.uniform(1, 8, n),
        }
    )


def test_extended_exports_present():
    out = _compute_cyq_for_stock(_synthetic_kdata())
    missing = [c for c in EXTENDED_COLS if c not in out.columns]
    assert not missing, f"缺扩展导出列: {missing}"
    assert len(out.columns) == 1 + 14 + 22  # date + 基础14 + 扩展22


def test_cost_percentiles_monotonic():
    out = _compute_cyq_for_stock(_synthetic_kdata())
    last = out.iloc[-1]
    vals = [last[c] for c in COST_COLS]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_mass_fractions_bounded():
    out = _compute_cyq_for_stock(_synthetic_kdata())
    last = out.iloc[-1]
    for c in (
        "mass_above_close",
        "mass_above_1_1x",
        "mass_above_1_2x",
        "mass_below_0_9x",
        "peak_mass",
    ):
        assert 0.0 <= last[c] <= 1.0, f"{c}={last[c]:.3f}"
    # 越高的阈值上方筹码越少
    assert (
        last["mass_above_close"] >= last["mass_above_1_1x"] >= last["mass_above_1_2x"]
    )


def test_entropy_gini_nonnegative():
    out = _compute_cyq_for_stock(_synthetic_kdata())
    last = out.iloc[-1]
    assert last["chip_entropy"] >= 0.0
    assert last["chip_gini"] >= 0.0
    assert last["peak_price"] > 0.0


def test_compute_cyq_panel_includes_extended():
    k = _synthetic_kdata()
    k["symbol"] = "000001"
    panel = compute_cyq_panel(
        k[["symbol", "date", "open", "close", "high", "low", "turnover_rate"]]
    )
    assert "peak_price" in panel.columns
    assert len(panel) == len(k) - 60  # 预热期跳过 120/2 天


def test_peak_roc_matches_peak_price():
    out = _compute_cyq_for_stock(_synthetic_kdata())
    last = out.iloc[-1]
    exp5 = last["peak_price"] / out["peak_price"].iloc[-6] - 1
    exp20 = last["peak_price"] / out["peak_price"].iloc[-21] - 1
    assert abs(last["peak_roc_5d"] - exp5) < 1e-9
    assert abs(last["peak_roc_20d"] - exp20) < 1e-9


def test_compute_cyq_today_returns_targets():
    k = _synthetic_kdata(250)
    k["symbol"] = "000001"
    # hist 需带 peak_price (回填后的面板即有); 用全量 cyq 导出回填
    cyq = _compute_cyq_for_stock(k)
    merged = k.merge(cyq[["date", "peak_price"]], on="date", how="left")
    hist = merged.iloc[:-1]
    today = merged.iloc[-1:].copy()
    out = compute_cyq_today(hist, today)
    assert set(out.columns) == {"symbol"} | set(TARGET_COLS)
    assert len(out) == 1
    row = out.iloc[0]
    for c in TARGET_COLS:
        assert pd.notna(row[c]), f"{c} 为 NaN"
    # 今日 peak_price 应与全量导出的末日一致
    assert abs(row["peak_price"] - cyq.iloc[-1]["peak_price"]) < 1e-6


def test_dim21_chip_tushare_keep_columns():
    """V3 删列 (2026-08-02): dim21 只产出 KEEP 派生列, 不产出已删列.

    输入为裸名 14 基础 CYQ 列 (has_calc 路径), 2 symbol × 25 交易日 + industry.
    """
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-01", periods=25)
    rows = []
    for sym in ("000001", "000002"):
        close = 10 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
        cost50 = close * rng.uniform(0.9, 1.1, len(dates))
        for i, d in enumerate(dates):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "industry": "bank",
                    "close": close[i],
                    "winner_ratio": rng.uniform(0.3, 0.9),
                    "avg_cost": close[i] * rng.uniform(0.95, 1.05),
                    "pct_70_low": close[i] * 0.7,
                    "pct_70_high": close[i] * 1.3,
                    "pct_70_con": 0.3,
                    "pct_90_low": close[i] * 0.6,
                    "pct_90_high": close[i] * 1.4,
                    "pct_90_con": 0.4,
                    "cost_5pct": close[i] * 0.6,
                    "cost_15pct": close[i] * 0.7,
                    "cost_50pct": cost50[i],
                    "cost_85pct": close[i] * 1.3,
                    "cost_95pct": close[i] * 1.5,
                    "weight_avg": close[i] * 1.05,
                }
            )
    df = pd.DataFrame(rows)

    out = FeatureEngineV35.dim21_chip_tushare(df.copy())

    keep = ["winner_ratio", "cost_bias", "conc_trend_20d", "conc_90_industry_rank"]
    for c in keep:
        assert c in out.columns, f"缺少 KEEP 列 {c}"

    # 20 日预热后末日应非 NaN
    last = out.sort_values(["symbol", "date"]).groupby("symbol").tail(1)
    for c in ["winner_ratio", "cost_bias", "conc_trend_20d", "conc_90_industry_rank"]:
        assert last[c].notna().all(), f"{c} 末日为 NaN"

    deleted = [
        "conc_90",
        "cost_spread",
        "chip_skew",
        "benefit_trend_5d",
        "conc_streak",
        "conc_streak_3d",
        "conc70_streak",
        "conc70_streak_3d",
        "conc_reversal",
        "benefit_vs_ma60",
        "benefit_dir_5d",
        "cost50_rank",
    ]
    for c in deleted:
        assert c not in out.columns, f"不应产出已删列 {c}"
