"""brute 推理后注入单测 (2026-08-30 生产修复: main bundle 85/353 特征推理缺失补 0).

验证 inject_missing_brute / brute_bases:
- 数值与 BruteForceGenerator 训练口径逐元素一致 (pct/mom 手工公式)
- float32 基列可注入 (raw_cols 显式传基列, 绕过 _eligible float64 过滤)
- 基列不在场 → 如实上报仍缺失名单, 不静默
- 非 _brute_ 缺失列不归本函数管 (predictor 补 0 语义不变)
- 索引/行序保持, 无关列不动
"""

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.feature_selector import brute_bases, inject_missing_brute


def _frame(dtype: str = "float64") -> pd.DataFrame:
    rng = np.random.RandomState(42)
    days = 70
    frames = []
    for si, sym in enumerate(("000001", "600000")):
        vals = 10 + si + np.cumsum(rng.normal(0, 0.1, days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": pd.date_range("2026-01-01", periods=days),
                    "A04": vals.astype(dtype),
                    "keep": np.arange(days, dtype=dtype),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_brute_bases_dedup_sorted():
    assert brute_bases(
        ["A04_brute_pct1", "VAR5_brute_mom40", "A04_brute_pct5", "plain"]
    ) == [
        "A04",
        "VAR5",
    ]
    assert brute_bases(None) == []
    assert brute_bases(["plain"]) == []


def test_pct_mom_values_match_training_math():
    df = _frame("float64")
    before = df.copy()
    still = inject_missing_brute(df, ["A04_brute_pct5", "A04_brute_mom40"])
    assert still == []
    assert "A04_brute_pct5" in df.columns and "A04_brute_mom40" in df.columns

    g = df[df["symbol"] == "000001"].sort_values("date")
    s = g["A04"].to_numpy(dtype=float)
    n = len(s)
    # 训练口径: o[w:] = (s[w:] - s[:-w]) / |s[:-w]| * 100; mom: s[w:] / |s[:-w]|
    expect_pct = np.full(n, np.nan)
    expect_pct[5:] = (s[5:] - s[:-5]) / np.abs(s[:-5]) * 100
    expect_mom = np.full(n, np.nan)
    expect_mom[40:] = s[40:] / np.abs(s[:-40])
    np.testing.assert_allclose(
        g["A04_brute_pct5"].to_numpy(dtype=float), expect_pct, equal_nan=True
    )
    np.testing.assert_allclose(
        g["A04_brute_mom40"].to_numpy(dtype=float), expect_mom, equal_nan=True
    )
    # 无关列不动
    assert df["keep"].equals(before["keep"])
    assert df["A04"].equals(before["A04"])


def test_float32_base_injectable():
    df = _frame("float32")
    still = inject_missing_brute(df, ["A04_brute_pct1", "A04_brute_ma5"])
    assert still == []
    assert df["A04_brute_pct1"].notna().sum() > 0


def test_missing_base_reported_not_silent():
    df = _frame("float64")
    still = inject_missing_brute(df, ["ZZZ_brute_pct5"])
    assert still == ["ZZZ_brute_pct5"]
    assert "ZZZ_brute_pct5" not in df.columns


def test_non_brute_missing_ignored():
    df = _frame("float64")
    assert inject_missing_brute(df, ["plain_missing_col"]) == []


def test_noop_when_all_present():
    df = _frame("float64")
    df["A04_brute_pct5"] = 1.0
    assert inject_missing_brute(df, ["A04_brute_pct5"]) == []
    assert (df["A04_brute_pct5"] == 1.0).all()


def test_custom_index_row_alignment():
    df = _frame("float64")
    df.index = pd.Index(np.arange(len(df)) * 2 + 7)  # 非.RangeIndex
    before_idx = df.index.copy()
    still = inject_missing_brute(df, ["A04_brute_pct5"])
    assert still == []
    assert df.index.equals(before_idx)
    g = df[df["symbol"] == "600000"].sort_values("date")
    s = g["A04"].to_numpy(dtype=float)
    expect = (s[5:] - s[:-5]) / np.abs(s[:-5]) * 100
    np.testing.assert_allclose(
        g["A04_brute_pct5"].to_numpy(dtype=float)[5:], expect, equal_nan=True
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
