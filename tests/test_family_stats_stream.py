"""family_stats 流式统计 ≡ generate_family 物化宽帧 (2026-08-11 OOM 修复).

FeatureSelector._run_bruteforce_dedup 消费 generate_family 的输出只需
每列 nan 率 + 5000 采样行值; family_stats 按 symbol 流式累计这两样,
不物化 (N×2544) float32 宽帧 (单家族 pct_change 就 11.6GB, 15.8GB 机 OOM).

等价性铁律: family_stats 输出必须与 generate_family → 逐列统计 完全一致
(同符号内特征数学共用 _family_for_symbol, 含 inf→nan 处理), 否则 dedup
相关性/方差漂移 → 选择结果改变 → 训练质量回归.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator

# 生产 raw 列 ⊆ _eligible (float64/int64, 非 EXCLUDE, 非 label_/dim 前缀)
_RAW = ["close", "open", "high", "low", "volume"]


def _synth_df(n_dates=40, n_symbols=4, seed=7):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_symbols):
        sym = f"{600000 + s}"
        base = 10 + s
        price = base + rng.randn(n_dates).cumsum() * 0.5
        vol = rng.rand(n_dates) * 1e5
        for i in range(n_dates):
            d = pd.Timestamp("2025-01-02") + pd.Timedelta(days=i)
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "close": price[i],
                    "open": price[i] - 0.1,
                    "high": price[i] + 0.2,
                    "low": price[i] - 0.2,
                    "volume": vol[i],
                }
            )
    df = pd.DataFrame(rows)
    # 生产帧特征: 有序 (symbol,date) + RangeIndex. 造 inf 行验证 stats 的 inf→nan 与
    # generate_family 的 .replace([inf,-inf], nan) 一致.
    df["close"].iloc[5] = np.inf
    df["volume"].iloc[12] = -np.inf
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _stats_from_frame(new, sample_pos):
    """从 generate_family 物化帧抽取 每列 nan率 + 采样行值 (旧路径口径)."""
    nan_rate = {c: float(new[c].isna().mean()) for c in new.columns}
    sample_vals = {
        c: new.loc[sample_pos, c].to_numpy(dtype=np.float32) for c in new.columns
    }
    return nan_rate, sample_vals


class TestFamilyStatsStream:
    def test_matches_generate_family_all_families(self):
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        assert len(raw_cols) >= 4  # 确保 raw 非空, 特征真的有量
        sample_pos = df.sample(min(20, len(df)), random_state=42).index

        for fam in BRUTE_FAMILIES:
            # 旧路径: 物化宽帧再逐列统计
            new = gen.generate_family(df, fam, raw_cols=raw_cols, dtype="float32")
            exp_nan, exp_svals = _stats_from_frame(new, sample_pos)
            del new
            # 新路径: 流式统计
            cols, nan_rate, svals = gen.family_stats(
                df, fam, sample_pos, raw_cols=raw_cols, dtype="float32"
            )
            assert list(cols) == list(exp_nan.keys()), f"{fam}: 列清单漂移"
            for c in cols:
                assert np.isclose(nan_rate[c], exp_nan[c]), (
                    f"{fam}/{c}: nan率 {nan_rate[c]:.4f} vs {exp_nan[c]:.4f}"
                )
                np.testing.assert_array_equal(
                    svals[c], exp_svals[c], err_msg=f"{fam}/{c}: 采样行值漂移"
                )

    def test_rolling_max_names_max_and_min(self):
        # rolling_max 族同时产出 max+min 列 (列名不走 suffix) — 特例必须一致
        df = _synth_df()
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        new = gen.generate_family(df, "rolling_max", raw_cols=raw_cols)
        assert any(c.endswith("_brute_max10") for c in new.columns)
        assert any(c.endswith("_brute_min10") for c in new.columns)
        cols, _, _ = gen.family_stats(df, "rolling_max", df.index[:10], raw_cols=raw_cols)
        assert set(cols) == set(new.columns)
