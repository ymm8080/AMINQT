# -*- coding: utf-8 -*-
"""ChipDistribution 固定网格因果性单测 (0912 前视修复).

背景: 旧网格 = linspace(全帧 low.min*0.9, high.max*1.1) → 未来价格拉伸网格,
历史行获利盘被改写 (合成验证漂移最高 100pp)。修复 = 固定绝对对数网格
(0.5~6000, 与数据/窗口/股票无关)。

断言:
1. 前缀因果: 前缀 build vs 全量 build → 前缀行 A01/A02/A03/获利盘 逐位相等;
2. 未来扰动不改历史: 扰动后 60% 行的价格, 前 60% 行获利盘不变 (dim09 层同断言);
3. 值域与方向锚点: 获利盘 ∈ [0,100]; 稳定上涨末端 > 90。
"""

import unittest

import numpy as np
import pandas as pd

from app.indicators.chip_distribution import ChipDistribution

CHIP_COLS = ("A01", "A02", "A03", "获利盘")


def _synth(
    n: int = 120,
    p0: float = 10.0,
    drift: float = 0.002,
    vol: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = p0 * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    o = close * np.exp(rng.normal(0, 0.004, n))
    s = np.abs(rng.normal(0, 0.008, n))
    return pd.DataFrame(
        {
            "open": o,
            "high": np.maximum(o, close) * (1 + s),
            "low": np.minimum(o, close) * (1 - s),
            "close": close,
            "volume": rng.uniform(0.5e6, 2.0e6, n),
        }
    )


class TestChipGridCausality(unittest.TestCase):
    def test_prefix_build_bit_identical(self):
        df = _synth()
        cut = len(df) // 2
        full = ChipDistribution().build(df.copy(), float_shares=5e8)
        pref = ChipDistribution().build(df.iloc[:cut].copy(), float_shares=5e8)
        for c in CHIP_COLS:
            np.testing.assert_array_equal(full[c].to_numpy()[:cut], pref[c].to_numpy())

    def test_future_perturbation_keeps_history(self):
        df = _synth()
        cut = len(df) // 2
        base = (
            ChipDistribution().build(df.copy(), float_shares=5e8)["获利盘"].to_numpy()
        )
        pert = df.copy()
        pert.loc[pert.index >= cut, ["close", "high"]] *= 1.5
        after = ChipDistribution().build(pert, float_shares=5e8)["获利盘"].to_numpy()
        np.testing.assert_array_equal(base[:cut], after[:cut])

    def test_bounds_and_rally_anchor(self):
        df = _synth()
        out = ChipDistribution().build(df.copy(), float_shares=5e8)
        self.assertTrue(out["获利盘"].between(0, 100).all())
        n = 120
        close = np.linspace(100, 200, n)
        rally = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(n, 1e7),
            }
        )
        r = ChipDistribution().build(rally, float_shares=1e8)
        self.assertGreater(r["获利盘"].iloc[-1], 90)


if __name__ == "__main__":
    unittest.main()
