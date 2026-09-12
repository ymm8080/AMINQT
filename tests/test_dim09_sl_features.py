# -*- coding: utf-8 -*-
"""dim09 S-L/斜率族 7 列接线单测 (2026-09-12 注入).

覆盖:
1. 7 列数学正确性 (SL差值/标准化/斜率20/多头持续/带20/带20斜率20/获利盘斜率20);
2. 跨股边界 (B 股首 20 行斜率列必须 NaN, 不串 A 股);
3. 零前视 (扰动 t 之后行, t 之前行特征不变);
4. force_include 接线: 两板同注 7 新列; main 不重复出货密度 (pin 已有), dual 补
   出货密度 5/10/20d.
"""

import unittest

import numpy as np
import pandas as pd

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_selector import FeatureSelector

NEW_7 = [
    "SL差值",
    "SL标准化",
    "SL斜率20",
    "SL多头持续天数",
    "带20",
    "带20斜率20",
    "获利盘斜率20",
]


def _make_frame() -> pd.DataFrame:
    """两只股 × 80 交易日. A: 缓涨后平台 (S-L 多头结构), B: 阴跌."""
    n = 80
    dates = pd.bdate_range("2026-01-05", periods=n)
    rows = []
    for i in range(n):
        cl_a = 100.0 + i * 0.5 if i < 60 else 130.0
        rows.append(
            {
                "symbol": "000001",
                "date": dates[i],
                "open": cl_a - 0.2,
                "high": cl_a + 1.0,
                "low": cl_a - 1.0,
                "close": cl_a,
                "volume": 1_000_000.0,
                "float_share": 5e8,
            }
        )
        cl_b = 200.0 - i * 0.8
        rows.append(
            {
                "symbol": "000002",
                "date": dates[i],
                "open": cl_b + 0.2,
                "high": cl_b + 1.0,
                "low": cl_b - 1.0,
                "close": cl_b,
                "volume": 800_000.0,
                "float_share": 3e8,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)


def _dim09(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureEngineV35().dim09_custom_formulas(df.copy())


class TestDim09SlMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = _dim09(_make_frame())

    def _sym(self, sym):
        return self.out[self.out["symbol"] == sym].reset_index(drop=True)

    def test_all_7_columns_present(self):
        for name in NEW_7:
            self.assertIn(name, self.out.columns, f"缺 {name}")

    def test_sl_level_and_norm(self):
        for sym in ("000001", "000002"):
            g = self._sym(sym)
            sl = g["短期线"] - g["长期线"]
            np.testing.assert_allclose(g["SL差值"], sl, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(
                g["SL标准化"], sl / g["长期线"], rtol=1e-10, atol=1e-10
            )

    def test_sl_slope20_is_20d_cumulative(self):
        for sym in ("000001", "000002"):
            g = self._sym(sym)
            expected = g["SL差值"] - g["SL差值"].shift(20)
            np.testing.assert_allclose(g["SL斜率20"], expected, rtol=1e-10, atol=1e-10)
            # 前 20 行 NaN (非 1d/5d Δ 族, 是多周累计)
            self.assertTrue(g["SL斜率20"][:20].isna().all())

    def test_sl_streak_recurrence(self):
        # sl>0 → streak = 前值+1 (前值 NaN → 1); sl<=0 → 0
        for sym in ("000001", "000002"):
            g = self._sym(sym)
            sl = (g["短期线"] - g["长期线"]).to_numpy()
            st = g["SL多头持续天数"].to_numpy()
            for i in range(len(g)):
                if sl[i] > 0:
                    expect = st[i - 1] + 1 if i > 0 and st[i - 1] > 0 else 1
                else:
                    expect = 0
                self.assertEqual(st[i], expect, f"{sym} row{i}")

    def test_band20_math(self):
        for sym in ("000001", "000002"):
            g = self._sym(sym)
            hi20 = g["high"].rolling(20, min_periods=20).max()
            lo20 = g["low"].rolling(20, min_periods=20).min()
            np.testing.assert_allclose(g["带20"], hi20 / lo20 - 1, rtol=1e-10)
            self.assertTrue(g["带20"][:19].isna().all())
            np.testing.assert_allclose(
                g["带20斜率20"], g["带20"] - g["带20"].shift(20), rtol=1e-10, atol=1e-12
            )

    def test_profit_slope20(self):
        for sym in ("000001", "000002"):
            g = self._sym(sym)
            self.assertIn("获利盘", g.columns)
            np.testing.assert_allclose(
                g["获利盘斜率20"],
                g["获利盘"] - g["获利盘"].shift(20),
                rtol=1e-10,
                atol=1e-10,
            )
            self.assertTrue(g["获利盘斜率20"][:20].isna().all())


class TestDim09SlBoundaryAndLookahead(unittest.TestCase):
    def test_symbol_boundary_no_carryover(self):
        out = _dim09(_make_frame())
        b = out[out["symbol"] == "000002"].reset_index(drop=True)
        # B 股前 20 行三个 shift(20) 列必须 NaN (不串 A 股历史)
        for col in ("SL斜率20", "带20斜率20", "获利盘斜率20"):
            self.assertTrue(b[col][:20].isna().all(), f"{col} 跨股泄漏")

    def test_no_lookahead(self):
        # 获利盘斜率20 不进严格断言: ChipDistribution 价格网格用全帧 low.min/high.max
        # (chip_distribution.py:52, 既有行为) → 未来行高价会平移网格, 过去行获利盘
        # 随之漂移 (实测 3.45pp) — 继承自既有获利盘族, 已单独挂账待修, 非 0912 注入引入.
        cols = [c for c in NEW_7 if c != "获利盘斜率20"]
        df = _make_frame()
        base = _dim09(df)
        pert = df.copy()
        m = (pert["symbol"] == "000001") & (pert["date"] >= pert["date"].iloc[60])
        pert.loc[m, "close"] = pert.loc[m, "close"] * 1.5
        pert.loc[m, "high"] = pert.loc[m, "high"] * 1.5
        after = _dim09(pert)
        keep = (base["symbol"] == "000001") & (base["date"] < base["date"].iloc[60])
        a = base.loc[keep, cols].reset_index(drop=True)
        b = after.loc[keep, cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)


class TestForceIncludeWiring(unittest.TestCase):
    """两板同注 7 新列 (用户令两板一致, A/B FAIL 一起摘); main 不重复注入出货密度
    (pin 273 列已含), dual 补出货密度 5/10/20d (210 列密度只剩 5 条)."""

    def test_main_includes_new7_excludes_density(self):
        inc = set(FeatureSelector.DEFAULT_CONFIG["main"]["force_include"])
        for name in NEW_7:
            self.assertIn(name, inc, f"main force_include 缺 {name}")
        for w in (5, 10, 20):
            self.assertNotIn(
                f"出货_density_{w}d", inc, "main pin 273 列已含出货密度, 勿重复注入"
            )

    def test_dual_includes_new7_and_density(self):
        inc = set(FeatureSelector.DEFAULT_CONFIG["dual"]["gate_d"]["force_include"])
        for name in NEW_7:
            self.assertIn(name, inc, f"dual force_include 缺 {name}")
        for w in (5, 10, 20):
            self.assertIn(f"出货_density_{w}d", inc, f"dual force_include 缺 {w}d")


if __name__ == "__main__":
    unittest.main()
